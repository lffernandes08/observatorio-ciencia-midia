"""
Gera, por combinação (dia, veículo), uma frase curta resumindo a tendência
da cobertura daquele dia — o único campo que genuinamente precisa olhar
várias matérias JUNTAS de uma vez (não dá pra extrair de uma matéria
isolada, diferente de enquadramento/área/abrangência/instituições/pessoas,
que migraram para extrair_keywords.py e são decididos matéria por matéria).

Uso:
    python analise_diaria.py
"""

import os
import sys
import json
import re
import time
import pandas as pd
from openai import OpenAI

# O Windows costuma redirecionar stdout/stderr para o console/log com a
# codificação cp1252 (Windows-1252), que não representa vários caracteres
# — protege preventivamente contra o mesmo UnicodeEncodeError que já
# derrubou os scrapers (cnn_brasil.py, bbc_brasil.py, g1_globo.py).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


ARQUIVOS_FONTES = ["folha.csv", "cnn_brasil.csv", "bbc_brasil.csv", "g1_globo.csv"]
ARQUIVO_SAIDA = "analise_diaria.jsonl"
LIMITE_TEXTO_MATERIA = 1500  # só o suficiente para captar o assunto de cada matéria

# Trava de segurança: quantas combinações (dia, veículo) no máximo processar
# em uma única execução. Rode o script de novo quantas vezes forem
# necessárias para esgotar a fila de pendências.
MAX_COMBINACOES_POR_EXECUCAO = 30
INTERVALO_ENTRE_CHAMADAS = 2

MODELO = "gpt-4o-mini"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def extrair_json(texto):
    texto = texto.strip()
    texto = re.sub(r"^```json", "", texto)
    texto = re.sub(r"^```", "", texto)
    texto = re.sub(r"```$", "", texto)
    return json.loads(texto.strip())


def montar_texto_materias(df_combinacao):
    """Só título e um recorte curto do texto de cada matéria — o suficiente
    para a IA identificar do que se tratou o dia, sem precisar do texto
    completo (esse campo não exige a mesma profundidade de leitura que a
    classificação por matéria em extrair_keywords.py)."""
    materias = []

    for _, row in df_combinacao.iterrows():
        item = f"""Título: {row.get("title", "")}
Resumo: {str(row.get("text", ""))[:LIMITE_TEXTO_MATERIA]}"""
        materias.append(item)

    return "\n---\n".join(materias)


def gerar_tendencia(df_combinacao, data_str, veiculo):
    texto_materias = montar_texto_materias(df_combinacao)

    prompt = f"""
Você é pesquisador em jornalismo científico. Abaixo estão os títulos e
resumos das matérias de ciência publicadas por {veiculo} no dia {data_str}.

Escreva uma única frase curta (máximo 30 palavras), em português do
Brasil, descrevendo a tendência geral da cobertura desse dia — o que
predominou, sem listar cada matéria individualmente.

Use apenas os dados abaixo. Não invente informações.

Matérias do dia:
{texto_materias}

Responda apenas com um objeto JSON válido, sem markdown, exatamente nesta
estrutura:

{{
  "tendencia_do_periodo": "frase curta sobre a cobertura do dia"
}}
"""

    resposta = client.chat.completions.create(
        model=MODELO,
        messages=[
            {
                "role": "system",
                "content": "Você resume tendências de cobertura jornalística de ciência em uma frase curta."
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    dados = extrair_json(resposta.choices[0].message.content)
    return dados.get("tendencia_do_periodo", "")


def carregar_jsonl(caminho):
    if not os.path.exists(caminho):
        return []

    registros = []
    with open(caminho, "r", encoding="utf-8") as f:
        for linha in f:
            linha = linha.strip()
            if not linha:
                continue
            try:
                registros.append(json.loads(linha))
            except json.JSONDecodeError:
                print(f"Aviso: linha inválida ignorada em {caminho}")

    return registros


def salvar_jsonl(caminho, registros):
    with open(caminho, "w", encoding="utf-8") as f:
        for registro in registros:
            f.write(json.dumps(registro, ensure_ascii=False) + "\n")


def carregar_corpus():
    """Carrega e combina todas as fontes de matérias disponíveis."""
    dataframes = []

    for caminho in ARQUIVOS_FONTES:
        if not os.path.exists(caminho):
            continue
        try:
            df_fonte = pd.read_csv(caminho, encoding="utf-8-sig")
            if "veiculo" not in df_fonte.columns:
                df_fonte["veiculo"] = "Folha de S.Paulo" if caminho == "folha.csv" else "Desconhecido"
            dataframes.append(df_fonte)
        except Exception as e:
            print(f"Aviso: erro ao ler {caminho}, ignorando esta fonte: {e}")

    if not dataframes:
        raise FileNotFoundError(
            f"Nenhuma fonte de dados encontrada (procurado: {', '.join(ARQUIVOS_FONTES)})."
        )

    return pd.concat(dataframes, ignore_index=True)


def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("Configure a variável de ambiente OPENAI_API_KEY.")

    df = carregar_corpus()

    df["date_dt"] = pd.to_datetime(df["date"], format="%d/%m/%Y", errors="coerce")
    df = df.dropna(subset=["date_dt"])

    if df.empty:
        raise ValueError("Nenhuma linha com data válida encontrada nas fontes de dados.")

    if "veiculo" not in df.columns:
        df["veiculo"] = "Desconhecido"
    df["veiculo"] = df["veiculo"].fillna("Desconhecido")

    df["data_normalizada"] = df["date_dt"].dt.normalize()

    combinacoes_no_corpus = sorted(
        df.groupby(["data_normalizada", "veiculo"]).groups.keys(),
        key=lambda par: (par[0], par[1])
    )

    total_processado = 0
    rodada = 0

    while True:
        registros = carregar_jsonl(ARQUIVO_SAIDA)
        combinacoes_ja_analisadas = {
            (r.get("data"), r.get("veiculo")) for r in registros
        }

        combinacoes_pendentes = [
            (data_dt, veiculo) for data_dt, veiculo in combinacoes_no_corpus
            if (pd.Timestamp(data_dt).strftime("%d/%m/%Y"), veiculo) not in combinacoes_ja_analisadas
        ]

        if not combinacoes_pendentes:
            break

        rodada += 1
        lote = combinacoes_pendentes[:MAX_COMBINACOES_POR_EXECUCAO]

        print(
            f"\n--- Lote {rodada}: {len(lote)} de "
            f"{len(combinacoes_pendentes)} combinação(ões) dia×veículo pendente(s) ---"
        )

        for i, (data_dt, veiculo) in enumerate(lote):
            data_str = pd.Timestamp(data_dt).strftime("%d/%m/%Y")
            df_combinacao = df[
                (df["data_normalizada"] == data_dt) & (df["veiculo"] == veiculo)
            ]

            print(
                f"[{total_processado + 1}] Gerando tendência de {data_str} — {veiculo} "
                f"({len(df_combinacao)} matéria(s))..."
            )

            try:
                tendencia = gerar_tendencia(df_combinacao, data_str, veiculo)
            except Exception as e:
                print(f"  Erro ao processar {data_str} ({veiculo}), pulando esta combinação: {e}")
                continue

            registro = {
                "data": data_str,
                "veiculo": veiculo,
                "n_materias": int(len(df_combinacao)),
                "tendencia_do_periodo": tendencia,
            }

            registros = [
                r for r in registros
                if not (r.get("data") == data_str and r.get("veiculo") == veiculo)
            ]
            registros.append(registro)

            registros_ordenados = sorted(
                registros,
                key=lambda r: (
                    pd.to_datetime(r.get("data"), format="%d/%m/%Y", errors="coerce"),
                    r.get("veiculo") or ""
                )
            )

            salvar_jsonl(ARQUIVO_SAIDA, registros_ordenados)
            registros = registros_ordenados

            total_processado += 1

            if i < len(lote) - 1:
                time.sleep(INTERVALO_ENTRE_CHAMADAS)

    total_combinacoes = len(combinacoes_no_corpus)
    if total_processado == 0:
        print(
            f"Nenhuma combinação dia×veículo pendente. Todas as "
            f"{total_combinacoes} combinação(ões) do dataset já possuem "
            f"tendência gerada em {ARQUIVO_SAIDA}."
        )
    else:
        print(
            f"\nConcluído: {total_processado} combinação(ões) processada(s) "
            f"nesta execução. Todas as {total_combinacoes} combinação(ões) "
            f"do dataset agora têm tendência em {ARQUIVO_SAIDA}."
        )


if __name__ == "__main__":
    main()