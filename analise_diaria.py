import os
import json
import re
import time
import pandas as pd
from openai import OpenAI


ARQUIVOS_FONTES = ["folha.csv", "cnn_brasil.csv", "bbc_brasil.csv"]
ARQUIVO_SAIDA = "analise_diaria.jsonl"
LIMITE_TEXTO_MATERIA = 2500

# Trava de segurança: quantos dias no máximo processar em uma única execução.
# Importante para cargas de corpus histórico grandes — evita que uma única
# chamada do script dispare centenas de requisições de uma vez. Rode o
# script de novo quantas vezes forem necessárias para esgotar a fila.
MAX_DIAS_POR_EXECUCAO = 30
INTERVALO_ENTRE_CHAMADAS = 2

MODELO = "gpt-4o-mini"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


ENQUADRAMENTOS = [
    "Nova descoberta científica",
    "Inovação tecnológica",
    "Saúde pública",
    "Medicina e tratamentos",
    "Meio ambiente",
    "Política científica",
    "Divulgação científica",
    "Risco e incerteza",
    "Ética científica",
    "Controvérsia científica",
    "Impacto econômico",
    "Biodiversidade e conservação",
    "Astronomia e espaço",
    "Energia",
    "Inteligência artificial",
    "Perfil de cientistas",
    "Eventos científicos",
    "Pandemias e epidemias",
    "Mudanças climáticas",
    "Outro"
]

AREAS = [
    "Ciências da Saúde",
    "Ciências Biológicas",
    "Ciências Exatas e da Terra",
    "Engenharias",
    "Ciências Agrárias",
    "Ciências Ambientais",
    "Ciências Humanas",
    "Ciências Sociais Aplicadas",
    "Linguística, Letras e Artes",
    "Interdisciplinar",
    "Outro"
]

ABRANGENCIA = [
    "Nacional",
    "Internacional",
    "Local",
    "Regional",
    "Não conclusivo"
]


def extrair_json(texto):
    texto = texto.strip()

    texto = re.sub(r"^```json", "", texto)
    texto = re.sub(r"^```", "", texto)
    texto = re.sub(r"```$", "", texto)
    texto = texto.strip()

    return json.loads(texto)


def montar_texto_materias(df_dia):
    materias = []

    for _, row in df_dia.iterrows():
        item = f"""
Veículo: {row.get("veiculo", "Desconhecido")}
Editoria: {row.get("section", "")}
Autor: {row.get("author", "")}
Título: {row.get("title", "")}

Texto:
{str(row.get("text", ""))[:LIMITE_TEXTO_MATERIA]}
"""
        materias.append(item)

    return "\n---\n".join(materias)


def analisar_dia(df_dia, data_str):
    texto_materias = montar_texto_materias(df_dia)

    prompt = f"""
Você é pesquisador em jornalismo científico e análise de cobertura jornalística.

Abaixo estão todas as matérias de ciência publicadas pela Folha no dia {data_str}.

Com base apenas nessas matérias, produza um JSON válido, sem markdown, sem
comentários e sem texto antes ou depois. Use exatamente esta estrutura:

{{
  "temas_em_destaque": ["tema 1", "tema 2", "tema 3"],
  "enquadramentos_predominantes": ["item da taxonomia", "item da taxonomia", "item da taxonomia"],
  "areas_predominantes": ["item da taxonomia", "item da taxonomia", "item da taxonomia"],
  "atores_mais_visiveis": ["nome próprio da instituição/ator 1", "nome próprio 2", "nome próprio 3"],
  "abrangencia_predominante": "item da taxonomia",
  "tendencia_do_periodo": "uma frase curta sobre a cobertura do dia"
}}

Taxonomia obrigatória para enquadramentos:
{chr(10).join("- " + item for item in ENQUADRAMENTOS)}

Taxonomia obrigatória para áreas:
{chr(10).join("- " + item for item in AREAS)}

Taxonomia obrigatória para abrangência:
{chr(10).join("- " + item for item in ABRANGENCIA)}

Regra específica para "atores_mais_visiveis":
- Liste APENAS nomes próprios específicos de instituições, empresas, agências ou
  pessoas (ex: "USP", "Fiocruz", "Instituto Butantan", "Nasa", "Universidade de
  Oxford", "OMS"). NÃO use categorias genéricas como "universidades brasileiras",
  "institutos de pesquisa", "governo" ou "agências internacionais" — se as matérias
  só mencionarem a categoria genérica sem nomear a instituição, não inclua esse
  item na lista.

Regra específica para "abrangencia_predominante":
- Refere-se à escala geográfica do FATO coberto (onde a descoberta, pesquisa,
  missão ou instituição central de cada matéria está situada/ocorre) — NÃO ao
  alcance do veículo de imprensa, que é sempre nacional por se tratar da Folha.
- Exemplo: uma matéria sobre uma missão da Nasa ou uma pesquisa em universidade
  estrangeira é "Internacional", mesmo publicada por veículo brasileiro. Uma
  matéria sobre uma política do Ministério da Saúde do Brasil é "Nacional".
- Se as matérias do dia indicarem abrangências muito diferentes entre si sem uma
  predominância clara, use "Não conclusivo" em vez de forçar uma escolha.

Regras gerais:
- Use apenas os dados das matérias abaixo.
- Não invente informações.
- Se algo não estiver claro, use "Não conclusivo".
- O JSON deve ser válido.
- Não use markdown.

Matérias do dia:

{texto_materias}
"""

    resposta = client.chat.completions.create(
        model=MODELO,
        messages=[
            {
                "role": "system",
                "content": "Você gera JSONs válidos para análise de cobertura jornalística."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.1
    )

    return extrair_json(resposta.choices[0].message.content)


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
    """Carrega e combina todas as fontes de matérias disponíveis (Folha,
    CNN Brasil, e outras que forem adicionadas depois). A análise diária
    passa a cobrir o dia inteiro em TODOS os veículos combinados — analisar
    cada veículo separadamente multiplicaria o custo de API por dia."""
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

    df["date_dt"] = pd.to_datetime(
        df["date"],
        format="%d/%m/%Y",
        errors="coerce"
    )
    df = df.dropna(subset=["date_dt"])

    if df.empty:
        raise ValueError("Nenhuma linha com data válida encontrada em folha.csv.")

    todas_as_datas = sorted(df["date_dt"].dt.normalize().unique())
    total_processado = 0
    rodada = 0

    # Loop externo: continua rodando lotes de até MAX_DIAS_POR_EXECUCAO dias
    # até esgotar a fila de pendências, tudo numa única execução do script
    # (não precisa mais rodar de novo manualmente para continuar o backfill).
    while True:
        registros = carregar_jsonl(ARQUIVO_SAIDA)
        datas_ja_analisadas = {r.get("data") for r in registros}

        dias_pendentes = [
            data_dt for data_dt in todas_as_datas
            if pd.Timestamp(data_dt).strftime("%d/%m/%Y") not in datas_ja_analisadas
        ]

        if not dias_pendentes:
            break

        rodada += 1
        dias_a_processar = dias_pendentes[:MAX_DIAS_POR_EXECUCAO]

        print(
            f"\n--- Lote {rodada}: {len(dias_a_processar)} de "
            f"{len(dias_pendentes)} dia(s) pendente(s) ---"
        )

        for i, data_dt in enumerate(dias_a_processar):
            data_str = pd.Timestamp(data_dt).strftime("%d/%m/%Y")
            df_dia = df[df["date_dt"].dt.normalize() == data_dt]

            print(
                f"[{total_processado + 1}] Analisando {data_str} "
                f"({len(df_dia)} matéria(s))..."
            )

            try:
                analise = analisar_dia(df_dia, data_str)
            except Exception as e:
                print(f"  Erro ao analisar {data_str}, pulando este dia: {e}")
                continue

            # Campos de controle, não dependem do modelo
            analise["data"] = data_str
            analise["n_materias"] = int(len(df_dia))

            # Idempotência: remove qualquer entrada antiga do mesmo dia antes
            # de adicionar a nova (protege contra reprocessamento duplicado).
            registros = [r for r in registros if r.get("data") != data_str]
            registros.append(analise)

            # Mantém o histórico ordenado por data para facilitar leitura/depuração
            registros_ordenados = sorted(
                registros,
                key=lambda r: pd.to_datetime(r.get("data"), format="%d/%m/%Y", errors="coerce")
            )

            # Salva a cada dia processado (checkpoint) — se o script for
            # interrompido no meio, o progresso já feito não se perde.
            salvar_jsonl(ARQUIVO_SAIDA, registros_ordenados)
            registros = registros_ordenados

            total_processado += 1

            if i < len(dias_a_processar) - 1:
                time.sleep(INTERVALO_ENTRE_CHAMADAS)

    if total_processado == 0:
        print(
            f"Nenhum dia pendente. Todos os {len(todas_as_datas)} dia(s) do "
            f"dataset já possuem análise em {ARQUIVO_SAIDA}."
        )
    else:
        print(
            f"\nConcluído: {total_processado} dia(s) processado(s) nesta "
            f"execução. Todos os {len(todas_as_datas)} dia(s) do dataset "
            f"agora têm análise em {ARQUIVO_SAIDA}."
        )


if __name__ == "__main__":
    main()