"""
Reclassifica APENAS a área do conhecimento de TODAS as matérias já
presentes em materias_keywords.csv, usando um prompt enxuto (só pede a
área, não o combo completo de palavras-chave/enquadramento/abrangência/
instituições/pessoas).

Por quê: a taxonomia de área foi corrigida (8 áreas oficiais do CNPq, sem
coringa "Multidisciplinar"/"Outro", com uma categoria à parte "Não
identificável" de uso restrito). Pra que TODAS as matérias tenham sido
classificadas com o MESMO critério (consistência do dataset inteiro),
sem precisar re-gastar com as outras 5 dimensões que não mudaram — muito
mais barato que reprocessar tudo com extrair_keywords.py de novo.

Precisa do texto completo de cada matéria (não fica salvo em
materias_keywords.csv), então recarrega e cruza com o corpus original.

Uso:
    python reclassificar_areas.py
"""

import os
import re
import json
import time
import pandas as pd
from openai import OpenAI

ARQUIVOS_FONTES = ["folha.csv", "cnn_brasil.csv", "bbc_brasil.csv", "g1_globo.csv"]
ARQUIVO_SAIDA = "materias_keywords.csv"
MODELO = "gpt-4o-mini"
INTERVALO_ENTRE_CHAMADAS = 1
LIMITE_TEXTO_MATERIA = 3000

# Mesma taxonomia e regra restritiva de extrair_keywords.py — mantidas
# sincronizadas manualmente entre os dois arquivos.
AREAS = [
    "Ciências Agrárias",
    "Ciências Biológicas",
    "Ciências da Saúde",
    "Ciências Exatas e da Terra",
    "Engenharias",
    "Ciências Humanas",
    "Ciências Sociais Aplicadas",
    "Linguística, Letras e Artes",
]
AREA_NAO_IDENTIFICADA = "Não identificável"
AREA_NAO_APLICAVEL = "Não aplicável"
AREAS_VALIDAS = AREAS + [AREA_NAO_IDENTIFICADA, AREA_NAO_APLICAVEL]

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def extrair_json(texto):
    texto = texto.strip()
    texto = re.sub(r"^```json", "", texto)
    texto = re.sub(r"^```", "", texto)
    texto = re.sub(r"```$", "", texto)
    return json.loads(texto.strip())


def montar_prompt_area(titulo, texto):
    return f"""Classifique a área do conhecimento predominante desta matéria jornalística de ciência.

Escolha exatamente 1 item desta taxonomia — as 8 grandes áreas oficiais do
CNPq. Não existe opção "multidisciplinar": mesmo quando a matéria tocar
mais de uma área, escolha a que for mais central ao fato principal coberto
(o que está sendo descoberto/estudado/relatado), não uma área só porque
foi mencionada de passagem.
{chr(10).join("- " + item for item in AREAS)}

Duas exceções, cada uma de uso restrito — não escolha nenhuma delas só
porque a matéria toca mais de uma área (nesse caso, ainda escolha a área
mais central entre as 8):

- "{AREA_NAO_IDENTIFICADA}": use apenas quando o texto for ambíguo, curto
  ou vago demais para saber do que trata.
- "{AREA_NAO_APLICAVEL}": use apenas quando o texto for claro, mas não
  tratar de nenhuma área científica (ex: checagem de fato/desmentido sem
  conteúdo científico central, perfil biográfico sem foco científico,
  nota de orçamento/política que só tangencia ciência, notícia geral ou
  regional que não é sobre pesquisa/conhecimento científico).

Título: {titulo}

Texto:
{texto[:LIMITE_TEXTO_MATERIA]}

Responda apenas com um objeto JSON válido, sem comentários, sem markdown,
exatamente nesta estrutura:
{{"area": "item da taxonomia"}}
"""


def classificar_area(titulo, texto):
    resposta = client.chat.completions.create(
        model=MODELO,
        messages=[
            {
                "role": "system",
                "content": "Você classifica a área do conhecimento de matérias jornalísticas de ciência."
            },
            {"role": "user", "content": montar_prompt_area(titulo, texto)}
        ],
        temperature=0.1
    )

    dados = extrair_json(resposta.choices[0].message.content)
    area = str(dados.get("area", "")).strip()

    if not area:
        return AREA_NAO_IDENTIFICADA
    if area not in AREAS_VALIDAS:
        print(f"    Aviso: área fora da taxonomia esperada: {area!r}")

    return area


def carregar_corpus_textos():
    """Carrega url -> text do corpus original — o texto completo da
    matéria não fica salvo em materias_keywords.csv, precisa recuperar
    das fontes originais para reclassificar."""
    dataframes = []

    for caminho in ARQUIVOS_FONTES:
        if not os.path.exists(caminho):
            continue
        try:
            df_fonte = pd.read_csv(caminho, encoding="utf-8-sig")
            dataframes.append(df_fonte[["url", "text"]])
        except Exception as e:
            print(f"Aviso: erro ao ler {caminho}, ignorando esta fonte: {e}")

    if not dataframes:
        raise FileNotFoundError(
            f"Nenhuma fonte de dados encontrada (procurado: {', '.join(ARQUIVOS_FONTES)})."
        )

    return pd.concat(dataframes, ignore_index=True).drop_duplicates(subset="url")


ARQUIVO_PROGRESSO = "reclassificar_areas_progresso.txt"


def salvar_com_retry(df, caminho, tentativas=5, espera_segundos=3):
    """Tenta salvar o CSV, com novas tentativas se o arquivo estiver
    temporariamente travado por outro programa (OneDrive sincronizando,
    Excel aberto, antivírus escaneando — comum no Windows). Levanta o
    erro original só se todas as tentativas falharem."""
    for tentativa in range(1, tentativas + 1):
        try:
            df.to_csv(caminho, index=False, encoding="utf-8-sig")
            return
        except PermissionError as e:
            if tentativa == tentativas:
                raise
            print(f"    Arquivo travado (tentativa {tentativa}/{tentativas}), "
                  f"aguardando {espera_segundos}s — feche o Excel/OneDrive se "
                  "estiver com o arquivo aberto...")
            time.sleep(espera_segundos)


def carregar_progresso():
    if not os.path.exists(ARQUIVO_PROGRESSO):
        return 0
    try:
        with open(ARQUIVO_PROGRESSO, "r", encoding="utf-8") as f:
            return int(f.read().strip())
    except Exception:
        return 0


def salvar_progresso(indice):
    with open(ARQUIVO_PROGRESSO, "w", encoding="utf-8") as f:
        f.write(str(indice))


def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("Configure a variável de ambiente OPENAI_API_KEY.")

    if not os.path.exists(ARQUIVO_SAIDA):
        raise FileNotFoundError(
            f"{ARQUIVO_SAIDA} não encontrado — rode extrair_keywords.py primeiro."
        )

    df_kw = pd.read_csv(ARQUIVO_SAIDA, encoding="utf-8-sig")
    df_textos = carregar_corpus_textos()

    df_kw = df_kw.merge(df_textos, on="url", how="left")

    total = len(df_kw)
    indice_inicial = carregar_progresso()

    if indice_inicial > 0:
        print(f"Retomando de onde parou: matéria {indice_inicial + 1}/{total} "
              f"({indice_inicial} já reclassificada(s) nesta rodada).")
    else:
        print(f"Reclassificando a área de {total} matéria(s) (só esse campo, prompt enxuto)...")

    for i in range(indice_inicial, total):
        titulo = df_kw.at[i, "title"] if pd.notna(df_kw.at[i, "title"]) else ""
        texto = df_kw.at[i, "text"] if pd.notna(df_kw.at[i, "text"]) else ""

        try:
            nova_area = classificar_area(titulo, str(texto))
        except Exception as e:
            print(f"  [{i + 1}/{total}] Erro, mantendo área anterior: {e}")
            time.sleep(INTERVALO_ENTRE_CHAMADAS)
            salvar_progresso(i + 1)
            continue

        df_kw.at[i, "area"] = nova_area
        print(f"  [{i + 1}/{total}] {nova_area} — {str(titulo)[:60]}")

        # Salva a cada matéria (checkpoint) — se o script for interrompido,
        # o progresso já feito não se perde. A coluna "text" (só usada
        # nesta reclassificação) é descartada antes de salvar, pra não
        # duplicar dado que já existe nos CSVs de origem.
        salvar_com_retry(df_kw.drop(columns=["text"], errors="ignore"), ARQUIVO_SAIDA)
        salvar_progresso(i + 1)

        time.sleep(INTERVALO_ENTRE_CHAMADAS)

    print(f"\nConcluído: área reclassificada para {total} matéria(s) em {ARQUIVO_SAIDA}.")

    if os.path.exists(ARQUIVO_PROGRESSO):
        os.remove(ARQUIVO_PROGRESSO)


if __name__ == "__main__":
    main()