import os
import json
import time
import pandas as pd
from openai import OpenAI

ARQUIVO_ENTRADA = "folha.csv"
ARQUIVO_SAIDA = "materias_keywords.csv"
MODELO = "gpt-4o-mini"
INTERVALO_ENTRE_CHAMADAS = 1

COLUNAS_SAIDA = ["date", "ano", "section", "title", "url", "palavras_chave"]

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def extrair_json(texto):
    texto = texto.strip().replace("```json", "").replace("```", "").strip()
    return json.loads(texto)


def gerar_keywords(row):
    prompt = f"""
Extraia de 5 a 8 palavras-chave semânticas da matéria abaixo.

Regras:
- Use termos substantivos e relevantes.
- Prefira conceitos, temas, instituições, doenças, tecnologias, áreas científicas e atores.
- Evite palavras genéricas como estudo, pesquisa, cientistas, Brasil, Folha.
- Padronize em português do Brasil.
- Retorne apenas JSON válido neste formato:

{{
  "palavras_chave": ["termo 1", "termo 2", "termo 3"]
}}

Título:
{row["title"]}

Texto:
{str(row["text"])[:3000]}
"""

    resposta = client.chat.completions.create(
        model=MODELO,
        messages=[
            {
                "role": "system",
                "content": "Você extrai palavras-chave para análise semântica de cobertura jornalística."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.1
    )

    dados = extrair_json(resposta.choices[0].message.content)
    return dados["palavras_chave"]


def carregar_existentes():
    """Carrega o materias_keywords.csv já existente, se houver. Usado para
    saber quais matérias (por URL) já têm palavras-chave extraídas, para não
    reprocessar (e pagar API de novo) o que já foi feito em execuções
    anteriores."""
    if not os.path.exists(ARQUIVO_SAIDA):
        return pd.DataFrame(columns=COLUNAS_SAIDA)

    try:
        existentes = pd.read_csv(ARQUIVO_SAIDA, encoding="utf-8-sig")
    except Exception as e:
        print(f"Aviso: não foi possível ler {ARQUIVO_SAIDA} existente ({e}). "
              "Começando do zero.")
        return pd.DataFrame(columns=COLUNAS_SAIDA)

    colunas_faltando = [c for c in COLUNAS_SAIDA if c not in existentes.columns]
    if colunas_faltando:
        print(f"Aviso: {ARQUIVO_SAIDA} existente está com colunas inesperadas "
              f"(faltando: {colunas_faltando}). Começando do zero.")
        return pd.DataFrame(columns=COLUNAS_SAIDA)

    return existentes


def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("Configure OPENAI_API_KEY.")

    df = pd.read_csv(ARQUIVO_ENTRADA, encoding="utf-8-sig")

    df["date_dt"] = pd.to_datetime(
        df["date"],
        format="%d/%m/%Y",
        errors="coerce"
    )
    df = df.dropna(subset=["date_dt"])
    df = df.sort_values("date_dt")
    df["Data"] = df["date_dt"].dt.strftime("%d/%m/%Y")
    df["Ano"] = df["date_dt"].dt.year

    existentes = carregar_existentes()
    urls_processadas = set(existentes["url"].dropna().astype(str))

    df_pendente = df[~df["url"].astype(str).isin(urls_processadas)]

    total_geral = len(df)
    total_pendente = len(df_pendente)
    total_ja_feito = total_geral - total_pendente

    if total_pendente == 0:
        print(
            f"Nada a processar — as {total_geral} matéria(s) do dataset já "
            f"têm palavras-chave em {ARQUIVO_SAIDA}."
        )
        return

    print(
        f"{total_pendente} matéria(s) pendente(s) de {total_geral} no total "
        f"({total_ja_feito} já processada(s) em execuções anteriores)."
    )

    resultados = existentes.to_dict("records")

    for i, (_, row) in enumerate(df_pendente.iterrows()):
        print(f"Processando matéria {i + 1}/{total_pendente}...")

        try:
            keywords = gerar_keywords(row)
        except Exception as e:
            print("  Erro:", e)
            keywords = []

        resultados.append({
            "date": row["Data"],
            "ano": row["Ano"],
            "section": row.get("section", ""),
            "title": row.get("title", ""),
            "url": row.get("url", ""),
            "palavras_chave": json.dumps(keywords, ensure_ascii=False)
        })

        # Salva a cada matéria processada (checkpoint), preservando também
        # tudo que já existia antes desta execução.
        pd.DataFrame(resultados).to_csv(
            ARQUIVO_SAIDA,
            index=False,
            encoding="utf-8-sig"
        )

        time.sleep(INTERVALO_ENTRE_CHAMADAS)

    print(f"\nArquivo salvo: {ARQUIVO_SAIDA} ({len(resultados)} matéria(s) no total).")


if __name__ == "__main__":
    main()