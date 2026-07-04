"""
Identifica matérias cuja data extraída (coluna `date`, vinda do news-please)
diverge do ano/mês embutido na própria URL da Folha.

As URLs da Folha seguem o padrão .../editoria/AAAA/MM/titulo-da-materia.shtml
— esse ano/mês é definido pela própria Folha na publicação e é bem mais
confiável do que a data que o news-please tenta adivinhar a partir do
conteúdo da página (que pode confundir uma data citada no texto com a data
de publicação real).

Este script só helper NÃO corrige nada sozinho — lista as suspeitas para
você revisar e decidir a correção manualmente, já que o dia exato (não só
ano/mês) não dá para confirmar com certeza só pela URL.

Uso:
    python verificar_datas_suspeitas.py
"""

import re
import pandas as pd

ARQUIVO_ENTRADA = "folha.csv"

PADRAO_URL = re.compile(r"/(\d{4})/(\d{2})/")


def extrair_ano_mes_url(url):
    match = PADRAO_URL.search(str(url))
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def main():
    df = pd.read_csv(ARQUIVO_ENTRADA, encoding="utf-8-sig")

    df["date_dt"] = pd.to_datetime(df["date"], format="%d/%m/%Y", errors="coerce")

    ano_mes_url = df["url"].apply(extrair_ano_mes_url)
    df["ano_url"] = ano_mes_url.apply(lambda t: t[0])
    df["mes_url"] = ano_mes_url.apply(lambda t: t[1])

    com_data_valida = df.dropna(subset=["date_dt"]).copy()
    com_data_valida["ano_csv"] = com_data_valida["date_dt"].dt.year
    com_data_valida["mes_csv"] = com_data_valida["date_dt"].dt.month

    suspeitas = com_data_valida[
        com_data_valida["ano_url"].notna() &
        (
            (com_data_valida["ano_csv"] != com_data_valida["ano_url"]) |
            (com_data_valida["mes_csv"] != com_data_valida["mes_url"])
        )
    ]

    sem_padrao_url = df[df["ano_url"].isna()]

    print(f"Total de matérias no arquivo: {len(df)}")
    print(f"Matérias cuja URL não bate no padrão AAAA/MM esperado: {len(sem_padrao_url)}")
    print(f"Matérias com ano/mês divergente entre a coluna 'date' e a URL: {len(suspeitas)}\n")

    if suspeitas.empty:
        print("Nenhuma divergência encontrada — as datas parecem coerentes com as URLs.")
        return

    print("Divergências encontradas (revise e corrija manualmente):\n")
    for _, row in suspeitas.iterrows():
        print(f"  Título: {row['title']}")
        print(f"    URL:            {row['url']}")
        print(f"    Data no CSV:    {row['date']}  (ano {int(row['ano_csv'])}, mês {int(row['mes_csv'])})")
        print(f"    Ano/mês na URL: {int(row['ano_url'])}/{int(row['mes_url']):02d}")
        print()

    print(
        "Sugestão: corrija a coluna 'date' dessas linhas para o ano/mês indicado "
        "pela URL (o dia exato não dá pra confirmar só por aqui — mantenha o dia "
        "que já está, ou abra a matéria no link para conferir, se precisar de "
        "precisão exata)."
    )


if __name__ == "__main__":
    main()