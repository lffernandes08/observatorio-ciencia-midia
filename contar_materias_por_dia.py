"""
Conta quantas matérias existem, no folha.csv atual, para cada uma das datas
para as quais você corrigiu matérias com data errada. Ajuda a confirmar se
essas datas já tinham outras matérias (e portanto já podiam ter uma análise
de IA prévia que precisa ser invalidada) ou se são dias "novos" no corpus.

Uso:
    python contar_materias_por_dia.py
"""

import pandas as pd

ARQUIVO = "folha.csv"

DATAS_PARA_VERIFICAR = [
    "11/08/2025",
    "28/02/2026",
    "01/07/2026",
]


def main():
    df = pd.read_csv(ARQUIVO, encoding="utf-8-sig")
    df["date_dt"] = pd.to_datetime(df["date"], format="%d/%m/%Y", errors="coerce")
    df = df.dropna(subset=["date_dt"])

    for data_str in DATAS_PARA_VERIFICAR:
        data_dt = pd.to_datetime(data_str, format="%d/%m/%Y")
        materias_do_dia = df[df["date_dt"] == data_dt]

        print(f"\n{data_str}: {len(materias_do_dia)} matéria(s) no total")
        for _, row in materias_do_dia.iterrows():
            print(f"    - {row['title']}")


if __name__ == "__main__":
    main()