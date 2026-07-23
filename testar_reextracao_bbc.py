"""
Remove as N matérias mais recentes de bbc_brasil.csv, forçando o scraper a
recoletá-las (e reextrair o conteúdo) na próxima execução — útil só pra
testar a extração nova depois de uma mudança no código, já que o scraper
normalmente pula qualquer URL que já esteja salva.

Uso:
    python testar_reextracao_bbc.py          # remove as 5 mais recentes
    python testar_reextracao_bbc.py 10       # remove as 10 mais recentes
"""

import sys
import pandas as pd

ARQUIVO = "bbc_brasil.csv"


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5

    df = pd.read_csv(ARQUIVO, encoding="utf-8-sig")
    df["date_dt"] = pd.to_datetime(df["date"], format="%d/%m/%Y", errors="coerce")
    df = df.sort_values("date_dt", ascending=False)

    removidas = df.head(n)
    mantidas = df.iloc[n:].drop(columns=["date_dt"])

    print(f"Removendo {len(removidas)} matéria(s) mais recente(s) de {ARQUIVO}:")
    for _, linha in removidas.iterrows():
        print(f"  {linha['date']} — {linha['title']}")

    mantidas.to_csv(ARQUIVO, index=False, encoding="utf-8-sig")

    print(f"\n{len(mantidas)} matéria(s) restante(s) em {ARQUIVO}.")
    print("Agora rode 'python bbc_brasil.py --historico-dias 7 --reiniciar' — essas "
          "matérias serão recoletadas com a extração nova.")


if __name__ == "__main__":
    main()