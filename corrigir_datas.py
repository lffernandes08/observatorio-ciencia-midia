"""
Aplica correções manuais de data no folha.csv, identificadas com
verificar_datas_suspeitas.py.

COMO USAR:
1. Rode primeiro: python verificar_datas_suspeitas.py
2. Para cada divergência que aparecer, adicione uma linha na lista
   CORRECOES abaixo, com a URL exata da matéria e a data correta.
3. Rode este script: python corrigir_datas.py

O script casa cada correção pela URL (mais confiável que pelo título, que
pode ter acentos/formatação levemente diferente) e avisa se alguma URL da
lista não for encontrada no arquivo (pra você conferir se copiou certo).
"""

import pandas as pd

ARQUIVO = "folha.csv"


# =========================
# PREENCHA AQUI as correções encontradas pelo verificar_datas_suspeitas.py
# Formato: (url_exata, data_correta_no_formato_DD/MM/AAAA)
# =========================
CORRECOES = [
    (
        "https://www1.folha.uol.com.br/ciencia/2026/02/ossos-encontrados-em-obra-de-ferrovia-no-maranhao-sao-de-dinossauro-pescocudo.shtml",
        "28/02/2026",
    ),
    # Adicione mais linhas aqui, uma por divergência encontrada, exemplo:
    (
        "https://www1.folha.uol.com.br/ciencia/2025/08/victor-nussenzveig-referencia-no-combate-a-malaria-morre-aos-97-anos.shtml",
        "11/08/2025",
    ),

    ("https://www1.folha.uol.com.br/ciencia/2026/07/pesquisas-questionam-ideia-de-que-desacreditar-o-livre-arbitrio-pode-levar-voce-a-ser-uma-pessoa-ma.shtml",
     "01/07/2026",
    ),
]


def main():
    if not CORRECOES:
        print("Nenhuma correção cadastrada na lista CORRECOES. Edite o script antes de rodar.")
        return

    df = pd.read_csv(ARQUIVO, encoding="utf-8-sig")

    total_aplicadas = 0
    total_nao_encontradas = 0

    for url, data_correta in CORRECOES:
        encontrou = (df["url"] == url).any()

        if not encontrou:
            print(f"⚠ URL não encontrada no arquivo, confira se copiou certo:\n  {url}")
            total_nao_encontradas += 1
            continue

        data_antiga = df.loc[df["url"] == url, "date"].values[0]
        df.loc[df["url"] == url, "date"] = data_correta
        total_aplicadas += 1

        print(f"✓ Corrigido: {url}")
        print(f"    {data_antiga}  ->  {data_correta}")

    if total_aplicadas > 0:
        df.to_csv(ARQUIVO, index=False, encoding="utf-8-sig")
        print(f"\n{total_aplicadas} correção(ões) aplicada(s) e salva(s) em {ARQUIVO}.")
    else:
        print("\nNenhuma correção foi aplicada (nenhuma URL da lista foi encontrada).")

    if total_nao_encontradas > 0:
        print(f"{total_nao_encontradas} URL(s) da lista não foram encontradas — nada foi alterado para elas.")


if __name__ == "__main__":
    main()