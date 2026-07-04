"""
Remove do analise_diaria.jsonl as entradas de dias específicos, forçando o
analise_diaria.py a reprocessá-los do zero na próxima execução — usado
depois de corrigir datas em folha.csv que "moveram" matérias para um dia
que já havia sido analisado antes da correção (senão o analise_diaria.py
pula esse dia, achando que já está feito, e as matérias corrigidas nunca
entram na análise de IA do dia certo).

Uso:
1. Edite a lista DATAS_PARA_REPROCESSAR abaixo com as datas corretas
   (formato DD/MM/AAAA) para onde as matérias foram corrigidas.
2. Rode: python invalidar_dias.py
3. Depois, rode: python analise_diaria.py
   (ele vai gerar novas entradas para esses dias, agora incluindo as
   matérias que foram corrigidas no folha.csv).
"""

import json
import os

ARQUIVO = "analise_diaria.jsonl"


# =========================
# PREENCHA AQUI as datas corretas que precisam ser reprocessadas
# =========================
DATAS_PARA_REPROCESSAR = [
    "11/08/2025",
    "28/02/2026",
    "01/07/2026",
]


def carregar_jsonl(caminho):
    if not os.path.exists(caminho):
        return []

    registros = []
    with open(caminho, "r", encoding="utf-8") as f:
        for linha in f:
            linha = linha.strip()
            if linha:
                registros.append(json.loads(linha))
    return registros


def salvar_jsonl(caminho, registros):
    with open(caminho, "w", encoding="utf-8") as f:
        for registro in registros:
            f.write(json.dumps(registro, ensure_ascii=False) + "\n")


def main():
    if not DATAS_PARA_REPROCESSAR:
        print("Nenhuma data cadastrada em DATAS_PARA_REPROCESSAR. Edite o script antes de rodar.")
        return

    registros = carregar_jsonl(ARQUIVO)
    print(f"Total de entradas no arquivo antes: {len(registros)}")

    datas_alvo = set(DATAS_PARA_REPROCESSAR)
    removidas = [r for r in registros if r.get("data") in datas_alvo]
    mantidas = [r for r in registros if r.get("data") not in datas_alvo]

    for data in DATAS_PARA_REPROCESSAR:
        if any(r.get("data") == data for r in removidas):
            print(f"✓ Entrada de {data} removida — será reprocessada na próxima execução do analise_diaria.py.")
        else:
            print(f"  {data} não tinha entrada no arquivo (já estava pendente, nada a remover).")

    if removidas:
        salvar_jsonl(ARQUIVO, mantidas)
        print(f"\n{len(removidas)} entrada(s) removida(s). Total de entradas agora: {len(mantidas)}.")
        print("\nPróximo passo: rode  python analise_diaria.py")
    else:
        print("\nNenhuma entrada foi removida (todas as datas já estavam pendentes ou o arquivo não mudou).")


if __name__ == "__main__":
    main()