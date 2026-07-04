"""
Verifica se existem dias sem nenhuma matéria coletada dentro do intervalo
já presente em folha.csv.

Diferente do modo `--auto` do main.py (que só olha a ponta — do último dia
salvo até hoje), este script varre o histórico INTEIRO em busca de lacunas
no meio, que podem ter ficado de coletas manuais interrompidas, erros
silenciosos, ou execuções puladas antes de a automação existir.

Uso:
    python verificar_gaps.py
"""

import pandas as pd

ARQUIVO_ENTRADA = "folha.csv"


def identificar_gaps():
    df = pd.read_csv(ARQUIVO_ENTRADA, encoding="utf-8-sig")

    df["date_dt"] = pd.to_datetime(df["date"], format="%d/%m/%Y", errors="coerce")
    df = df.dropna(subset=["date_dt"])

    if df.empty:
        print("Nenhuma data válida encontrada em folha.csv.")
        return None

    datas_presentes = set(df["date_dt"].dt.normalize().unique())
    data_min = df["date_dt"].min().normalize()
    data_max = df["date_dt"].max().normalize()

    todas_as_datas = pd.date_range(data_min, data_max, freq="D")
    dias_faltando = [d for d in todas_as_datas if d not in datas_presentes]

    return dias_faltando, data_min, data_max, len(todas_as_datas)


def agrupar_em_intervalos(dias_faltando):
    """Agrupa dias faltantes consecutivos num único intervalo, para um
    relatório mais legível do que listar cada dia isolado."""
    if not dias_faltando:
        return []

    intervalos = []
    inicio = dias_faltando[0]
    anterior = dias_faltando[0]

    for dia in dias_faltando[1:]:
        if (dia - anterior).days == 1:
            anterior = dia
        else:
            intervalos.append((inicio, anterior))
            inicio = dia
            anterior = dia

    intervalos.append((inicio, anterior))
    return intervalos


def main():
    resultado = identificar_gaps()
    if resultado is None:
        return

    dias_faltando, data_min, data_max, total_dias = resultado

    print(
        f"Intervalo coletado: {data_min.strftime('%d/%m/%Y')} a "
        f"{data_max.strftime('%d/%m/%Y')} ({total_dias} dia(s) no total)."
    )

    if not dias_faltando:
        print("\nNenhuma lacuna encontrada — todos os dias do intervalo têm ao menos 1 matéria.")
        return

    intervalos = agrupar_em_intervalos(dias_faltando)

    print(f"\n{len(dias_faltando)} dia(s) sem nenhuma matéria, em {len(intervalos)} intervalo(s):\n")

    for inicio, fim in intervalos:
        if inicio == fim:
            print(f"  {inicio.strftime('%d/%m/%Y')} (1 dia)")
        else:
            dias = (fim - inicio).days + 1
            print(f"  {inicio.strftime('%d/%m/%Y')} a {fim.strftime('%d/%m/%Y')} ({dias} dias)")

    print("\nPara coletar cada lacuna manualmente, rode:")
    for inicio, fim in intervalos:
        print(f"  python main.py --ini {inicio.strftime('%d/%m/%Y')} --fim {fim.strftime('%d/%m/%Y')}")

    print(
        "\nObs.: um dia sem matéria pode ser normal (editoria de baixo volume "
        "de publicação) e não necessariamente indica falha de coleta. Use bom "
        "senso antes de rodar tudo de uma vez — vale conferir se o período "
        "coincide com algo plausível (fim de semana, feriado, etc.) antes de "
        "assumir que é uma falha real."
    )


if __name__ == "__main__":
    main()