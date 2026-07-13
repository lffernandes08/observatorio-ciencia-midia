"""
Verifica se existem dias sem nenhuma matéria coletada dentro do intervalo
já presente em cada fonte de dados (Folha, CNN Brasil).

Diferente do modo `--auto` dos scrapers (que só olha a ponta — do último
dia salvo até hoje), este script varre o histórico INTEIRO de cada fonte
em busca de lacunas no meio, que podem ter ficado de coletas manuais
interrompidas, erros silenciosos, ou execuções puladas.

Uso:
    python verificar_gaps.py
"""

from datetime import datetime
import pandas as pd

ARQUIVOS_FONTES = [
    ("folha.csv", "Folha de S.Paulo"),
    ("cnn_brasil.csv", "CNN Brasil"),
    ("bbc_brasil.csv", "BBC News Brasil"),
]


def identificar_gaps(arquivo):
    try:
        df = pd.read_csv(arquivo, encoding="utf-8-sig")
    except FileNotFoundError:
        return "arquivo_ausente"

    df["date_dt"] = pd.to_datetime(df["date"], format="%d/%m/%Y", errors="coerce")
    df = df.dropna(subset=["date_dt"])

    if df.empty:
        print(f"Nenhuma data válida encontrada em {arquivo}.")
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


def sugerir_comando(arquivo, inicio, fim):
    """Cada scraper tem uma forma diferente de pedir um intervalo
    específico — a Folha aceita --ini/--fim diretamente; CNN Brasil e BBC
    Brasil só têm --historico-dias (contado a partir de hoje), então
    convertemos."""
    if arquivo == "folha.csv":
        return f"python main.py --ini {inicio.strftime('%d/%m/%Y')} --fim {fim.strftime('%d/%m/%Y')}"

    if arquivo in ("cnn_brasil.csv", "bbc_brasil.csv"):
        script = "cnn_brasil.py" if arquivo == "cnn_brasil.csv" else "bbc_brasil.py"
        dias_ate_o_inicio_do_gap = (datetime.now() - inicio).days
        return (
            f"python {script} --historico-dias {dias_ate_o_inicio_do_gap} "
            "(cobre o gap, mas também revarre dias já coletados — não tem "
            "problema, são ignorados na extração por já existirem no CSV)"
        )

    return None


def main():
    algum_gap_encontrado = False

    for arquivo, nome_veiculo in ARQUIVOS_FONTES:
        print(f"\n{'=' * 60}")
        print(f"{nome_veiculo} ({arquivo})")
        print("=" * 60)

        resultado = identificar_gaps(arquivo)

        if resultado == "arquivo_ausente":
            print(f"Arquivo {arquivo} ainda não existe — nada a checar.")
            continue

        if resultado is None:
            continue

        dias_faltando, data_min, data_max, total_dias = resultado

        print(
            f"Intervalo coletado: {data_min.strftime('%d/%m/%Y')} a "
            f"{data_max.strftime('%d/%m/%Y')} ({total_dias} dia(s) no total)."
        )

        if not dias_faltando:
            print("Nenhuma lacuna encontrada — todos os dias do intervalo têm ao menos 1 matéria.")
            continue

        algum_gap_encontrado = True
        intervalos = agrupar_em_intervalos(dias_faltando)

        print(f"\n{len(dias_faltando)} dia(s) sem nenhuma matéria, em {len(intervalos)} intervalo(s):\n")

        for inicio, fim in intervalos:
            if inicio == fim:
                print(f"  {inicio.strftime('%d/%m/%Y')} (1 dia)")
            else:
                dias = (fim - inicio).days + 1
                print(f"  {inicio.strftime('%d/%m/%Y')} a {fim.strftime('%d/%m/%Y')} ({dias} dias)")

        print("\nPara coletar cada lacuna:")
        for inicio, fim in intervalos:
            print(f"  {sugerir_comando(arquivo, inicio, fim)}")

    if algum_gap_encontrado:
        print(
            "\n\nObs.: um dia sem matéria pode ser normal (baixo volume de "
            "publicação naquele veículo) e não necessariamente indica falha "
            "de coleta. Use bom senso antes de rodar tudo de uma vez — vale "
            "conferir se o período coincide com algo plausível (fim de "
            "semana, feriado, etc.) antes de assumir que é uma falha real."
        )


if __name__ == "__main__":
    main()