import re
import os
import html
import json
import math
from itertools import combinations
from collections import Counter, defaultdict

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import plotly.express as px
import plotly.io as pio

pio.templates.default = "plotly_dark"

try:
    import networkx as nx
    import community as community_louvain
    REDE_SEMANTICA_DISPONIVEL = True
except ImportError:
    REDE_SEMANTICA_DISPONIVEL = False


st.set_page_config(
    page_title="Observatório de ciência na mídia",
    layout="wide"
)

st.title("OBSERVATÓRIO DE CIÊNCIA NA MÍDIA")
st.caption(
    "Acompanhamento diário e exploração histórica da cobertura de ciência "
    "na mídia brasileira — combinando estatística, processamento de "
    "linguagem natural e análise por inteligência artificial."
)


import nltk
from nltk.corpus import stopwords


@st.cache_resource
def baixar_recursos_nltk():
    """Baixa recursos do NLTK uma única vez por sessão do servidor
    (cache_resource evita repetir a checagem de disco a cada rerun)."""
    nltk.download("stopwords", quiet=True)


baixar_recursos_nltk()


@st.cache_data
def carregar_stopwords():
    stopwords_nltk = set(stopwords.words("portuguese"))

    try:
        with open("stopwords.txt", encoding="utf-8") as f:
            stopwords_personalizadas = {
                linha.strip().lower()
                for linha in f
                if linha.strip()
            }
    except FileNotFoundError:
        st.sidebar.warning(
            "Arquivo stopwords.txt não encontrado. "
            "Usando apenas a lista padrão do NLTK."
        )
        stopwords_personalizadas = set()

    return stopwords_nltk | stopwords_personalizadas


STOPWORDS = carregar_stopwords()


# =========================
# Carregamento de dados
# =========================

COLUNAS_ESPERADAS = ["date", "title", "text", "section", "author", "url"]


def formatar_autor(valor):
    """Alguns registros trazem o autor como string de lista Python
    (ex: "['Fulano']", herdado do scraper) em vez do nome puro.
    Normaliza para exibição sem quebrar se o valor já vier limpo."""
    texto = str(valor).strip()

    if texto.startswith("[") and texto.endswith("]"):
        try:
            import ast
            lista = ast.literal_eval(texto)
            if isinstance(lista, list) and lista:
                return ", ".join(str(item).strip() for item in lista)
        except (ValueError, SyntaxError):
            pass

    return texto


@st.cache_data
def carregar_dados():
    try:
        df = pd.read_csv("folha.csv", encoding="utf-8-sig")
    except FileNotFoundError:
        st.error(
            "Arquivo folha.csv não encontrado na pasta do app. "
            "Verifique se o arquivo está presente antes de continuar."
        )
        st.stop()
    except Exception as e:
        st.error(f"Erro ao ler folha.csv: {e}")
        st.stop()

    colunas_faltando = [c for c in COLUNAS_ESPERADAS if c not in df.columns]
    if colunas_faltando:
        st.error(
            "O arquivo folha.csv está com colunas faltando: "
            f"{', '.join(colunas_faltando)}. "
            f"Colunas esperadas: {', '.join(COLUNAS_ESPERADAS)}."
        )
        st.stop()

    df["date_dt"] = pd.to_datetime(
        df["date"],
        format="%d/%m/%Y",
        errors="coerce"
    )

    df = df.dropna(subset=["date_dt"])

    if df.empty:
        st.error("Nenhuma linha com data válida foi encontrada em folha.csv.")
        st.stop()

    # "Data" (exibição) e agrupamentos por Mês/Ano
    df["Data"] = df["date_dt"].dt.strftime("%d/%m/%Y")

    df["mes_dt"] = df["date_dt"].dt.to_period("M").dt.to_timestamp()
    df["Mês"] = df["mes_dt"].dt.strftime("01/%m/%Y")

    df["ano_dt"] = df["date_dt"].dt.to_period("Y").dt.to_timestamp()
    df["Ano"] = df["ano_dt"].dt.strftime("01/01/%Y")

    df["Título"] = df["title"].fillna("")
    df["Texto"] = df["text"].fillna("")
    df["Editoria"] = df["section"].fillna("")
    df["Autor"] = df["author"].fillna("").apply(formatar_autor)
    df["URL"] = df["url"].fillna("")

    df["Palavras"] = df["Texto"].apply(lambda x: len(str(x).split()))

    return df


@st.cache_data
def carregar_analise_diaria():
    """Carrega o histórico de análises diárias de IA gerado por
    analise_diaria.py. Cada linha do arquivo é o JSON da análise de um dia
    específico (chave 'data' no formato DD/MM/AAAA)."""
    try:
        with open("analise_diaria.jsonl", "r", encoding="utf-8") as f:
            linhas = [linha.strip() for linha in f if linha.strip()]
    except FileNotFoundError:
        return []

    registros = []
    for linha in linhas:
        try:
            registros.append(json.loads(linha))
        except json.JSONDecodeError:
            continue

    return registros


def obter_analise_do_dia(registros, data_alvo):
    """Retorna o registro cujo campo 'data' bate com data_alvo (DD/MM/AAAA),
    ou None se a análise daquele dia ainda não foi gerada."""
    if not registros or not data_alvo:
        return None

    for registro in reversed(registros):
        if registro.get("data") == data_alvo:
            return registro

    return None


def analise_get(analise, chave, padrao=None):
    """Acesso seguro a chaves da análise diária, evitando KeyError
    se o formato do registro mudar ou vier incompleto."""
    if padrao is None:
        padrao = []
    valor = analise.get(chave, padrao)
    return valor if valor is not None else padrao


def agregar_analises_periodo(analises, datas_no_filtro, top_n=8):
    """Agrega, em Python (sem nenhuma chamada de IA), as análises diárias
    cuja data está no conjunto de datas do filtro atual. Listas categóricas
    viram rankings de frequência; abrangência vira a moda entre os dias."""
    analises_periodo = [a for a in analises if a.get("data") in datas_no_filtro]

    if not analises_periodo:
        return None

    contadores = {
        "temas_em_destaque": Counter(),
        "enquadramentos_predominantes": Counter(),
        "areas_predominantes": Counter(),
        "atores_mais_visiveis": Counter(),
    }
    contagem_abrangencia = Counter()
    tendencias_diarias = []
    total_materias = 0

    for a in analises_periodo:
        for chave, contador in contadores.items():
            for item in analise_get(a, chave):
                if item:
                    contador[item] += 1

        abrangencia = a.get("abrangencia_predominante")
        if abrangencia:
            contagem_abrangencia[abrangencia] += 1

        tendencia = a.get("tendencia_do_periodo")
        if tendencia:
            tendencias_diarias.append((a.get("data"), tendencia))

        total_materias += analise_get(a, "n_materias", 0)

    return {
        "temas_em_destaque": [item for item, _ in contadores["temas_em_destaque"].most_common(top_n)],
        "enquadramentos_predominantes": [item for item, _ in contadores["enquadramentos_predominantes"].most_common(top_n)],
        "areas_predominantes": [item for item, _ in contadores["areas_predominantes"].most_common(top_n)],
        "atores_mais_visiveis": [item for item, _ in contadores["atores_mais_visiveis"].most_common(top_n)],
        "abrangencia_predominante": (
            contagem_abrangencia.most_common(1)[0][0] if contagem_abrangencia else "Não conclusivo"
        ),
        "n_dias": len(analises_periodo),
        "n_materias": total_materias,
        "tendencias_diarias": sorted(tendencias_diarias, key=lambda t: t[0]),
    }


def sintetizar_tendencia_periodo(tendencias_diarias):
    """Chamada de IA sob demanda (só quando o usuário clica no botão).
    Input é barato: usa apenas as frases curtas de tendência já geradas
    pelo analise_diaria.py, não as matérias originais — não reprocessa nada."""
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Variável de ambiente OPENAI_API_KEY não configurada no ambiente do app."
        )

    client = OpenAI(api_key=api_key)

    linhas = "\n".join(f"- {data}: {texto}" for data, texto in tendencias_diarias)

    prompt = f"""
Você é pesquisador em jornalismo científico. Abaixo estão frases curtas que
resumem a tendência da cobertura de ciência da Folha em cada dia de um período.

Sintetize essas frases em um único parágrafo coeso (3 a 5 frases), em português
do Brasil, descrevendo a tendência geral do período como um todo — não repita
as frases dia a dia, produza uma leitura consolidada.

Não invente informações além do que está nas frases abaixo. Não use markdown.

Frases diárias do período:
{linhas}
"""

    resposta = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "Você sintetiza tendências de cobertura jornalística de ciência a partir de resumos diários já prontos."
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    return resposta.choices[0].message.content.strip()


@st.cache_data
def carregar_keywords():
    """Carrega as palavras-chave extraídas por IA (extrair_keywords.py).
    Retorna None se o arquivo ainda não foi gerado, para não quebrar o app."""
    try:
        df_kw = pd.read_csv("materias_keywords.csv", encoding="utf-8-sig")
    except FileNotFoundError:
        return None
    except Exception as e:
        st.sidebar.warning(f"Erro ao ler materias_keywords.csv: {e}")
        return None

    if "url" not in df_kw.columns or "palavras_chave" not in df_kw.columns:
        st.sidebar.warning(
            "materias_keywords.csv está com colunas inesperadas "
            "(esperado: url, palavras_chave)."
        )
        return None

    def _parse_lista(valor):
        try:
            lista = json.loads(valor)
            return lista if isinstance(lista, list) else []
        except (TypeError, ValueError):
            return []

    df_kw["palavras_chave_lista"] = df_kw["palavras_chave"].apply(_parse_lista)
    df_kw["url"] = df_kw["url"].fillna("")

    return df_kw


def numero_br(valor):
    return f"{valor:,.0f}".replace(",", ".")


def tokenizar(texto):
    texto = str(texto).lower()
    texto = re.sub(r"[^a-zà-úçãõáéíóúâêôü\s]", " ", texto)
    tokens = texto.split()
    return [t for t in tokens if len(t) > 2 and t not in STOPWORDS]


def gerar_ngrams(tokens, n):
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]


@st.cache_data
def contar_termos(textos, n=1, top_n=30):
    """Cacheada: 'textos' precisa ser uma tupla (hashable) para o cache funcionar."""
    todos = []

    for texto in textos:
        tokens = tokenizar(texto)

        if n == 1:
            todos.extend(tokens)
        else:
            todos.extend(gerar_ngrams(tokens, n))

    contagem = Counter(todos)

    return pd.DataFrame(
        contagem.most_common(top_n),
        columns=["Termo", "Frequência"]
    )


def agrupar_por_escala(df, escala, coluna_valor=None):
    """Agrupa um DataFrame por Dia/Mês/Ano, evitando duplicar a lógica
    de agrupamento em vários pontos do app.

    Se coluna_valor for None, agrupa por contagem de linhas (size).
    Se for informado, soma os valores dessa coluna.
    """
    mapa_colunas = {
        "Dia": ("Data", "date_dt"),
        "Mês": ("Mês", "mes_dt"),
        "Ano": ("Ano", "ano_dt"),
    }
    col_rotulo, col_ordenacao = mapa_colunas[escala]

    if coluna_valor is None:
        resultado = (
            df.groupby([col_rotulo, col_ordenacao])
            .size()
            .reset_index(name="Matérias")
            .sort_values(col_ordenacao)
        )
    else:
        resultado = (
            df.groupby([col_rotulo, col_ordenacao])[coluna_valor]
            .sum()
            .reset_index()
            .sort_values(col_ordenacao)
        )

    resultado["Período"] = resultado[col_rotulo]
    return resultado

df_original = carregar_dados()
st.write("🔍 Veículos em df_original:", df_original["Veículo"].unique().tolist())
st.write("🔍 Linhas por veículo:", df_original["Veículo"].value_counts().to_dict())
veiculos_teste = sorted(df_original["Veículo"].unique())
st.write("🔍 len(veiculos_disponiveis) > 1 ?", len(veiculos_teste) > 1)
df = df_original.copy()


# =========================
# Filtros
# =========================

st.sidebar.header("🧭 Explorar")
st.sidebar.subheader("Período")

data_min = df["date_dt"].min().date()
data_max = df["date_dt"].max().date()

periodo_selecionado = st.sidebar.date_input(
    "Intervalo de datas",
    value=(data_min, data_max),
    min_value=data_min,
    max_value=data_max,
    format="DD/MM/YYYY"
)

# st.date_input com range pode retornar 1 ou 2 datas dependendo do estado da seleção
if isinstance(periodo_selecionado, tuple) and len(periodo_selecionado) == 2:
    inicio, fim = periodo_selecionado
    df = df[
        (df["date_dt"] >= pd.Timestamp(inicio)) &
        (df["date_dt"] <= pd.Timestamp(fim))
    ]
    periodo_label_sidebar = f"{inicio.strftime('%d/%m/%Y')} a {fim.strftime('%d/%m/%Y')}"
else:
    st.sidebar.info("Selecione a data final para aplicar o filtro de período.")
    periodo_label_sidebar = "período incompleto"

st.sidebar.subheader("Visualização temporal")

escala = st.sidebar.radio(
    "Agrupar por",
    ["Dia", "Mês", "Ano"],
    index=0
)

st.sidebar.subheader("Seção")

secao_selecionada = st.sidebar.radio(
    "Escolha a visualização",
    [
        "Panorama do dia",
        "Visão geral",
        "Temas",
        "Sismógrafo",
        "Análise IA",
        "Rede semântica"
    ],
    index=0,
    key="secao_selecionada"
)

st.sidebar.caption(
    "\"Panorama do dia\" sempre mostra o dia mais recente do dataset. As "
    "demais seções usam o período e a escala selecionados aqui em cima."
)

st.sidebar.subheader("Busca")

if "busca_lateral" not in st.session_state:
    st.session_state["busca_lateral"] = ""


def _preencher_busca(termo):
    st.session_state["busca_lateral"] = termo


busca = st.sidebar.text_input(
    "Buscar palavra no título ou texto",
    key="busca_lateral"
)


@st.cache_data
def termos_populares_corpus(top_n=8):
    """Termos mais frequentes de todo o corpus (não filtrado), usados
    como sugestões rápidas de busca."""
    textos = tuple((df_original["Título"] + " " + df_original["Texto"]).tolist())
    termos_df = contar_termos(textos, n=1, top_n=top_n)
    return termos_df["Termo"].tolist()


termos_chips = termos_populares_corpus(8)

if termos_chips:
    st.sidebar.caption("🔥 Termos em alta no corpus")
    col_chips = st.sidebar.columns(2)
    for i, termo in enumerate(termos_chips):
        with col_chips[i % 2]:
            st.button(
                termo,
                key=f"chip_{termo}",
                on_click=_preencher_busca,
                args=(termo,),
                use_container_width=True
            )

if busca:
    busca_lower = busca.lower()
    df = df[
        df["Título"].str.lower().str.contains(busca_lower, regex=False) |
        df["Texto"].str.lower().str.contains(busca_lower, regex=False)
    ]

    st.sidebar.caption(f"🔎 {numero_br(len(df))} matéria(s) encontrada(s) para **{busca}**")

    with st.sidebar.expander("📈 Tendência do termo", expanded=False):
        if len(df) > 0:
            evolucao_mini = agrupar_por_escala(df, "Mês")
            if len(evolucao_mini) > 1:
                fig_mini = px.line(evolucao_mini, x="Período", y="Matérias", height=150)
                fig_mini.update_layout(
                    margin=dict(l=0, r=0, t=10, b=0),
                    xaxis_visible=False,
                    yaxis_visible=False,
                    showlegend=False
                )
                fig_mini.update_traces(line_color="#5FBFA0")
                st.plotly_chart(
                    fig_mini,
                    use_container_width=True,
                    config={"displayModeBar": False}
                )
            else:
                st.caption("Poucos pontos no período para exibir uma tendência.")
        else:
            st.caption("Nenhuma matéria encontrada.")

if df.empty:
    st.warning("Nenhuma matéria encontrada para os filtros selecionados.")
    st.stop()


# =========================
# Panorama IA
# =========================

def card(titulo, itens, icone=""):
    itens_seguros = [html.escape(str(item)) for item in itens]
    html_itens = "".join([f"<span class='tag'>{item}</span>" for item in itens_seguros])
    st.markdown(
        f"""
        <div class="card-panorama">
            <div class="card-titulo">{icone} {html.escape(titulo)}</div>
            <div class="tags">{html_itens}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def mostrar_periodo_no_topo(periodo_label, escala_label):
    """Exibe um indicador fixo no topo da seção com o período e a escala
    selecionados na sidebar — para o usuário nunca perder de vista a que
    período a análise se refere, independente de qual seção está vendo."""
    st.markdown(
        f'<div class="painel-eyebrow">📅 Período selecionado: '
        f'{html.escape(periodo_label)} · agrupado por {html.escape(escala_label).lower()}</div>',
        unsafe_allow_html=True
    )


st.markdown(
    """
    <style>
    :root {
        --obs-ink: #10151F;
        --obs-ink-2: #171E2C;
        --obs-paper: #E8E4D8;
        --obs-muted: #8B93A7;
        --obs-grid: #2A3244;
        --obs-teal: #5FBFA0;
        --obs-amber: #E8A33D;
    }
    .painel-eyebrow {
        font-family: 'SF Mono', 'Cascadia Code', Consolas, Menlo, monospace;
        font-size: 0.72rem;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: var(--obs-teal);
        margin-bottom: 4px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .painel-eyebrow::before {
        content: "";
        width: 7px; height: 7px;
        border-radius: 50%;
        background: var(--obs-amber);
        box-shadow: 0 0 8px var(--obs-amber);
    }
    .card-panorama {
        background: var(--obs-ink-2);
        border: 1px solid var(--obs-grid);
        border-left: 3px solid var(--obs-teal);
        border-radius: 14px;
        padding: 18px 20px;
        margin-bottom: 14px;
        min-height: 150px;
        transition: transform 0.18s ease, border-color 0.18s ease;
    }
    .card-panorama:hover {
        transform: translateY(-3px);
        border-left-color: var(--obs-amber);
    }
    .card-titulo {
        font-family: 'SF Mono', 'Cascadia Code', Consolas, Menlo, monospace;
        font-size: 0.82rem;
        font-weight: 600;
        color: var(--obs-paper);
        margin-bottom: 14px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .tags {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
    }
    .tag {
        background: rgba(95,191,160,0.10);
        border: 1px solid rgba(95,191,160,0.35);
        color: var(--obs-paper);
        border-radius: 999px;
        padding: 6px 12px;
        font-size: 0.88rem;
        line-height: 1.2;
    }
    .insight-box {
        background: var(--obs-ink-2);
        border: 1px solid var(--obs-grid);
        border-left: 5px solid var(--obs-amber);
        border-radius: 12px;
        padding: 18px 22px;
        margin-top: 10px;
        margin-bottom: 18px;
        font-size: 1rem;
        line-height: 1.5;
        color: var(--obs-paper);
    }
    .insight-title {
        font-weight: 700;
        margin-bottom: 6px;
        color: var(--obs-paper);
    }

    /* Nuvem de palavras-chave (IA) */
    .nuvem-container {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        align-items: center;
        padding: 20px 6px;
    }
    .chip-nuvem {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 999px;
        line-height: 1.3;
        font-weight: 600;
        transition: transform 0.15s ease;
    }
    .chip-nuvem:hover {
        transform: scale(1.08);
    }
    .chip-nuvem.tom-a {
        background: rgba(95,191,160,0.12);
        border: 1px solid rgba(95,191,160,0.45);
        color: #9FE0C7;
    }
    .chip-nuvem.tom-b {
        background: rgba(232,163,61,0.12);
        border: 1px solid rgba(232,163,61,0.45);
        color: #F3C784;
    }
    </style>
    """,
    unsafe_allow_html=True
)

data_mais_recente_dt = df_original["date_dt"].max()
data_mais_recente_str = (
    data_mais_recente_dt.strftime("%d/%m/%Y") if pd.notna(data_mais_recente_dt) else None
)

analises_diarias = carregar_analise_diaria()
analise = obter_analise_do_dia(analises_diarias, data_mais_recente_str)
df_keywords = carregar_keywords()


# =========================
# Medidor de ritmo de publicação (reutilizável: dia e período)
# =========================

@st.cache_data
def calcular_temperatura_cobertura(chave_cache):
    """Compara o ritmo de publicação do período filtrado (matérias/dia)
    com a média histórica de todo o corpus. Retorna a razão entre os dois
    (1.0 = na média histórica), já com guarda contra divisão por zero."""
    serie_historica = agrupar_por_escala(df_original, "Dia")
    ritmo_historico = serie_historica["Matérias"].mean()

    serie_atual = agrupar_por_escala(df, "Dia")
    ritmo_atual = serie_atual["Matérias"].mean() if len(serie_atual) > 0 else 0.0

    razao = ritmo_atual / ritmo_historico if ritmo_historico > 0 else 1.0

    return ritmo_atual, ritmo_historico, razao


chave_cache_temperatura = (
    len(df), df["date_dt"].min(), df["date_dt"].max(), len(df_original)
)
ritmo_atual, ritmo_historico, razao_temperatura = calcular_temperatura_cobertura(
    chave_cache_temperatura
)

# Ritmo do dia mais recente (não depende do filtro da sidebar): compara o
# volume de matérias do dia com a média histórica diária do corpus inteiro.
n_materias_hoje = (
    int((df_original["date_dt"] == data_mais_recente_dt).sum())
    if pd.notna(data_mais_recente_dt) else 0
)
razao_dia = n_materias_hoje / ritmo_historico if ritmo_historico > 0 else 1.0


GAUGE_HTML_TEMPLATE = """
<style>
  #gauge-root {
    --ink: #10151F;
    --ink-2: #171E2C;
    --paper: #E8E4D8;
    --muted: #8B93A7;
    --grid: #2A3244;
    --fonte-mono: 'SF Mono', 'Cascadia Code', Consolas, Menlo, monospace;
    --fonte-sans: -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    background: var(--ink-2);
    border: 1px solid var(--grid);
    border-radius: 16px;
    padding: 18px 26px 12px;
    font-family: var(--fonte-sans);
    color: var(--paper);
    display: flex;
    align-items: center;
    gap: 28px;
    flex-wrap: wrap;
  }
  #gauge-root .gauge-eyebrow {
    font-family: var(--fonte-mono);
    font-size: 0.72rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: __COR_STATUS__;
    margin-bottom: 2px;
  }
  #gauge-root .gauge-svg-wrap {
    flex-shrink: 0;
    width: 190px;
  }
  #gauge-root .gauge-leitura {
    min-width: 220px;
  }
  #gauge-root .gauge-status {
    font-size: 1.15rem;
    font-weight: 700;
    color: __COR_STATUS__;
    margin-bottom: 6px;
  }
  #gauge-root .gauge-detalhe {
    font-family: var(--fonte-mono);
    font-size: 0.82rem;
    color: var(--muted);
    line-height: 1.6;
  }
  #agulha {
    transform-origin: 150px 150px;
    transform: rotate(-90deg);
    transition: transform 1.3s cubic-bezier(.34,1.2,.4,1);
  }
</style>

<div id="gauge-root">
  <div class="gauge-svg-wrap">
    <div class="gauge-eyebrow">__EYEBROW__</div>
    <svg viewBox="0 0 300 165" xmlns="http://www.w3.org/2000/svg">
      <path d="M 40,150 A 110,110 0 0,1 100.1,52.0" fill="none" stroke="#5FBFA0" stroke-width="16" stroke-linecap="round" opacity="0.85"/>
      <path d="M 100.1,52.0 A 110,110 0 0,1 199.9,52.0" fill="none" stroke="#8B93A7" stroke-width="16" stroke-linecap="round" opacity="0.55"/>
      <path d="M 199.9,52.0 A 110,110 0 0,1 260,150" fill="none" stroke="#E8A33D" stroke-width="16" stroke-linecap="round" opacity="0.85"/>
      <line id="agulha" x1="150" y1="150" x2="150" y2="55" stroke="#E8E4D8" stroke-width="4" stroke-linecap="round"/>
      <circle cx="150" cy="150" r="8" fill="#E8E4D8"/>
    </svg>
  </div>
  <div class="gauge-leitura">
    <div class="gauge-status">__STATUS__</div>
    <div class="gauge-detalhe">
      __ROTULO_VALOR__ ▸ __RITMO_ATUAL__ matéria(s)/dia<br/>
      Histórico do corpus ▸ __RITMO_HISTORICO__ matéria(s)/dia<br/>
      Variação ▸ __DIFERENCA__% vs. histórico
    </div>
  </div>
</div>

<script>
(function() {
  setTimeout(function() {
    var agulha = document.getElementById('agulha');
    if (agulha) {
      agulha.style.transform = 'rotate(__ANGULO__deg)';
    }
  }, 250);
})();
</script>
"""


def renderizar_gauge_ritmo(eyebrow, rotulo_valor, ritmo_valor, ritmo_hist, razao, altura=200):
    """Renderiza o gauge de ritmo de publicação. Reutilizável para qualquer
    par (valor atual, valor histórico) — usado tanto para o dia mais recente
    quanto para o período filtrado pelo usuário."""
    razao_clamp = max(0.0, min(razao, 2.0))
    angulo_final = (razao_clamp / 2.0) * 180 - 90

    if razao < 0.7:
        status = "Abaixo do ritmo histórico"
        cor_status = "#5FBFA0"
    elif razao > 1.3:
        status = "Acima do ritmo histórico"
        cor_status = "#E8A33D"
    else:
        status = "Na média histórica"
        cor_status = "#E8E4D8"

    diferenca_pct = (razao - 1.0) * 100

    gauge_html = (
        GAUGE_HTML_TEMPLATE
        .replace("__EYEBROW__", html.escape(eyebrow))
        .replace("__ROTULO_VALOR__", html.escape(rotulo_valor))
        .replace("__COR_STATUS__", cor_status)
        .replace("__STATUS__", html.escape(status))
        .replace("__RITMO_ATUAL__", f"{ritmo_valor:.1f}".replace(".", ","))
        .replace("__RITMO_HISTORICO__", f"{ritmo_hist:.1f}".replace(".", ","))
        .replace("__DIFERENCA__", f"{diferenca_pct:+.0f}".replace(".", ","))
        .replace("__ANGULO__", f"{angulo_final:.1f}")
    )

    components.html(gauge_html, height=altura, scrolling=False)


# =========================
# Seção "Panorama do dia" — uma opção do menu lateral, como as demais.
# Só aparece quando selecionada (deixa de disputar espaço com as seções
# dependentes do período quando o usuário troca de visualização).
# =========================

if secao_selecionada == "Panorama do dia":
    st.markdown('<div class="painel-eyebrow">Panorama do dia</div>', unsafe_allow_html=True)
    st.subheader(f"📌 Panorama da cobertura — {data_mais_recente_str}")
    st.caption(
        f"{numero_br(n_materias_hoje)} matéria(s) publicadas em {data_mais_recente_str}. "
        "Este painel sempre mostra o dia mais recente do dataset, independente do "
        "período selecionado na barra lateral — as demais seções do app respeitam o filtro."
    )

    renderizar_gauge_ritmo(
        "Ritmo de publicação (hoje)",
        "Dia mais recente",
        n_materias_hoje, ritmo_historico, razao_dia
    )

    if analise:
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            card(
                "Temas em destaque",
                analise_get(analise, "temas_em_destaque"),
                "🧪"
            )

            card(
                "Enquadramentos",
                analise_get(analise, "enquadramentos_predominantes"),
                "📰"
            )

        with col_b:
            card(
                "Áreas predominantes",
                analise_get(analise, "areas_predominantes"),
                "🔬"
            )
            card(
                "Atores mais visíveis",
                analise_get(analise, "atores_mais_visiveis"),
                "👥"
            )

        with col_c:
            card(
                "Abrangência",
                [analise_get(analise, "abrangencia_predominante", "—")],
                "🌎"
            )

        st.markdown(
            f"""
            <div class="insight-box">
                <div class="insight-title">📈 Tendência do dia</div>
                {html.escape(analise_get(analise, "tendencia_do_periodo", ""))}
            </div>
            """,
            unsafe_allow_html=True
        )

    else:
        aviso_data = data_mais_recente_str or "o dia mais recente do dataset"
        st.info(
            f"Análise de IA ainda não gerada para {aviso_data} "
            "(rode: python analise_diaria.py). O ritmo de publicação acima já está "
            "disponível, pois não depende dessa etapa."
        )

    st.divider()

    col_titulo_sorteio, col_botao_sorteio = st.columns([4, 1])
    with col_titulo_sorteio:
        st.markdown("#### 🎲 Três matérias do dia, para começar por algum lugar")
    with col_botao_sorteio:
        sortear_novamente = st.button("🔀 Sortear outras", key="btn_sortear_materias")

    df_dia_completo = df_original[df_original["date_dt"] == data_mais_recente_dt] if pd.notna(data_mais_recente_dt) else df_original.iloc[0:0]

    precisa_sortear = (
        sortear_novamente
        or "materias_aleatorias_data" not in st.session_state
        or st.session_state.get("materias_aleatorias_data") != data_mais_recente_str
    )

    if precisa_sortear:
        n_amostra = min(3, len(df_dia_completo))
        st.session_state["materias_aleatorias"] = (
            df_dia_completo.sample(n=n_amostra) if n_amostra > 0 else df_dia_completo
        )
        st.session_state["materias_aleatorias_data"] = data_mais_recente_str

    materias_amostra = st.session_state.get("materias_aleatorias")

    if materias_amostra is None or materias_amostra.empty:
        st.caption("Nenhuma matéria disponível para sorteio neste dia.")
    else:
        cols_materias = st.columns(len(materias_amostra))
        for col, (_, materia) in zip(cols_materias, materias_amostra.iterrows()):
            with col:
                titulo_materia = html.escape(str(materia["Título"]) or "(sem título)")
                editoria_materia = html.escape(str(materia["Editoria"]) or "—")
                autor_materia = html.escape(materia["Autor"] or "Redação")
                url_materia = str(materia["URL"])

                if url_materia.startswith("http"):
                    titulo_html = f'<a href="{url_materia}" target="_blank" style="color:#E8E4D8;text-decoration:none;">{titulo_materia}</a>'
                else:
                    titulo_html = titulo_materia

                st.markdown(
                    f"""
                    <div class="card-panorama" style="min-height:130px;">
                        <div style="font-family:'SF Mono',Consolas,monospace;font-size:0.7rem;
                                    color:#8B93A7;text-transform:uppercase;letter-spacing:0.05em;
                                    margin-bottom:8px;">{editoria_materia}</div>
                        <div style="font-weight:600;margin-bottom:10px;line-height:1.4;">{titulo_html}</div>
                        <div style="font-size:0.8rem;color:#8B93A7;font-style:italic;">{autor_materia}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


st.divider()

if secao_selecionada == "Visão geral":
    mostrar_periodo_no_topo(periodo_label_sidebar, escala)

    st.subheader("Indicadores gerais")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Matérias", numero_br(len(df)))
    col2.metric("Autores", numero_br(df["Autor"].astype(str).nunique()))
    col3.metric("Palavras", numero_br(int(df["Palavras"].sum())))
    col4.metric(
        "Média de palavras",
        numero_br(int(df["Palavras"].mean())) if len(df) else "0"
    )

    st.divider()

    st.subheader("Evolução da cobertura")

    serie = agrupar_por_escala(df, escala)

    fig = px.line(
        serie,
        x="Período",
        y="Matérias",
        markers=True,
        title=f"Evolução por {escala.lower()}"
    )

    fig.update_layout(
        xaxis_title="Período",
        yaxis_title="Número de matérias",
        xaxis_type="category"
    )

    fig.update_traces(
        hovertemplate="<b>Período:</b> %{x}<br><b>Matérias:</b> %{y}<extra></extra>"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("Distribuição por editoria")

    por_editoria = (
        df.groupby("Editoria")
        .size()
        .reset_index(name="Matérias")
        .sort_values("Matérias", ascending=True)
    )

    fig_editoria = px.bar(
        por_editoria,
        x="Matérias",
        y="Editoria",
        orientation="h",
        title="Matérias por editoria",
        text="Matérias"
    )

    st.plotly_chart(fig_editoria, use_container_width=True)

    st.divider()
    st.subheader("Textos mais longos")

    tabela = df[["Data", "Título", "Editoria", "Autor", "Palavras", "URL"]].copy()
    tabela = tabela.sort_values("Palavras", ascending=False)

    st.dataframe(
        tabela,
        use_container_width=True,
        hide_index=True
    )


if secao_selecionada == "Temas":
    mostrar_periodo_no_topo(periodo_label_sidebar, escala)


    if df_keywords is not None:
        st.subheader("🧠 Palavras-chave identificadas por IA")
        st.caption(
            "Extraídas via modelo de linguagem a partir do conteúdo de cada matéria "
            "(ver extrair_keywords.py) — tendem a ser mais específicas do que a "
            "contagem bruta de palavras e n-gramas abaixo, já que descartam termos "
            "genéricos e mantêm conceitos, instituições, tecnologias e atores."
        )

        urls_filtradas = set(df["URL"]) - {""}
        kw_filtrado = df_keywords[df_keywords["url"].isin(urls_filtradas)]

        if kw_filtrado.empty:
            st.info(
                "Nenhuma matéria do período/busca filtrados possui palavras-chave "
                "extraídas ainda. Rode extrair_keywords.py para atualizar "
                "materias_keywords.csv, ou amplie os filtros na barra lateral."
            )
        else:
            todas_kw = [
                termo
                for lista in kw_filtrado["palavras_chave_lista"]
                for termo in lista
                if termo and str(termo).strip()
            ]

            if not todas_kw:
                st.info("As matérias filtradas não possuem palavras-chave extraídas.")
            else:
                contagem_kw = (
                    pd.Series(Counter(todas_kw))
                    .sort_values(ascending=False)
                )

                top_n_kw = st.slider(
                    "Quantidade de palavras-chave em destaque",
                    min_value=10, max_value=60, value=25,
                    key="top_n_kw"
                )

                kw_top = contagem_kw.head(top_n_kw)

                # Nuvem de chips: tamanho da fonte proporcional à frequência
                freq_max = float(kw_top.max())
                freq_min = float(kw_top.min())

                def _tamanho_chip(freq):
                    if freq_max == freq_min:
                        return 1.15
                    return 0.85 + (freq - freq_min) / (freq_max - freq_min) * 1.25

                chips_html = "".join(
                    '<span class="chip-nuvem {tom}" style="font-size:{tam:.2f}rem;">{termo}</span>'.format(
                        tom="tom-a" if i % 2 == 0 else "tom-b",
                        tam=_tamanho_chip(freq),
                        termo=html.escape(str(termo))
                    )
                    for i, (termo, freq) in enumerate(kw_top.items())
                )

                st.markdown(
                    f'<div class="nuvem-container">{chips_html}</div>',
                    unsafe_allow_html=True
                )

                with st.expander("Ver como tabela e gráfico"):
                    df_kw_top = kw_top.reset_index()
                    df_kw_top.columns = ["Palavra-chave", "Frequência"]

                    col_kw_tab, col_kw_graf = st.columns([1, 2])

                    with col_kw_tab:
                        st.dataframe(
                            df_kw_top, use_container_width=True, hide_index=True
                        )

                    with col_kw_graf:
                        fig_kw = px.bar(
                            df_kw_top.sort_values("Frequência"),
                            x="Frequência",
                            y="Palavra-chave",
                            orientation="h",
                            title="Palavras-chave mais frequentes (IA)",
                            text="Frequência"
                        )
                        st.plotly_chart(fig_kw, use_container_width=True)

                st.caption(
                    f"Cobertura: {numero_br(len(kw_filtrado))} de {numero_br(len(df))} "
                    "matéria(s) no filtro atual possuem palavras-chave extraídas."
                )

        st.divider()

        st.subheader("📈 Termos em ascensão")
        st.caption(
            "Diferente do ranking acima (que mostra o que já está em alta), esta "
            "seção compara a taxa de menções de cada palavra-chave na janela mais "
            "recente do período filtrado com a taxa no restante do período — "
            "sinalizando temas que estão acelerando, mesmo que ainda não tenham "
            "volume alto o suficiente para aparecer no topo do ranking geral. "
            "O ranking usa um teste estatístico de significância (não percentual "
            "bruto), para não confundir ruído de amostra pequena (ex: 1 menção "
            "virar 2, um salto de \"+100%\" que não significa nada) com um "
            "aumento genuíno."
        )

        col_janela, col_topn, col_sensibilidade = st.columns(3)
        with col_janela:
            dias_janela_recente = st.slider(
                "Janela recente (dias)",
                min_value=7, max_value=30, value=14,
                key="dias_janela_ascensao"
            )
        with col_topn:
            top_n_ascensao = st.slider(
                "Quantidade de termos em destaque",
                min_value=5, max_value=20, value=10,
                key="top_n_ascensao"
            )
        with col_sensibilidade:
            sensibilidade_ascensao = st.select_slider(
                "Sensibilidade do sinal",
                options=["Alta confiança", "Confiança padrão", "Mais permissivo"],
                value="Confiança padrão",
                key="sensibilidade_ascensao"
            )

        LIMIAR_Z = {
            "Alta confiança": 2.33,     # ~99% de confiança unicaudal
            "Confiança padrão": 1.64,   # ~95% de confiança unicaudal
            "Mais permissivo": 1.28,    # ~90% de confiança unicaudal
        }[sensibilidade_ascensao]

        @st.cache_data
        def detectar_termos_ascensao(chave_cache, dias_recente_param, top_n_param, limiar_z, min_ocorrencias=3):
            """Compara, para cada palavra-chave, a taxa de menções (por dia) na
            janela recente do período filtrado com a taxa no restante do
            período (linha de base), usando um teste de significância para
            duas taxas de Poisson independentes (aproximação de Wald):

                z = (taxa_recente - taxa_base) / sqrt(taxa_recente/dias_recente + taxa_base/dias_base)

            Só termos com |z| acima do limiar escolhido (equivalente a um
            nível de confiança) são sinalizados como 'em ascensão'. Isso
            penaliza naturalmente contagens pequenas (que produzem z baixo,
            mesmo com percentual de variação alto) e recompensa saltos
            genuínos, mesmo quando o percentual parece modesto."""
            kw_com_data = df.merge(
                df_keywords[["url", "palavras_chave_lista"]],
                left_on="URL", right_on="url", how="inner"
            )

            if kw_com_data.empty:
                return {"status": "sem_dados"}

            data_min = kw_com_data["date_dt"].min()
            data_max = kw_com_data["date_dt"].max()
            dias_totais = (data_max - data_min).days + 1

            if dias_totais < dias_recente_param * 2:
                return {"status": "periodo_curto", "dias_totais": dias_totais}

            corte_recente = data_max - pd.Timedelta(days=dias_recente_param - 1)

            recente = kw_com_data[kw_com_data["date_dt"] >= corte_recente]
            base = kw_com_data[kw_com_data["date_dt"] < corte_recente]

            dias_recente_real = (data_max - corte_recente).days + 1
            dias_base_real = (corte_recente - data_min).days

            if dias_base_real <= 0:
                return {"status": "periodo_curto", "dias_totais": dias_totais}

            termos_recente = Counter(
                t for lista in recente["palavras_chave_lista"] for t in lista if t
            )
            termos_base = Counter(
                t for lista in base["palavras_chave_lista"] for t in lista if t
            )

            todos_termos = set(termos_recente) | set(termos_base)
            registros = []

            for termo in todos_termos:
                n_recente = termos_recente.get(termo, 0)
                n_base = termos_base.get(termo, 0)

                if (n_recente + n_base) < min_ocorrencias:
                    continue

                taxa_recente = n_recente / dias_recente_real
                taxa_base = n_base / dias_base_real

                if taxa_recente <= taxa_base:
                    continue  # só nos interessa quem está subindo de verdade

                variancia = (taxa_recente / dias_recente_real) + (taxa_base / dias_base_real)
                z_score = (taxa_recente - taxa_base) / math.sqrt(variancia) if variancia > 0 else 0.0

                if z_score < limiar_z:
                    continue  # sinal fraco demais para distinguir de ruído de amostra

                eh_novo = n_base == 0
                variacao_pct = (
                    ((taxa_recente - taxa_base) / taxa_base) * 100
                    if taxa_base > 0 else None
                )

                registros.append({
                    "Termo": termo,
                    "Menções (janela recente)": n_recente,
                    "Menções (linha de base)": n_base,
                    "Taxa recente (m/dia)": round(taxa_recente, 2),
                    "Taxa de base (m/dia)": round(taxa_base, 2),
                    "Variação": "novo" if eh_novo else f"+{variacao_pct:.0f}%",
                    "Força do sinal (z)": round(z_score, 2),
                })

            if not registros:
                return {"status": "sem_ascensao"}

            df_ascensao = pd.DataFrame(registros).sort_values(
                by="Força do sinal (z)",
                ascending=False
            ).head(top_n_param)

            return {
                "status": "ok",
                "tabela": df_ascensao,
                "dias_recente": dias_recente_real,
                "dias_base": dias_base_real,
                "corte_recente": corte_recente,
            }

        chave_cache_ascensao = (
            len(df), df["date_dt"].min(), df["date_dt"].max(),
            len(df_keywords), dias_janela_recente
        )
        resultado_ascensao = detectar_termos_ascensao(
            chave_cache_ascensao, dias_janela_recente, top_n_ascensao, LIMIAR_Z
        )

        if resultado_ascensao["status"] == "sem_dados":
            st.info(
                "Nenhuma matéria do período/busca filtrados possui palavras-chave "
                "extraídas ainda."
            )
        elif resultado_ascensao["status"] == "periodo_curto":
            st.info(
                f"O período filtrado tem {resultado_ascensao['dias_totais']} dia(s), "
                f"curto demais para comparar com uma janela recente de "
                f"{dias_janela_recente} dias. Amplie o período filtrado na barra "
                "lateral ou reduza a janela recente."
            )
        elif resultado_ascensao["status"] == "sem_ascensao":
            st.info(
                "Nenhum termo passou no teste de significância deste filtro — "
                "sem sinais de ascensão estatisticamente distinguíveis de ruído. "
                "Tente \"Mais permissivo\" na sensibilidade, ou amplie o período."
            )
        else:
            tabela_ascensao = resultado_ascensao["tabela"]

            chips_ascensao_html = "".join(
                '<span class="chip-nuvem {tom}">{termo} <b>({variacao})</b></span>'.format(
                    tom="tom-b" if row["Variação"] == "novo" else "tom-a",
                    termo=html.escape(str(row["Termo"])),
                    variacao=html.escape(str(row["Variação"]))
                )
                for _, row in tabela_ascensao.iterrows()
            )

            st.markdown(
                f'<div class="nuvem-container">{chips_ascensao_html}</div>',
                unsafe_allow_html=True
            )

            st.caption(
                f"Janela recente: últimos {resultado_ascensao['dias_recente']} dia(s) "
                f"(desde {resultado_ascensao['corte_recente'].strftime('%d/%m/%Y')}) · "
                f"linha de base: os {resultado_ascensao['dias_base']} dia(s) anteriores "
                f"do período filtrado · limiar de significância: z ≥ {LIMIAR_Z:.2f} "
                f"({sensibilidade_ascensao.lower()}). \"novo\" = termo que não "
                "aparecia na linha de base."
            )

            with st.expander("Ver como tabela (inclui força do sinal)"):
                st.dataframe(
                    tabela_ascensao.sort_values("Força do sinal (z)", ascending=False),
                    use_container_width=True, hide_index=True
                )

        st.divider()

    else:
        st.info(
            "O arquivo materias_keywords.csv ainda não foi encontrado. "
            "Rode extrair_keywords.py para habilitar as palavras-chave por IA nesta aba."
        )
        st.divider()

    st.subheader("Temas mais frequentes")

    fonte_texto = st.radio(
        "Analisar",
        ["Títulos e textos", "Apenas títulos", "Apenas textos"],
        horizontal=True
    )

    top_n = st.slider("Quantidade de termos", 10, 100, 30)

    if fonte_texto == "Títulos e textos":
        textos = tuple((df["Título"] + " " + df["Texto"]).tolist())
    elif fonte_texto == "Apenas títulos":
        textos = tuple(df["Título"].tolist())
    else:
        textos = tuple(df["Texto"].tolist())

    col_palavras, col_bigramas, col_trigramas = st.columns(3)

    with col_palavras:
        st.markdown("### Palavras")
        palavras = contar_termos(textos, n=1, top_n=top_n)
        st.dataframe(palavras, use_container_width=True, hide_index=True)

    with col_bigramas:
        st.markdown("### Bigramas")
        bigramas = contar_termos(textos, n=2, top_n=top_n)
        st.dataframe(bigramas, use_container_width=True, hide_index=True)

    with col_trigramas:
        st.markdown("### Trigramas")
        trigramas = contar_termos(textos, n=3, top_n=top_n)
        st.dataframe(trigramas, use_container_width=True, hide_index=True)

    st.divider()

    st.subheader("Ranking visual de termos")

    tipo_termo = st.selectbox(
        "Tipo de termo",
        ["Palavras", "Bigramas", "Trigramas"]
    )

    if tipo_termo == "Palavras":
        termos_grafico = palavras
    elif tipo_termo == "Bigramas":
        termos_grafico = bigramas
    else:
        termos_grafico = trigramas

    termos_grafico = termos_grafico.sort_values("Frequência", ascending=True)

    fig_termos = px.bar(
        termos_grafico,
        x="Frequência",
        y="Termo",
        orientation="h",
        title=f"{tipo_termo} mais frequentes",
        text="Frequência"
    )

    st.plotly_chart(fig_termos, use_container_width=True)

    st.divider()

    st.subheader("Evolução de um termo")

    termo_busca = st.text_input(
        "Digite um termo para acompanhar ao longo do tempo",
        value="vacina"
    )

    if termo_busca:
        termo = termo_busca.lower()

        df_termo = df.copy()
        df_termo["Ocorrências"] = (
            (df_termo["Título"] + " " + df_termo["Texto"])
            .str.lower()
            .str.count(re.escape(termo))
        )

        df_termo = df_termo[df_termo["Ocorrências"] > 0]

        if df_termo.empty:
            st.warning("Nenhuma ocorrência encontrada para esse termo.")
        else:
            evolucao = agrupar_por_escala(df_termo, escala, coluna_valor="Ocorrências")

            fig_evolucao = px.line(
                evolucao,
                x="Período",
                y="Ocorrências",
                markers=True,
                title=f"Evolução do termo: {termo_busca}"
            )

            fig_evolucao.update_layout(
                xaxis_title="Período",
                yaxis_title="Ocorrências",
                xaxis_type="category"
            )

            st.plotly_chart(fig_evolucao, use_container_width=True)

            st.markdown("### Matérias em que o termo aparece")

            tabela_termo = df_termo[
                ["Data", "Título", "Editoria", "Autor", "Ocorrências", "URL"]
            ].copy()

            tabela_termo = tabela_termo.sort_values(
                "Ocorrências",
                ascending=False
            )

            st.dataframe(
                tabela_termo,
                use_container_width=True,
                hide_index=True
            )

    st.divider()

    st.subheader("Top termos animado por mês")

    tipo_animacao = st.selectbox(
        "Tipo de termo na animação",
        ["Palavras", "Bigramas", "Trigramas"],
        key="tipo_animacao"
    )

    top_animacao = st.slider(
        "Quantidade de termos por mês",
        5, 20, 10,
        key="top_animacao"
    )

    if tipo_animacao == "Palavras":
        n_animacao = 1
    elif tipo_animacao == "Bigramas":
        n_animacao = 2
    else:
        n_animacao = 3

    @st.cache_data
    def gerar_dados_animacao(df_hash_key, n_animacao, top_animacao):
        """Cacheada por (período filtrado, tipo de n-grama, top_n).
        df_hash_key é uma tupla derivada do df já filtrado, para servir
        de chave de cache estável (DataFrames não são hashable)."""
        registros = []

        for mes, grupo in df.groupby("mes_dt"):
            textos_mes = tuple(
                (grupo["Título"].fillna("") + " " + grupo["Texto"].fillna("")).tolist()
            )

            termos_mes = contar_termos(
                textos_mes,
                n=n_animacao,
                top_n=top_animacao
            )

            termos_mes["Mês"] = mes.strftime("%m/%Y")
            termos_mes["mes_ordem"] = mes

            registros.append(termos_mes)

        return registros

    # Chave simples para invalidar o cache quando o df filtrado mudar
    chave_cache_animacao = (len(df), df["date_dt"].min(), df["date_dt"].max())
    registros = gerar_dados_animacao(chave_cache_animacao, n_animacao, top_animacao)

    if registros:
        df_animado = pd.concat(registros, ignore_index=True)

        df_animado = df_animado.sort_values(
            ["mes_ordem", "Frequência"],
            ascending=[True, True]
        )

        fig_animado = px.bar(
            df_animado,
            x="Frequência",
            y="Termo",
            orientation="h",
            animation_frame="Mês",
            range_x=[
                0,
                int(df_animado["Frequência"].max() * 1.15)
            ],
            title=f"Top {top_animacao} {tipo_animacao.lower()} por mês",
            text="Frequência"
        )

        fig_animado.update_layout(
            xaxis_title="Frequência",
            yaxis_title="Termo",
            yaxis={"categoryorder": "total ascending"},
            height=650
        )

        fig_animado.update_traces(
            textposition="outside",
            hovertemplate=(
                "<b>Termo:</b> %{y}<br>"
                "<b>Frequência:</b> %{x}<extra></extra>"
            )
        )

        st.plotly_chart(fig_animado, use_container_width=True)

    else:
        st.warning("Não há dados suficientes para gerar a animação.")

    if df_keywords is not None:
        st.divider()
        st.subheader("🧠 Top palavras-chave animado por mês (IA)")
        st.caption(
            "Mesma ideia da animação acima, mas usando as palavras-chave extraídas "
            "por IA em vez de n-gramas — costuma mostrar uma evolução mais limpa "
            "dos temas, sem ruído de palavras genéricas."
        )

        top_animacao_kw = st.slider(
            "Quantidade de palavras-chave por mês",
            5, 20, 10,
            key="top_animacao_kw"
        )

        @st.cache_data
        def gerar_dados_animacao_keywords(chave_cache, top_n):
            """Casa as palavras-chave (via URL) com as matérias já filtradas
            e conta ocorrências por mês. chave_cache invalida o cache quando
            o df filtrado ou o materias_keywords.csv mudarem."""
            kw_com_data = df.merge(
                df_keywords[["url", "palavras_chave_lista"]],
                left_on="URL", right_on="url", how="inner"
            )

            registros = []
            for mes, grupo in kw_com_data.groupby("mes_dt"):
                termos_mes = [
                    termo
                    for lista in grupo["palavras_chave_lista"]
                    for termo in lista
                    if termo and str(termo).strip()
                ]
                if not termos_mes:
                    continue

                contagem_mes = (
                    pd.Series(Counter(termos_mes))
                    .sort_values(ascending=False)
                    .head(top_n)
                )
                df_mes = contagem_mes.reset_index()
                df_mes.columns = ["Termo", "Frequência"]
                df_mes["Mês"] = mes.strftime("%m/%Y")
                df_mes["mes_ordem"] = mes

                registros.append(df_mes)

            return registros

        chave_cache_kw_animado = (
            len(df), df["date_dt"].min(), df["date_dt"].max(), len(df_keywords)
        )
        registros_kw = gerar_dados_animacao_keywords(
            chave_cache_kw_animado, top_animacao_kw
        )

        if registros_kw:
            df_animado_kw = pd.concat(registros_kw, ignore_index=True)
            df_animado_kw = df_animado_kw.sort_values(
                ["mes_ordem", "Frequência"],
                ascending=[True, True]
            )

            fig_animado_kw = px.bar(
                df_animado_kw,
                x="Frequência",
                y="Termo",
                orientation="h",
                animation_frame="Mês",
                range_x=[
                    0,
                    int(df_animado_kw["Frequência"].max() * 1.15)
                ],
                title=f"Top {top_animacao_kw} palavras-chave (IA) por mês",
                text="Frequência",
                color_discrete_sequence=["#5FBFA0"]
            )

            fig_animado_kw.update_layout(
                xaxis_title="Frequência",
                yaxis_title="Palavra-chave",
                yaxis={"categoryorder": "total ascending"},
                height=650
            )

            fig_animado_kw.update_traces(
                textposition="outside",
                hovertemplate=(
                    "<b>Palavra-chave:</b> %{y}<br>"
                    "<b>Frequência:</b> %{x}<extra></extra>"
                )
            )

            st.plotly_chart(fig_animado_kw, use_container_width=True)
        else:
            st.info(
                "Nenhuma matéria do período filtrado possui palavras-chave "
                "extraídas com dados de mês suficientes para animar."
            )


if secao_selecionada == "Sismógrafo":
    mostrar_periodo_no_topo(periodo_label_sidebar, escala)

    st.subheader("📡 Sismógrafo de cobertura")
    st.caption(
        "Detecta automaticamente os picos de publicação no período filtrado "
        "e transforma cada um em um boletim — a matéria mais representativa daquele momento."
    )

    n_eventos = st.slider(
        "Quantidade de picos a destacar",
        min_value=4, max_value=15, value=8,
        key="n_eventos_sismografo"
    )

    MAPA_COL_ESCALA = {"Dia": "date_dt", "Mês": "mes_dt", "Ano": "ano_dt"}

    @st.cache_data
    def detectar_eventos_destaque(chave_cache, escala_evt, n_eventos_evt):
        """Identifica picos estatísticos de cobertura (contagem acima de
        média + 1 desvio-padrão) e extrai, para cada pico, a matéria mais
        longa publicada naquele período como 'manchete' representativa.

        chave_cache serve apenas para invalidar o cache quando o df
        filtrado mudar (DataFrames não são hashable)."""
        serie = agrupar_por_escala(df, escala_evt)

        if len(serie) < 2:
            return []

        media = serie["Matérias"].mean()
        desvio = serie["Matérias"].std()
        limiar = media + desvio if desvio and desvio > 0 else media

        candidatos = serie[serie["Matérias"] >= limiar].copy()
        if len(candidatos) < 2:
            candidatos = serie.copy()

        candidatos = candidatos.nlargest(
            min(n_eventos_evt, len(candidatos)), "Matérias"
        )

        col_dt = MAPA_COL_ESCALA[escala_evt]
        eventos = []

        for _, linha in candidatos.iterrows():
            periodo_dt = linha[col_dt]

            if escala_evt == "Dia":
                inicio_p = periodo_dt
                fim_p = periodo_dt + pd.Timedelta(days=1)
                rotulo = periodo_dt.strftime("%d/%m/%Y")
            elif escala_evt == "Mês":
                inicio_p = periodo_dt
                fim_p = periodo_dt + pd.offsets.MonthBegin(1)
                rotulo = periodo_dt.strftime("%m/%Y")
            else:
                inicio_p = periodo_dt
                fim_p = periodo_dt + pd.offsets.YearBegin(1)
                rotulo = periodo_dt.strftime("%Y")

            grupo = df[(df[col_dt] >= inicio_p) & (df[col_dt] < fim_p)]
            if grupo.empty:
                continue

            principal = grupo.loc[grupo["Palavras"].idxmax()]

            texto = str(principal["Texto"]).strip()
            resumo = texto[:220] + ("…" if len(texto) > 220 else "")

            url = str(principal["URL"]) if str(principal["URL"]).startswith("http") else ""

            eventos.append({
                "data": rotulo,
                "data_ordenacao": periodo_dt.isoformat(),
                "titulo": principal["Título"] or "(sem título)",
                "editoria": principal["Editoria"] or "—",
                "autor": principal["Autor"] or "Redação",
                "url": url,
                "resumo": resumo if resumo.strip() else "Resumo indisponível para esta matéria.",
                "intensidade": int(linha["Matérias"]),
            })

        eventos.sort(key=lambda e: e["data_ordenacao"])
        return eventos

    chave_cache_sismografo = (len(df), df["date_dt"].min(), df["date_dt"].max())
    eventos_destaque = detectar_eventos_destaque(
        chave_cache_sismografo, escala, n_eventos
    )

    if len(eventos_destaque) < 2:
        st.info(
            "Dados insuficientes no período filtrado para montar o sismógrafo. "
            "Tente ampliar o intervalo de datas na barra lateral."
        )
    else:
        eventos_json = json.dumps(eventos_destaque, ensure_ascii=False)

        SISMOGRAFO_HTML = """
<style>
  #sismografo-root {
    --ink: #10151F;
    --ink-2: #171E2C;
    --paper: #E8E4D8;
    --muted: #8B93A7;
    --grid: #2A3244;
    --teal: #5FBFA0;
    --amber: #E8A33D;
    --fonte-serif: Georgia, 'Iowan Old Style', 'Palatino Linotype', 'Times New Roman', serif;
    --fonte-mono: 'SF Mono', 'Cascadia Code', Consolas, 'Liberation Mono', Menlo, monospace;
    --fonte-sans: -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    background: var(--ink);
    border-radius: 18px;
    padding: 28px 26px 22px;
    font-family: var(--fonte-sans);
    color: var(--paper);
    position: relative;
    overflow: hidden;
  }
  #sismografo-root .rotulo-topo {
    font-family: var(--fonte-mono);
    font-size: 0.72rem;
    letter-spacing: 0.14em;
    color: var(--teal);
    text-transform: uppercase;
    margin-bottom: 26px;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  #sismografo-root .rotulo-topo::before {
    content: "";
    width: 7px; height: 7px;
    border-radius: 50%;
    background: var(--amber);
    box-shadow: 0 0 8px var(--amber);
    animation: piscar 1.6s ease-in-out infinite;
  }
  @keyframes piscar { 0%,100% { opacity: 1; } 50% { opacity: 0.25; } }

  #waveform-track {
    position: relative;
    height: 210px;
    margin: 0 6px 8px;
    border-bottom: 2px solid var(--grid);
  }
  #waveform-track::before {
    content: "";
    position: absolute;
    left: 0; right: 0; top: 0; bottom: 0;
    background-image: repeating-linear-gradient(to top, transparent, transparent 34px, var(--grid) 35px);
    opacity: 0.35;
    pointer-events: none;
  }
  .scan-line {
    position: absolute;
    top: 0; bottom: 8px;
    width: 2px;
    background: linear-gradient(to bottom, transparent, var(--amber) 20%, var(--amber) 80%, transparent);
    box-shadow: 0 0 14px var(--amber);
    left: 0;
    animation: varrer 1.9s cubic-bezier(.4,0,.2,1) forwards;
    z-index: 3;
  }
  @keyframes varrer { from { left: 0%; } to { left: 100%; } }

  .pulso {
    position: absolute;
    bottom: 0;
    width: 5px;
    height: var(--altura, 20%);
    background: linear-gradient(to top, var(--teal), rgba(95,191,160,0.05));
    border-radius: 3px 3px 0 0;
    transform: scaleY(0);
    transform-origin: bottom;
    opacity: 0;
    transition: opacity 0.5s ease, filter 0.3s ease;
    cursor: pointer;
  }
  .pulso.alerta { background: linear-gradient(to top, var(--amber), rgba(232,163,61,0.05)); }
  .pulso.visivel { opacity: 1; animation: subir 0.55s ease-out forwards; }
  @keyframes subir { to { transform: scaleY(1); } }
  .pulso.selecionado { filter: drop-shadow(0 0 10px var(--teal)); }
  .pulso.alerta.selecionado { filter: drop-shadow(0 0 10px var(--amber)); }
  .pulso:hover { filter: brightness(1.3); }

  .marcador {
    position: absolute;
    top: -7px; left: 50%;
    transform: translateX(-50%);
    width: 12px; height: 12px;
    border-radius: 50%;
    background: var(--ink);
    border: 2px solid var(--teal);
    transition: transform 0.2s ease;
  }
  .pulso.alerta .marcador { border-color: var(--amber); }
  .pulso:hover .marcador { transform: translateX(-50%) scale(1.35); }

  .rotulo-data {
    position: absolute;
    bottom: -24px; left: 50%;
    transform: translateX(-50%) rotate(-38deg);
    transform-origin: top left;
    font-family: var(--fonte-mono);
    font-size: 0.68rem;
    color: var(--muted);
    white-space: nowrap;
  }

  #boletim {
    margin-top: 46px;
    background: var(--ink-2);
    border: 1px solid var(--grid);
    border-left: 3px solid var(--amber);
    border-radius: 10px;
    padding: 20px 24px;
    transition: opacity 0.3s ease;
  }
  #boletim.oculto { opacity: 0; }
  #bt-cabecalho {
    font-family: var(--fonte-mono);
    font-size: 0.72rem;
    letter-spacing: 0.08em;
    color: var(--amber);
    margin-bottom: 12px;
  }
  #bt-titulo {
    font-family: var(--fonte-serif);
    font-weight: 700;
    font-size: 1.5rem;
    line-height: 1.3;
    color: var(--paper);
    min-height: 1.3em;
    border-right: 2px solid transparent;
  }
  #bt-autor {
    font-family: var(--fonte-sans);
    font-size: 0.85rem;
    color: var(--muted);
    margin: 8px 0 14px;
    font-style: italic;
  }
  #bt-resumo {
    font-family: var(--fonte-sans);
    font-size: 0.95rem;
    line-height: 1.55;
    color: #C7C2B4;
  }
  #bt-link {
    display: inline-block;
    margin-top: 16px;
    font-family: var(--fonte-mono);
    font-size: 0.78rem;
    letter-spacing: 0.05em;
    color: var(--teal);
    text-decoration: none;
    border: 1px solid var(--teal);
    border-radius: 999px;
    padding: 7px 16px;
    transition: background 0.2s ease, color 0.2s ease;
  }
  #bt-link:hover { background: var(--teal); color: var(--ink); }
  #bt-link.escondido { display: none; }
</style>

<div id="sismografo-root">
  <div class="rotulo-topo">Leitura em tempo real — intensidade de cobertura</div>
  <div id="waveform-track">
    <div class="scan-line"></div>
  </div>
  <div id="boletim" class="oculto">
    <div id="bt-cabecalho">BOLETIM</div>
    <div id="bt-titulo"></div>
    <div id="bt-autor"></div>
    <div id="bt-resumo"></div>
    <a id="bt-link" href="#" target="_blank" rel="noopener">Ler matéria completa →</a>
  </div>
</div>

<script>
(function() {
  const eventos = __EVENTOS_JSON__;
  const track = document.getElementById('waveform-track');
  const maxIntensidade = Math.max(...eventos.map(e => e.intensidade));
  const minDate = new Date(eventos[0].data_ordenacao).getTime();
  const maxDate = new Date(eventos[eventos.length - 1].data_ordenacao).getTime();
  const spanDate = Math.max(maxDate - minDate, 1);

  eventos.forEach((ev, i) => {
    const t = new Date(ev.data_ordenacao).getTime();
    const xPct = spanDate > 0 ? ((t - minDate) / spanDate) * 92 + 4 : 50;
    const alturaPct = Math.max((ev.intensidade / maxIntensidade) * 85, 18);

    const pulso = document.createElement('div');
    pulso.className = 'pulso' + (ev.intensidade === maxIntensidade ? ' alerta' : '');
    pulso.style.left = xPct + '%';
    pulso.style.setProperty('--altura', alturaPct + '%');
    pulso.style.transitionDelay = (i * 70) + 'ms';
    pulso.dataset.index = i;

    const marcador = document.createElement('div');
    marcador.className = 'marcador';
    marcador.title = ev.data + ' — ' + ev.titulo + ' (' + ev.intensidade + ' matéria(s))';
    pulso.appendChild(marcador);

    const rotulo = document.createElement('div');
    rotulo.className = 'rotulo-data';
    rotulo.textContent = ev.data;
    pulso.appendChild(rotulo);

    pulso.addEventListener('click', function() { mostrarBoletim(i); });
    track.appendChild(pulso);
  });

  setTimeout(function() {
    document.querySelectorAll('.pulso').forEach(function(p) { p.classList.add('visivel'); });
  }, 350);

  function mostrarBoletim(i) {
    const ev = eventos[i];
    document.querySelectorAll('.pulso').forEach(function(p) { p.classList.remove('selecionado'); });
    const alvo = document.querySelector('.pulso[data-index="' + i + '"]');
    if (alvo) alvo.classList.add('selecionado');

    const boletim = document.getElementById('boletim');
    boletim.classList.remove('oculto');

    document.getElementById('bt-cabecalho').textContent =
      'BOLETIM ▸ ' + ev.data + ' ▸ ' + String(ev.editoria).toUpperCase() + ' ▸ ' + ev.intensidade + ' MATÉRIA(S) NO PERÍODO';
    document.getElementById('bt-autor').textContent = 'Por ' + ev.autor;
    document.getElementById('bt-resumo').textContent = ev.resumo;

    const link = document.getElementById('bt-link');
    if (ev.url) {
      link.href = ev.url;
      link.classList.remove('escondido');
    } else {
      link.classList.add('escondido');
    }

    const tituloEl = document.getElementById('bt-titulo');
    tituloEl.textContent = '';
    let idx = 0;
    if (window.__sismografoTyper) clearInterval(window.__sismografoTyper);
    window.__sismografoTyper = setInterval(function() {
      tituloEl.textContent += ev.titulo[idx] || '';
      idx++;
      if (idx >= ev.titulo.length) clearInterval(window.__sismografoTyper);
    }, 16);
  }

  setTimeout(function() { mostrarBoletim(eventos.length - 1); }, 2100);
})();
</script>
"""

        SISMOGRAFO_HTML = SISMOGRAFO_HTML.replace("__EVENTOS_JSON__", eventos_json)

        components.html(SISMOGRAFO_HTML, height=560, scrolling=False)

        st.caption(
            "Clique em qualquer marcador da linha para ver o boletim daquele pico de cobertura. "
            "Barras em âmbar indicam o maior pico do período filtrado."
        )

    renderizar_gauge_ritmo(
        "Ritmo de publicação (período selecionado)",
        "Período filtrado",
        ritmo_atual, ritmo_historico, razao_temperatura
    )


if secao_selecionada == "Análise IA":
    mostrar_periodo_no_topo(periodo_label_sidebar, escala)
    st.subheader("Análise estruturada por IA")
    st.caption(
        "Agrega, sem nenhuma chamada de IA nova, as análises diárias já geradas "
        "(analise_diaria.jsonl) para os dias que caem no filtro atual da barra "
        "lateral. Cada dia é analisado pela IA só uma vez, então filtrar datas "
        "aqui não tem custo adicional."
    )

    datas_no_filtro = set(df["Data"].unique())
    resultado_periodo = agregar_analises_periodo(analises_diarias, datas_no_filtro)

    if not resultado_periodo:
        st.warning(
            "Nenhum dia do período/busca filtrados possui análise diária de IA "
            "ainda. Rode analise_diaria.py nos dias faltantes para preencher "
            "o histórico."
        )
    else:
        st.caption(
            f"📊 {resultado_periodo['n_dias']} dia(s) com análise disponível no "
            f"período, totalizando {numero_br(resultado_periodo['n_materias'])} matéria(s)."
        )

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Temas em destaque")
            for item in resultado_periodo["temas_em_destaque"]:
                st.write(f"• {item}")

            st.markdown("### Enquadramentos predominantes")
            for item in resultado_periodo["enquadramentos_predominantes"]:
                st.write(f"• {item}")

            st.markdown("### Áreas predominantes")
            for item in resultado_periodo["areas_predominantes"]:
                st.write(f"• {item}")

        with col2:
            st.markdown("### Atores mais visíveis")
            for item in resultado_periodo["atores_mais_visiveis"]:
                st.write(f"• {item}")

            st.markdown("### Abrangência predominante")
            st.write(f"• {resultado_periodo['abrangencia_predominante']}")

        st.divider()

        st.markdown("### Tendência do período")
        st.caption(
            "Este campo não dá para agregar por contagem — é texto corrido. "
            "Gere uma síntese sob demanda (1 chamada de IA, usando só as frases "
            "diárias já existentes, não as matérias originais)."
        )

        chave_periodo = tuple(resultado_periodo["tendencias_diarias"])

        if st.button("🧬 Gerar síntese do período", key="btn_sintese_periodo"):
            with st.spinner("Sintetizando tendência do período..."):
                try:
                    texto_sintese = sintetizar_tendencia_periodo(
                        resultado_periodo["tendencias_diarias"]
                    )
                    st.session_state["sintese_periodo_texto"] = texto_sintese
                    st.session_state["sintese_periodo_chave"] = chave_periodo
                except Exception as e:
                    st.error(f"Não foi possível gerar a síntese: {e}")

        sintese_valida = (
            st.session_state.get("sintese_periodo_chave") == chave_periodo
            and st.session_state.get("sintese_periodo_texto")
        )

        if sintese_valida:
            st.info(st.session_state["sintese_periodo_texto"])
        else:
            with st.expander("Ver frases diárias sem sintetizar"):
                for data_item, texto_item in resultado_periodo["tendencias_diarias"]:
                    st.write(f"**{data_item}** — {texto_item}")


@st.cache_data
def construir_elementos_rede(listas_termos, top_clusters=8):
    """Constrói o grafo de coocorrência de palavras-chave e detecta
    comunidades via Louvain — mesma lógica de rede_semantica.py, mas
    rodando sob demanda sobre qualquer conjunto de matérias (aqui, as do
    período filtrado pela sidebar), em vez de um ano inteiro pré-processado."""
    G = nx.Graph()
    frequencia_nos = Counter()

    for termos in listas_termos:
        termos = [t.strip() for t in termos if isinstance(t, str) and len(t.strip()) > 2]
        termos = list(dict.fromkeys(termos))

        for termo in termos:
            frequencia_nos[termo] += 1
            G.add_node(termo)

        for a, b in combinations(termos, 2):
            if G.has_edge(a, b):
                G[a][b]["weight"] += 1
            else:
                G.add_edge(a, b, weight=1)

    arestas_fracas = [(a, b) for a, b, d in G.edges(data=True) if d["weight"] < 2]
    G.remove_edges_from(arestas_fracas)
    G.remove_nodes_from(list(nx.isolates(G)))

    if len(G.nodes()) == 0:
        return [], []

    particao = community_louvain.best_partition(G, weight="weight", random_state=42)
    freq_comunidades = Counter(particao.values())

    maiores_comunidades = [c for c, _ in freq_comunidades.most_common(top_clusters)]
    comunidade_para_indice = {c: i for i, c in enumerate(maiores_comunidades)}

    cores = [
        "#5FBFA0", "#E8A33D", "#8B93A7", "#9467BD",
        "#E07A5F", "#4D908E", "#F2CC8F", "#277DA1"
    ]

    termos_por_comunidade = defaultdict(list)
    for node in G.nodes():
        termos_por_comunidade[particao[node]].append(node)

    rotulos_comunidades = {}
    for comunidade in maiores_comunidades:
        termos_ordenados = sorted(
            termos_por_comunidade[comunidade],
            key=lambda t: frequencia_nos[t],
            reverse=True
        )
        rotulos_comunidades[comunidade] = ", ".join(termos_ordenados[:4])

    elementos = []

    # Nós-container por comunidade (compound nodes) — viram as "bolhas de
    # fundo" que agrupam visualmente os termos de cada cluster.
    for comunidade in maiores_comunidades:
        indice = comunidade_para_indice[comunidade]
        elementos.append({
            "data": {
                "id": f"cluster_{indice}",
                "label": rotulos_comunidades[comunidade],
                "cor_cluster": cores[indice % len(cores)],
                "tipo": "cluster"
            }
        })

    tem_outras = False

    for node in G.nodes():
        freq = frequencia_nos[node]
        grau = G.degree(node)
        forca = sum(G[node][viz]["weight"] for viz in G.neighbors(node))
        tamanho_no = 45 + math.sqrt(freq) * 18

        comunidade_original = particao[node]
        if comunidade_original in comunidade_para_indice:
            indice_cluster = comunidade_para_indice[comunidade_original]
            cor = cores[indice_cluster % len(cores)]
            rotulo_cluster = rotulos_comunidades[comunidade_original]
            parent_id = f"cluster_{indice_cluster}"
        else:
            indice_cluster = 999
            cor = "#3A4258"
            rotulo_cluster = "Outras comunidades"
            parent_id = "cluster_outras"
            tem_outras = True

        elementos.append({
            "data": {
                "id": node,
                "label": node,
                "freq": freq,
                "grau": grau,
                "forca": forca,
                "size": tamanho_no,
                "cluster": indice_cluster,
                "cluster_label": rotulo_cluster,
                "color": cor,
                "tipo": "termo",
                "parent": parent_id
            }
        })

    if tem_outras:
        elementos.append({
            "data": {
                "id": "cluster_outras",
                "label": "Outras comunidades",
                "cor_cluster": "#3A4258",
                "tipo": "cluster"
            }
        })

    for a, b, d in G.edges(data=True):
        peso = d["weight"]
        elementos.append({
            "data": {
                "id": f"{a}__{b}",
                "source": a,
                "target": b,
                "weight": peso,
                "width": 1 + math.sqrt(peso)
            }
        })

    legenda = []
    for comunidade in maiores_comunidades:
        indice = comunidade_para_indice[comunidade]
        legenda.append({
            "nome": rotulos_comunidades[comunidade],
            "cor": cores[indice % len(cores)],
            "cluster": indice,
            "n": freq_comunidades[comunidade]
        })

    outras = sum(q for c, q in freq_comunidades.items() if c not in maiores_comunidades)
    if outras > 0:
        legenda.append({"nome": "Outras comunidades", "cor": "#3A4258", "cluster": 999, "n": outras})

    return elementos, legenda


def renderizar_rede_semantica_html(elementos, legenda, titulo):
    """Monta o HTML/Cytoscape.js da rede semântica, no mesmo estilo visual
    (ink/teal/âmbar) do resto do app. Inclui: layout fcose (com fallback
    para cose se o CDN falhar), halo por cluster, bolhas de fundo por
    comunidade (compound nodes), destaque de comunidade ao passar o mouse
    e entrada animada dos elementos."""
    elementos_json = json.dumps(elementos, ensure_ascii=False)
    legenda_json = json.dumps(legenda, ensure_ascii=False)
    titulo_escapado = html.escape(titulo)

    return f"""
<style>
  body {{ margin: 0; font-family: -apple-system, 'Segoe UI', Arial, sans-serif; background: #10151F; }}
  #cy {{ width: 100%; height: 720px; display: block; }}
  #titulo {{
    position: absolute; top: 16px; left: 20px; z-index: 10;
    background: rgba(23,30,44,0.92); padding: 10px 14px; border-radius: 8px;
    font-size: 15px; font-weight: bold; color: #E8E4D8;
    border: 1px solid #2A3244;
  }}
  #legenda {{
    position: absolute; top: 62px; left: 20px; z-index: 10;
    background: rgba(23,30,44,0.92); padding: 12px 14px; border-radius: 8px;
    font-size: 13px; color: #E8E4D8; max-width: 320px;
    border: 1px solid #2A3244;
  }}
  .legenda-titulo {{
    font-weight: bold; margin-bottom: 8px; font-family: 'SF Mono', Consolas, monospace;
    font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; color: #5FBFA0;
  }}
  .item-legenda {{ display: flex; align-items: flex-start; margin-bottom: 7px; gap: 8px; line-height: 1.25; }}
  .cor-legenda {{ width: 14px; height: 14px; border-radius: 50%; display: inline-block; flex: 0 0 14px; margin-top: 2px; }}
  .texto-legenda {{ flex: 1; }}
  .n-legenda {{ color: #8B93A7; font-size: 12px; }}
</style>

<div id="titulo">{titulo_escapado}</div>
<div id="legenda"></div>
<div id="cy"></div>

<script src="https://unpkg.com/cytoscape@3.28.1/dist/cytoscape.min.js"></script>
<script src="https://unpkg.com/layout-base/layout-base.js"></script>
<script src="https://unpkg.com/cose-base/cose-base.js"></script>
<script src="https://unpkg.com/cytoscape-fcose/cytoscape-fcose.js"></script>
<script>
const elementos = {elementos_json};
const legendaDados = {legenda_json};

// fcose dá um layout bem mais limpo pra grafos com clusters do que o cose
// padrão; se o CDN falhar por algum motivo, cai pro cose sem quebrar a página.
var fcoseDisponivel = (typeof cytoscapeFcose !== 'undefined');
if (fcoseDisponivel) {{
    cytoscape.use(cytoscapeFcose);
}}

const legenda = document.getElementById("legenda");
const tituloLegenda = document.createElement("div");
tituloLegenda.className = "legenda-titulo";
tituloLegenda.innerText = "Principais comunidades";
legenda.appendChild(tituloLegenda);

legendaDados.forEach(function(item) {{
    const linha = document.createElement("div");
    linha.className = "item-legenda";
    const cor = document.createElement("span");
    cor.className = "cor-legenda";
    cor.style.backgroundColor = item.cor;
    const texto = document.createElement("div");
    texto.className = "texto-legenda";
    texto.innerHTML = item.nome + "<br><span class='n-legenda'>" + item.n + " termos</span>";
    linha.appendChild(cor);
    linha.appendChild(texto);
    legenda.appendChild(linha);
}});

const cy = cytoscape({{
    container: document.getElementById('cy'),
    elements: elementos,
    style: [
        {{ selector: 'node[tipo = "cluster"]', style: {{
            'shape': 'round-rectangle',
            'background-color': 'data(cor_cluster)',
            'background-opacity': 0.10,
            'border-width': 1,
            'border-color': 'data(cor_cluster)',
            'border-opacity': 0.4,
            'border-style': 'dashed',
            'label': 'data(label)',
            'text-valign': 'top',
            'text-halign': 'center',
            'text-margin-y': -8,
            'font-size': 11,
            'font-family': "'SF Mono', Consolas, monospace",
            'font-weight': 'bold',
            'text-transform': 'uppercase',
            'color': 'data(cor_cluster)',
            'text-opacity': 0.9,
            'padding': 26,
            'z-compound-depth': 'bottom'
        }} }},
        {{ selector: 'node[tipo = "termo"]', style: {{
            'width': 'data(size)', 'height': 'data(size)',
            'background-color': 'data(color)',
            'border-width': 2, 'border-color': '#E8E4D8',
            'label': 'data(label)', 'text-valign': 'center', 'text-halign': 'center',
            'text-wrap': 'wrap', 'text-max-width': '95px',
            'font-size': 16, 'font-family': 'Arial', 'font-weight': 'bold',
            'color': '#10151F', 'text-outline-width': 2, 'text-outline-color': '#E8E4D8',
            'overlay-color': 'data(color)',
            'overlay-opacity': 0.22,
            'overlay-padding': 10,
            'overlay-shape': 'ellipse',
            'z-compound-depth': 'top'
        }} }},
        {{ selector: 'edge', style: {{
            'width': 'data(width)', 'line-color': '#4A5468', 'opacity': 0.55, 'curve-style': 'bezier'
        }} }},
        {{ selector: 'node[tipo = "termo"]:hover', style: {{ 'border-width': 4, 'border-color': '#E8A33D', 'font-size': 20, 'overlay-opacity': 0.4 }} }},
        {{ selector: 'edge:hover', style: {{ 'line-color': '#E8A33D', 'opacity': 0.9 }} }},
        {{ selector: '.dimmed', style: {{ 'opacity': 0.10, 'text-opacity': 0.10 }} }}
    ],
    layout: {{ name: 'preset' }}
}});

// Entrada animada: tudo começa invisível, o layout calcula as posições em
// segundo plano, e só depois os elementos aparecem progressivamente.
cy.elements().style('opacity', 0);

const opcoesLayout = fcoseDisponivel ? {{
    name: 'fcose',
    quality: 'proof',
    animate: false,
    randomize: true,
    nodeSeparation: 90,
    idealEdgeLength: 120,
    nodeRepulsion: 8000,
    nestingFactor: 0.4,
    packComponents: true,
    fit: false
}} : {{
    name: 'cose',
    animate: false,
    randomize: true,
    nodeRepulsion: 800000,
    idealEdgeLength: 140,
    edgeElasticity: 80,
    gravity: 80,
    numIter: 1000,
    fit: false
}};

const layout = cy.layout(opcoesLayout);

layout.on('layoutstop', function() {{
    cy.fit(undefined, 40);

    const nosTermo = cy.nodes('[tipo = "termo"]');
    const nosCluster = cy.nodes('[tipo = "cluster"]');

    nosCluster.forEach(function(n) {{
        n.animate({{ style: {{ opacity: 1 }} }}, {{ duration: 500, easing: 'ease-out' }});
    }});

    nosTermo.forEach(function(n, i) {{
        n.delay(i * 22).animate({{ style: {{ opacity: 1 }} }}, {{ duration: 380, easing: 'ease-out' }});
    }});

    cy.edges().delay(nosTermo.length * 22 + 250).animate(
        {{ style: {{ opacity: 0.55 }} }}, {{ duration: 500, easing: 'ease-out' }}
    );
}});

layout.run();

// Destaque por comunidade ao passar o mouse (spotlight): esmaece tudo
// que não pertence à mesma bolha do nó em foco.
cy.on('mouseover', 'node[tipo = "termo"]', function(evt) {{
    const n = evt.target;
    const parentId = n.data('parent');

    let grupo = n.closedNeighborhood('node');
    if (parentId) {{
        grupo = cy.nodes('[parent = "' + parentId + '"]').union(cy.getElementById(parentId));
    }}

    const arestasInternas = grupo.connectedEdges().filter(function(e) {{
        return grupo.contains(e.source()) && grupo.contains(e.target());
    }});

    const destaque = grupo.union(arestasInternas);

    cy.batch(function() {{
        cy.elements().addClass('dimmed');
        destaque.removeClass('dimmed');
    }});

    const d = n.data();
    n.style('label', d.label + '\\nMatérias: ' + d.freq + '\\nConexões: ' + d.grau + '\\nComunidade: ' + d.cluster_label);
}});

cy.on('mouseout', 'node[tipo = "termo"]', function(evt) {{
    cy.batch(function() {{
        cy.elements().removeClass('dimmed');
    }});
    const n = evt.target;
    n.style('label', n.data('label'));
}});
</script>
"""


if secao_selecionada == "Rede semântica":
    mostrar_periodo_no_topo(periodo_label_sidebar, escala)

    st.subheader("🕸️ Rede semântica de palavras-chave")
    st.caption(
        "Grafo de coocorrência das palavras-chave extraídas por IA nas matérias "
        "do período filtrado na barra lateral — dois termos se conectam quando "
        "aparecem juntos na mesma matéria. Cores indicam comunidades detectadas "
        "automaticamente (algoritmo de Louvain). Muda dinamicamente com o filtro, "
        "sem precisar gerar arquivo nenhum antecipadamente."
    )

    if not REDE_SEMANTICA_DISPONIVEL:
        st.warning(
            "As bibliotecas networkx e python-louvain não estão instaladas neste "
            "ambiente. Rode: pip install networkx python-louvain"
        )
    elif df_keywords is None:
        st.info(
            "O arquivo materias_keywords.csv ainda não foi encontrado. "
            "Rode extrair_keywords.py para habilitar a rede semântica."
        )
    else:
        top_clusters = st.slider(
            "Quantidade de comunidades em destaque",
            min_value=3, max_value=12, value=8,
            key="top_clusters_rede"
        )

        urls_filtradas_rede = set(df["URL"]) - {""}
        kw_filtrado_rede = df_keywords[df_keywords["url"].isin(urls_filtradas_rede)]

        if kw_filtrado_rede.empty:
            st.info(
                "Nenhuma matéria do período/busca filtrados possui palavras-chave "
                "extraídas. Amplie o filtro na barra lateral ou rode extrair_keywords.py."
            )
        else:
            listas_termos = tuple(
                tuple(lista) for lista in kw_filtrado_rede["palavras_chave_lista"]
            )

            elementos, legenda = construir_elementos_rede(listas_termos, top_clusters)

            if not elementos:
                st.info(
                    "Poucas coocorrências de palavras-chave no período filtrado para "
                    "montar uma rede (os termos aparecem sozinhos ou só uma vez "
                    "acompanhados). Amplie o período selecionado na barra lateral."
                )
            else:
                periodo_label = (
                    f"{df['date_dt'].min().strftime('%d/%m/%Y')} a "
                    f"{df['date_dt'].max().strftime('%d/%m/%Y')}"
                )
                titulo_rede = (
                    f"Rede semântica — {periodo_label} "
                    f"({numero_br(len(kw_filtrado_rede))} matéria(s))"
                )

                html_rede = renderizar_rede_semantica_html(elementos, legenda, titulo_rede)
                components.html(html_rede, height=760, scrolling=False)

                st.caption(
                    f"{numero_br(len(elementos))} elemento(s) no grafo (nós + arestas) · "
                    f"{numero_br(len(kw_filtrado_rede))} matéria(s) com palavras-chave no período."
                )