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


_PADRAO_URL_DOMINIO = re.compile(r'(https?://|www\.|\.com\b|\.com\.br\b|\.org\b|\.net\b)', re.IGNORECASE)


def formatar_autor(valor):
    """Alguns registros trazem o autor como string de lista Python
    (ex: "['Fulano']", herdado do scraper) em vez do nome puro.
    Normaliza para exibição sem quebrar se o valor já vier limpo.

    Também filtra itens que pareçam URL/domínio (ex: 'www.facebook.com') —
    proteção extra caso algum scraper capture um link de compartilhamento
    junto com o nome do autor de verdade (já visto com a BBC)."""
    texto = str(valor).strip()

    if texto.startswith("[") and texto.endswith("]"):
        try:
            import ast
            lista = ast.literal_eval(texto)
            if isinstance(lista, list) and lista:
                itens_limpos = [
                    str(item).strip() for item in lista
                    if item and not _PADRAO_URL_DOMINIO.search(str(item))
                ]
                if itens_limpos:
                    return ", ".join(itens_limpos)
                return ", ".join(str(item).strip() for item in lista)
        except (ValueError, SyntaxError):
            pass

    return texto


ARQUIVOS_FONTES = [
    ("folha.csv", "Folha de S.Paulo"),
    ("cnn_brasil.csv", "CNN Brasil"),
    ("bbc_brasil.csv", "BBC News Brasil"),
    ("g1_globo.csv", "G1"),
]


@st.cache_data
def carregar_dados():
    dataframes = []

    for caminho, veiculo_padrao in ARQUIVOS_FONTES:
        if not os.path.exists(caminho):
            continue

        try:
            df_fonte = pd.read_csv(caminho, encoding="utf-8-sig")
        except Exception:
            st.warning(f"Não foi possível carregar os dados de {veiculo_padrao} — essa fonte foi ignorada nesta sessão.")
            continue

        colunas_faltando = [c for c in COLUNAS_ESPERADAS if c not in df_fonte.columns]
        if colunas_faltando:
            st.warning(f"Os dados de {veiculo_padrao} estão incompletos e foram ignorados nesta sessão.")
            continue

        # Arquivos antigos (ex: folha.csv coletado antes do suporte a múltiplos
        # veículos) podem não ter a coluna 'veiculo' — nesse caso, assume o
        # veículo padrão daquele arquivo.
        if "veiculo" not in df_fonte.columns:
            df_fonte["veiculo"] = veiculo_padrao
        else:
            df_fonte["veiculo"] = df_fonte["veiculo"].fillna(veiculo_padrao)

        dataframes.append(df_fonte)

    if not dataframes:
        st.error(
            "Nenhuma fonte de dados foi encontrada. Este é um problema de "
            "configuração do app — entre em contato com quem mantém o projeto."
        )
        st.stop()

    df = pd.concat(dataframes, ignore_index=True)

    df["date_dt"] = pd.to_datetime(
        df["date"],
        format="%d/%m/%Y",
        errors="coerce"
    )

    df = df.dropna(subset=["date_dt"])

    if df.empty:
        st.error("Nenhuma linha com data válida foi encontrada nos arquivos de dados.")
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
    df["Veículo"] = df["veiculo"].fillna("Desconhecido")

    # Os 4 scrapers já salvam a URL da imagem de capa de cada matéria (via
    # news-please), mas o app nunca expunha essa coluna. "ERRO!!!" é o
    # placeholder usado pelos scrapers quando a extração falhou — tratado
    # aqui como "sem imagem", não como uma URL de verdade.
    if "image_url" in df.columns:
        df["Imagem"] = df["image_url"].fillna("")
        df.loc[df["Imagem"] == "ERRO!!!", "Imagem"] = ""
    else:
        df["Imagem"] = ""

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


def analise_get(analise, chave, padrao=None):
    """Acesso seguro a chaves da análise diária, evitando KeyError
    se o formato do registro mudar ou vier incompleto."""
    if padrao is None:
        padrao = []
    valor = analise.get(chave, padrao)
    return valor if valor is not None else padrao


def agregar_analises_periodo(analises, datas_no_filtro, veiculos_no_filtro=None):
    """Agrega, em Python, as entradas de analise_diaria.jsonl — agora só
    contendo 'tendencia_do_periodo' por combinação (dia, veículo), já que
    enquadramento/área/abrangência/instituições/pessoas migraram para
    extrair_keywords.py (matéria por matéria). veiculos_no_filtro=None
    combina todos os veículos disponíveis."""
    analises_periodo = [
        a for a in analises
        if a.get("data") in datas_no_filtro
        and (veiculos_no_filtro is None or a.get("veiculo") in veiculos_no_filtro)
    ]

    if not analises_periodo:
        return None

    tendencias_diarias = []
    total_materias = 0
    datas_distintas = set()

    for a in analises_periodo:
        tendencia = a.get("tendencia_do_periodo")
        if tendencia:
            tendencias_diarias.append((a.get("data"), a.get("veiculo", "Desconhecido"), tendencia))

        total_materias += analise_get(a, "n_materias", 0)
        datas_distintas.add(a.get("data"))

    return {
        "n_dias": len(datas_distintas),
        "n_combinacoes": len(analises_periodo),
        "n_materias": total_materias,
        "tendencias_diarias": sorted(
            tendencias_diarias,
            key=lambda t: (pd.to_datetime(t[0], format="%d/%m/%Y", errors="coerce"), t[1])
        ),
    }


# Rótulos legíveis para os códigos de taxonomia salvos por
# extrair_keywords.py (frame_predominante e abrangencia vêm em
# MAIUSCULO_COM_UNDERSCORE, mais fácil de validar no prompt/código).
ROTULOS_FRAME = {
    "DESCOBERTA_CIENTIFICA": "Descoberta científica",
    "INOVACAO_TECNOLOGICA": "Inovação tecnológica",
    "PROMESSA_E_BENEFICIOS": "Promessa e benefícios",
    "RISCO_E_AMEACA": "Risco e ameaça",
    "INCERTEZA_CIENTIFICA": "Incerteza científica",
    "CONFLITO_E_CONTROVERSIA": "Conflito e controvérsia",
    "IMPACTO_SOCIAL": "Impacto social",
    "POLITICA_CIENTIFICA_E_GOVERNANCA": "Política científica e governança",
    "ETICA_E_MORALIDADE": "Ética e moralidade",
    "EDUCACAO_E_EXPLICACAO_CIENTIFICA": "Educação e explicação científica",
    "ECONOMIA_E_MERCADO": "Economia e mercado",
    "PERSONALIZACAO_E_HUMANIZACAO": "Personalização e humanização",
    "RESPONSABILIDADE_E_ATRIBUICAO": "Responsabilidade e atribuição",
    "COMPETICAO_E_PRESTIGIO": "Competição e prestígio",
    "SEM_FRAME_CIENTIFICO_IDENTIFICAVEL": "Sem frame científico identificável",
    "ERRO": "Erro na classificação",
}

ROTULOS_ABRANGENCIA = {
    "NACIONAL": "Nacional",
    "INTERNACIONAL": "Internacional",
    "NAO_CONCLUSIVO": "Não conclusivo",
    "ERRO": "Erro na classificação",
}


def contar_com_normalizacao_de_caixa(nomes):
    """Conta ocorrências de nomes (instituições/pessoas) tratando variações
    de maiúsculas/minúsculas como a mesma entidade — a IA nem sempre
    capitaliza o mesmo nome de forma consistente entre chamadas diferentes
    (ex: "Nasa" vs "NASA" contados como duas instituições distintas, cada
    uma com metade da contagem real). Para exibição, usa a grafia mais
    frequente entre as variantes encontradas, não uma escolha arbitrária."""
    contagem_por_chave = Counter()
    variantes_por_chave = {}

    for nome in nomes:
        nome = str(nome).strip()
        if not nome:
            continue
        chave = nome.casefold()
        contagem_por_chave[chave] += 1
        variantes_por_chave.setdefault(chave, Counter())[nome] += 1

    resultado = Counter()
    for chave, total in contagem_por_chave.items():
        grafia_mais_comum = variantes_por_chave[chave].most_common(1)[0][0]
        resultado[grafia_mais_comum] = total

    return resultado


def agregar_classificacoes_materias(df_com_keywords, top_n=8):
    """Agrega, em Python (sem chamada de IA), o enquadramento, área,
    abrangência, instituições e pessoas das matérias já classificadas por
    extrair_keywords.py. df_com_keywords precisa ser o resultado de um
    merge entre o corpus principal (já filtrado por período/veículo) e
    materias_keywords.csv (via URL) — funciona com qualquer filtro,
    porque a classificação já existe por matéria; agregar é gratuito."""
    if df_com_keywords.empty:
        return None

    contagem_frame = Counter(f for f in df_com_keywords["frame_predominante"] if f)
    contagem_abrangencia = Counter(a for a in df_com_keywords["abrangencia"] if a)
    contagem_area = Counter(a for a in df_com_keywords["area"] if a)
    contagem_instituicoes = contar_com_normalizacao_de_caixa(
        nome for lista in df_com_keywords["instituicoes_lista"] for nome in lista if nome
    )
    contagem_pessoas = contar_com_normalizacao_de_caixa(
        nome for lista in df_com_keywords["pessoas_lista"] for nome in lista if nome
    )

    if contagem_abrangencia:
        codigo_abrangencia = contagem_abrangencia.most_common(1)[0][0]
        abrangencia_predominante = ROTULOS_ABRANGENCIA.get(codigo_abrangencia, codigo_abrangencia)
    else:
        abrangencia_predominante = "Não conclusivo"

    return {
        "enquadramentos_predominantes": [
            (ROTULOS_FRAME.get(f, f), c) for f, c in contagem_frame.most_common(top_n)
        ],
        "areas_predominantes": contagem_area.most_common(top_n),
        "abrangencia_predominante": abrangencia_predominante,
        "instituicoes_mais_visiveis": contagem_instituicoes.most_common(top_n),
        "pessoas_mais_visiveis": contagem_pessoas.most_common(top_n),
        "n_materias": len(df_com_keywords),
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

    linhas = "\n".join(
        f"- {data} ({veiculo}): {texto}" for data, veiculo, texto in tendencias_diarias
    )

    prompt = f"""
Você é pesquisador em jornalismo científico. Abaixo estão frases curtas que
resumem a tendência da cobertura de ciência de veículos brasileiros, uma por
combinação de dia e veículo, dentro de um período.

Sintetize essas frases em um único parágrafo coeso (3 a 5 frases), em português
do Brasil, descrevendo a tendência geral do período como um todo — não repita
as frases dia a dia, produza uma leitura consolidada. Se houver diferenças
notáveis de enfoque entre veículos, pode mencioná-las brevemente, mas o foco
principal é a tendência do período como um todo.

Não invente informações além do que está nas frases abaixo. Não use markdown.

Frases por dia e veículo do período:
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
    """Carrega a classificação por matéria gerada por extrair_keywords.py:
    palavras-chave, enquadramento, área, abrangência, instituições e
    pessoas. Retorna None se o arquivo ainda não existe ou está num
    formato antigo/incompleto (força reprocessamento com
    extrair_keywords.py), para não quebrar o app."""
    try:
        df_kw = pd.read_csv("materias_keywords.csv", encoding="utf-8-sig")
    except FileNotFoundError:
        return None
    except Exception:
        st.sidebar.warning("Não foi possível carregar a classificação por IA das matérias.")
        return None

    colunas_esperadas = [
        "url", "palavras_chave", "frame_predominante", "area",
        "abrangencia", "instituicoes", "pessoas"
    ]
    colunas_faltando = [c for c in colunas_esperadas if c not in df_kw.columns]
    if colunas_faltando:
        st.sidebar.warning("A classificação por IA das matérias está desatualizada e precisa ser reprocessada.")
        return None

    def _parse_lista(valor):
        try:
            lista = json.loads(valor)
            return lista if isinstance(lista, list) else []
        except (TypeError, ValueError):
            return []

    df_kw["palavras_chave_lista"] = df_kw["palavras_chave"].apply(_parse_lista)
    df_kw["instituicoes_lista"] = df_kw["instituicoes"].apply(_parse_lista)
    df_kw["pessoas_lista"] = df_kw["pessoas"].apply(_parse_lista)
    df_kw["url"] = df_kw["url"].fillna("")
    df_kw["frame_predominante"] = df_kw["frame_predominante"].fillna("")
    df_kw["area"] = df_kw["area"].fillna("")
    df_kw["abrangencia"] = df_kw["abrangencia"].fillna("")

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
df = df_original.copy()

# Calculado logo após o carregamento (não depende de nenhum filtro da
# sidebar) para poder exibir o frescor dos dados já no cabeçalho — antes,
# essa informação só aparecia navegando até uma seção específica.
data_mais_recente_dt = df_original["date_dt"].max()
data_mais_recente_str = (
    data_mais_recente_dt.strftime("%d/%m/%Y") if pd.notna(data_mais_recente_dt) else None
)

if data_mais_recente_str:
    dias_desde_atualizacao = (pd.Timestamp.now().normalize() - data_mais_recente_dt).days
    if dias_desde_atualizacao <= 1:
        rotulo_frescor = "🟢"
    elif dias_desde_atualizacao <= 4:
        rotulo_frescor = "🟡"
    else:
        rotulo_frescor = "🔴"
    st.caption(f"{rotulo_frescor} Dados atualizados até **{data_mais_recente_str}**")


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

veiculos_disponiveis = sorted(df_original["Veículo"].unique())

if len(veiculos_disponiveis) > 1:
    st.sidebar.subheader("Veículo")
    veiculos_selecionados = st.sidebar.multiselect(
        "Filtrar por veículo",
        veiculos_disponiveis,
        default=veiculos_disponiveis,
        key="veiculos_selecionados"
    )
    if veiculos_selecionados:
        df = df[df["Veículo"].isin(veiculos_selecionados)]
    else:
        st.sidebar.warning("Nenhum veículo selecionado — mostrando todos.")
        veiculos_selecionados = veiculos_disponiveis
else:
    veiculos_selecionados = veiculos_disponiveis

st.sidebar.subheader("Seção")

secao_selecionada = st.sidebar.radio(
    "Escolha a visualização",
    [
        "Análise IA",
        "Visão geral",
        "Temas",
        "Sismógrafo",
        "Rede semântica"
    ],
    index=0,
    key="secao_selecionada"
)

SECOES_COM_AGRUPAMENTO_TEMPORAL = {"Visão geral", "Temas", "Sismógrafo"}

if secao_selecionada in SECOES_COM_AGRUPAMENTO_TEMPORAL:
    st.sidebar.subheader("Visualização temporal")
    escala = st.sidebar.radio(
        "Agrupar por",
        ["Dia", "Mês", "Ano"],
        index=0
    )
else:
    # "Análise IA" e "Rede semântica" não usam agrupamento temporal — o
    # controle fica escondido em vez de aparecer sem nenhum efeito visível
    # nessas telas. O valor abaixo nunca chega a ser usado por essas
    # seções (só existe para a variável não ficar indefinida).
    escala = "Dia"

st.sidebar.caption(
    "\"Análise IA\" sempre traz um resumo do dia mais recente no topo, além da "
    "análise do período selecionado aqui em cima. As demais seções usam o "
    "período e a escala selecionados."
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

def icone_ajuda(texto_explicativo):
    """Ícone de informação com tooltip nativo do navegador (atributo
    title, sem precisar de CSS/JS extra) — usado para explicações que não
    precisam ficar sempre visíveis na tela, só disponíveis a quem quiser
    passar o cursor por cima. Use junto de um st.markdown(..., unsafe_
    allow_html=True), não sozinho."""
    return f'<span title="{html.escape(texto_explicativo)}" style="cursor:help;opacity:0.55;font-size:0.82em;">ⓘ</span>'


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


def card_ranking(titulo, itens_com_contagem, icone=""):
    """Como card(), mas para rankings de frequência — mostra uma barra de
    proporção e o número ao lado de cada item, em vez de uma nuvem de
    chips do mesmo tamanho. Sem isso, dois itens em posições muito
    diferentes do ranking pareciam visualmente equivalentes (crítica:
    'ranking sem número não é ranking'). itens_com_contagem é uma lista
    de tuplas (item, contagem), já ordenada do maior para o menor."""
    if not itens_com_contagem:
        card(titulo, [], icone)
        return

    maximo = max(contagem for _, contagem in itens_com_contagem) or 1

    linhas_html = "".join(
        f'''<div class="ranking-linha">
              <div class="ranking-rotulo">{html.escape(str(item))}</div>
              <div class="ranking-barra-fundo">
                <div class="ranking-barra-preenchida" style="width:{(contagem / maximo * 100):.0f}%;"></div>
              </div>
              <div class="ranking-contagem">{numero_br(contagem)}</div>
            </div>'''
        for item, contagem in itens_com_contagem
    )

    st.markdown(
        f"""
        <div class="card-panorama">
            <div class="card-titulo">{icone} {html.escape(titulo)}</div>
            <div class="ranking-lista">{linhas_html}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


LIMIAR_AMOSTRA_PEQUENA = 10


def avisar_se_amostra_pequena(n_materias, contexto="esta classificação"):
    """Um ranking sobre poucas matérias é ruído, não sinal — mas a
    interface não distinguia visualmente 'predominante entre 3 matérias'
    de 'predominante entre 300'. Mostra um aviso discreto quando a
    amostra é pequena, para o usuário calibrar a confiança do que está
    vendo (mesmo espírito do teste de significância já usado nos
    'Termos em ascensão', só que aqui é um limiar simples de tamanho)."""
    if n_materias < LIMIAR_AMOSTRA_PEQUENA:
        st.caption(
            f"⚠ Baseado em poucas matérias (n={n_materias}) para {contexto} — "
            "leia os rankings abaixo com cautela."
        )


def mostrar_periodo_no_topo(periodo_label, escala_label=None):
    """Exibe um indicador fixo no topo da seção com o período selecionado
    na sidebar — para o usuário nunca perder de vista a que período a
    análise se refere, independente de qual seção está vendo. escala_label
    é omitida quando a seção atual não usa agrupamento temporal (Dia/Mês/
    Ano), para não mostrar um valor que não influencia nada nessa tela."""
    sufixo_escala = (
        f" · agrupado por {html.escape(escala_label).lower()}" if escala_label else ""
    )
    st.markdown(
        f'<div class="painel-eyebrow">📅 Período selecionado: '
        f'{html.escape(periodo_label)}{sufixo_escala}</div>',
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

    /* Rankings com peso visível (card_ranking) — barra de proporção +
    número ao lado de cada item, para não esconder a magnitude relativa
    entre o 1º e o último colocado do ranking. */
    .ranking-lista {
        display: flex;
        flex-direction: column;
        gap: 7px;
    }
    .ranking-linha {
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .ranking-rotulo {
        flex: 0 0 auto;
        max-width: 46%;
        font-size: 0.84rem;
        color: var(--obs-paper);
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    .ranking-barra-fundo {
        flex: 1 1 auto;
        height: 12px;
        background: rgba(95,191,160,0.08);
        border-radius: 6px;
        overflow: hidden;
    }
    .ranking-barra-preenchida {
        height: 100%;
        background: var(--obs-teal);
        border-radius: 6px;
        min-width: 3px;
    }
    .ranking-contagem {
        flex: 0 0 auto;
        min-width: 28px;
        text-align: right;
        font-family: 'SF Mono', 'Cascadia Code', Consolas, Menlo, monospace;
        font-size: 0.78rem;
        color: var(--obs-muted);
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

analises_diarias = carregar_analise_diaria()
df_keywords = carregar_keywords()

# Matérias de hoje já respeitando o filtro de veículo da sidebar — usada
# no sorteio de matérias na parte "Hoje" da seção Análise IA.
df_hoje_filtrado = (
    df_original[
        (df_original["date_dt"] == data_mais_recente_dt)
        & (df_original["Veículo"].isin(veiculos_selecionados))
    ]
    if pd.notna(data_mais_recente_dt) else df_original.iloc[0:0]
)

resultado_tendencia_hoje = agregar_analises_periodo(
    analises_diarias,
    {data_mais_recente_str} if data_mais_recente_str else set(),
    veiculos_no_filtro=veiculos_selecionados
)


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

NOMES_DIA_SEMANA = ["segunda-feira", "terça-feira", "quarta-feira", "quinta-feira",
                    "sexta-feira", "sábado", "domingo"]


@st.cache_data
def calcular_ritmo_por_dia_semana(chave_cache):
    """Ritmo histórico médio de matérias/dia, separado por dia da semana.
    Mídia costuma publicar menos ciência aos fins de semana — comparar um
    domingo contra a média geral (dominada por dias de semana, tipicamente
    com volume maior) faria qualquer domingo normal parecer sistematicamente
    'abaixo do ritmo', mesmo sem nada de anormal acontecendo. Comparar cada
    dia contra a média histórica DAQUELE MESMO dia da semana evita esse viés."""
    serie_historica = agrupar_por_escala(df_original, "Dia")
    dias_semana = serie_historica["date_dt"].dt.dayofweek
    medias = serie_historica.groupby(dias_semana)["Matérias"].mean()
    return medias.to_dict()


chave_cache_ritmo_semana = (len(df_original), df_original["date_dt"].min(), df_original["date_dt"].max())
ritmo_por_dia_semana = calcular_ritmo_por_dia_semana(chave_cache_ritmo_semana)

# Ritmo do dia mais recente (não depende do filtro da sidebar): compara o
# volume de matérias do dia com a média histórica do MESMO DIA DA SEMANA
# (não a média geral do corpus) — ver calcular_ritmo_por_dia_semana acima.
n_materias_hoje = (
    int((df_original["date_dt"] == data_mais_recente_dt).sum())
    if pd.notna(data_mais_recente_dt) else 0
)

if pd.notna(data_mais_recente_dt):
    dia_semana_hoje = data_mais_recente_dt.dayofweek
    nome_dia_semana_hoje = NOMES_DIA_SEMANA[dia_semana_hoje]
    ritmo_historico_dia_semana_hoje = ritmo_por_dia_semana.get(dia_semana_hoje, ritmo_historico)
else:
    nome_dia_semana_hoje = ""
    ritmo_historico_dia_semana_hoje = ritmo_historico

razao_dia = (
    n_materias_hoje / ritmo_historico_dia_semana_hoje
    if ritmo_historico_dia_semana_hoje > 0 else 1.0
)


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
      __ROTULO_HISTORICO__ ▸ __RITMO_HISTORICO__ matéria(s)/dia<br/>
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


def renderizar_gauge_ritmo(eyebrow, rotulo_valor, ritmo_valor, ritmo_hist, razao,
                            altura=200, rotulo_historico="Histórico do corpus"):
    """Renderiza o gauge de ritmo de publicação. Reutilizável para qualquer
    par (valor atual, valor histórico) — usado tanto para o dia mais recente
    quanto para o período filtrado pelo usuário. rotulo_historico deixa
    explícito A QUE o valor histórico se refere — importante desde que o
    gauge de "hoje" passou a comparar contra a média do MESMO DIA DA SEMANA
    (não mais a média geral do corpus), para não ficar ambíguo qual
    histórico está sendo mostrado."""
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
        .replace("__ROTULO_HISTORICO__", html.escape(rotulo_historico))
        .replace("__COR_STATUS__", cor_status)
        .replace("__STATUS__", html.escape(status))
        .replace("__RITMO_ATUAL__", f"{ritmo_valor:.1f}".replace(".", ","))
        .replace("__RITMO_HISTORICO__", f"{ritmo_hist:.1f}".replace(".", ","))
        .replace("__DIFERENCA__", f"{diferenca_pct:+.0f}".replace(".", ","))
        .replace("__ANGULO__", f"{angulo_final:.1f}")
    )

    components.html(gauge_html, height=altura, scrolling=False)


# =========================
# "Panorama do dia" foi fundido dentro da seção "Análise IA" (mais abaixo)
# — deixou de ser uma opção separada do menu lateral, para não duplicar os
# cards de classificação que antes apareciam idênticos nas duas seções.
# =========================

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

    if len(veiculos_disponiveis) > 1:
        st.markdown(
            f"### Comparação por veículo "
            f"{icone_ajuda('Contagem bruta mistura dois efeitos: foco editorial real em ciência e profundidade do histórico coletado por cada fonte (algumas têm mais dias de backfill que outras). A coluna Matérias/dia e o modo normalizado do gráfico dividem pelo número de dias efetivamente cobertos por cada veículo, para uma comparação mais justa.')}",
            unsafe_allow_html=True
        )

        por_veiculo = (
            df.groupby("Veículo")
            .agg(Matérias=("URL", "count"), Média_palavras=("Palavras", "mean"))
            .reset_index()
        )
        por_veiculo["Média_palavras"] = por_veiculo["Média_palavras"].round(0).fillna(0).astype(int)

        # Dias efetivamente cobertos por CADA veículo dentro do período
        # filtrado (não o período filtrado inteiro) — evita que um veículo
        # com backfill mais raso pareça artificialmente menor só por ter
        # menos dias de histórico no acervo.
        cobertura_por_veiculo = df.groupby("Veículo")["date_dt"].agg(["min", "max"])
        cobertura_por_veiculo["Dias_cobertos"] = (
            (cobertura_por_veiculo["max"] - cobertura_por_veiculo["min"]).dt.days + 1
        )
        por_veiculo = por_veiculo.merge(
            cobertura_por_veiculo[["Dias_cobertos"]], on="Veículo", how="left"
        )
        por_veiculo["Matérias/dia"] = (
            por_veiculo["Matérias"] / por_veiculo["Dias_cobertos"]
        ).round(2)

        por_veiculo.columns = [
            "Veículo", "Matérias", "Média de palavras", "Dias cobertos", "Matérias/dia"
        ]
        por_veiculo = por_veiculo.sort_values("Matérias", ascending=False)

        col_tabela_veiculo, col_grafico_veiculo = st.columns([1, 2])

        with col_tabela_veiculo:
            st.dataframe(
                por_veiculo[["Veículo", "Matérias", "Matérias/dia", "Média de palavras"]],
                use_container_width=True, hide_index=True
            )

        with col_grafico_veiculo:
            modo_comparacao = st.radio(
                "Comparar por",
                ["Contagem bruta", "Normalizado (matérias/dia)"],
                horizontal=True,
                key="modo_comparacao_veiculo"
            )
            coluna_valor_pizza = (
                "Matérias" if modo_comparacao == "Contagem bruta" else "Matérias/dia"
            )
            titulo_pizza = (
                "Participação por veículo (nº bruto de matérias)"
                if modo_comparacao == "Contagem bruta"
                else "Participação por veículo (matérias/dia coberto — normalizado)"
            )
            fig_veiculo_pizza = px.pie(
                por_veiculo, names="Veículo", values=coluna_valor_pizza,
                title=titulo_pizza, hole=0.4
            )
            st.plotly_chart(fig_veiculo_pizza, use_container_width=True)

        mapa_col_escala_veiculo = {
            "Dia": ("Data", "date_dt"),
            "Mês": ("Mês", "mes_dt"),
            "Ano": ("Ano", "ano_dt")
        }
        col_rotulo_v, col_ordem_v = mapa_col_escala_veiculo[escala]

        serie_veiculo = (
            df.groupby([col_rotulo_v, col_ordem_v, "Veículo"])
            .size()
            .reset_index(name="Matérias")
            .sort_values(col_ordem_v)
        )
        serie_veiculo["Período"] = serie_veiculo[col_rotulo_v]

        fig_veiculo_evolucao = px.line(
            serie_veiculo,
            x="Período", y="Matérias", color="Veículo",
            markers=True,
            title=f"Evolução por {escala.lower()}, por veículo"
        )
        fig_veiculo_evolucao.update_layout(xaxis_title="Período", xaxis_type="category")
        st.plotly_chart(fig_veiculo_evolucao, use_container_width=True)

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

    col_titulo_amostra, col_slider_amostra, col_botao_amostra = st.columns([3, 2, 1])
    with col_titulo_amostra:
        st.subheader("Amostra de matérias do período")
    with col_slider_amostra:
        tamanho_amostra = st.slider(
            "Quantidade de matérias",
            min_value=5, max_value=30, value=15,
            key="tamanho_amostra_visao_geral"
        )
    with col_botao_amostra:
        st.markdown("<br>", unsafe_allow_html=True)
        sortear_amostra_novamente = st.button("🔀 Sortear outras", key="btn_sortear_amostra_visao_geral")

    st.caption(
        "Amostra aleatória do período/busca/veículo filtrados — não é um ranking "
        "por nenhuma métrica. Clique em qualquer coluna da tabela para reordenar "
        "como quiser."
    )

    chave_amostra_atual = (
        len(df), df["date_dt"].min(), df["date_dt"].max(),
        tuple(sorted(veiculos_selecionados)), busca, tamanho_amostra
    )

    precisa_sortear_amostra = (
        sortear_amostra_novamente
        or "amostra_visao_geral_chave" not in st.session_state
        or st.session_state.get("amostra_visao_geral_chave") != chave_amostra_atual
    )

    if precisa_sortear_amostra:
        n_amostra_visao_geral = min(tamanho_amostra, len(df))
        st.session_state["amostra_visao_geral"] = (
            df.sample(n=n_amostra_visao_geral) if n_amostra_visao_geral > 0 else df.iloc[0:0]
        )
        st.session_state["amostra_visao_geral_chave"] = chave_amostra_atual

    tabela_amostra = st.session_state.get("amostra_visao_geral")

    if tabela_amostra is None or tabela_amostra.empty:
        st.caption("Nenhuma matéria disponível para amostragem com os filtros atuais.")
    else:
        tabela = tabela_amostra[["Data", "Título", "Editoria", "Autor", "Palavras", "URL"]].copy()

        st.dataframe(
            tabela,
            use_container_width=True,
            hide_index=True
        )


if secao_selecionada == "Temas":
    mostrar_periodo_no_topo(periodo_label_sidebar, escala)

    st.markdown('<div class="painel-eyebrow">Temas</div>', unsafe_allow_html=True)
    with st.expander("📖 Como ler esta seção (4 formas diferentes de medir o mesmo assunto)"):
        st.markdown(
            "Esta seção reúne 4 formas diferentes de responder \"do que a mídia "
            "está falando\", cada uma com um método distinto — elas vão discordar "
            "entre si às vezes, porque medem coisas diferentes.\n\n"
            "- **Palavras-chave (IA)** captam *conceito* — a IA lê cada matéria e "
            "extrai termos relevantes, descartando ruído.\n"
            "- **Termos em ascensão** são as mesmas palavras-chave, mas testadas "
            "estatisticamente para detectar o que está *acelerando*, não só o que "
            "é mais frequente.\n"
            "- **N-gramas** (palavras/bigramas/trigramas) captam *frequência "
            "literal* de texto — mais bruto, sem julgamento de relevância, mas "
            "não depende de IA.\n"
            "- **Animações por mês** mostram qualquer uma das anteriores "
            "evoluindo no tempo."
        )

    if df_keywords is not None:
        st.markdown(
            f"### 🧠 Palavras-chave identificadas por IA "
            f"{icone_ajuda('Extraídas via modelo de linguagem a partir do conteúdo de cada matéria — tendem a ser mais específicas que a contagem bruta de palavras e n-gramas abaixo, já que descartam termos genéricos e mantêm conceitos, instituições, tecnologias e atores.')}",
            unsafe_allow_html=True
        )

        urls_filtradas = set(df["URL"]) - {""}
        kw_filtrado = df_keywords[df_keywords["url"].isin(urls_filtradas)]

        if kw_filtrado.empty:
            st.info(
                "Nenhuma matéria do período/busca filtrados possui palavras-chave "
                "extraídas ainda. Amplie os filtros na barra lateral, ou aguarde o "
                "próximo processamento."
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

        st.markdown(
            f"### 📈 Termos em ascensão "
            f"{icone_ajuda('Diferente do ranking acima (que mostra o que já está em alta), compara a taxa de menções de cada palavra-chave na janela mais recente do período com a taxa no restante — sinaliza temas acelerando mesmo sem volume alto. Usa um teste estatístico de significância (não percentual bruto), para não confundir ruído de amostra pequena (1 menção virar 2 = +100%, sem significar nada) com aumento genuíno.')}",
            unsafe_allow_html=True
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
            "As palavras-chave por IA ainda não estão disponíveis para as matérias "
            "deste dataset — essa análise é gerada em etapas de processamento "
            "separadas do app."
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
        st.markdown(
            f"### 🧠 Top palavras-chave animado por mês (IA) "
            f"{icone_ajuda('Mesma ideia da animação acima, mas usando as palavras-chave extraídas por IA em vez de n-gramas — costuma mostrar uma evolução mais limpa dos temas, sem ruído de palavras genéricas.')}",
            unsafe_allow_html=True
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

    st.markdown(
        f"### 📡 Sismógrafo de cobertura "
        f"{icone_ajuda('Detecta automaticamente os picos de publicação no período filtrado e transforma cada um em um boletim — a matéria mais representativa daquele momento.')}",
        unsafe_allow_html=True
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
    st.markdown('<div class="painel-eyebrow">Análise IA</div>', unsafe_allow_html=True)
    st.subheader("🧠 Análise estruturada por IA")

    # =========================
    # Parte 1: pulso de hoje — compacto, sempre o dia mais recente do
    # dataset, independente do período selecionado na barra lateral.
    # =========================
    st.markdown(f"#### 📌 Hoje — {data_mais_recente_str}")
    st.caption(
        f"{numero_br(n_materias_hoje)} matéria(s) publicadas em {data_mais_recente_str} "
        f"({nome_dia_semana_hoje}, todos os veículos). Este resumo sempre mostra o dia "
        "mais recente do dataset."
    )

    renderizar_gauge_ritmo(
        "Ritmo de publicação (hoje)",
        f"Hoje ({nome_dia_semana_hoje})",
        n_materias_hoje, ritmo_historico_dia_semana_hoje, razao_dia,
        rotulo_historico=f"Histórico de {nome_dia_semana_hoje}s"
    )
    st.markdown(
        f'<span style="font-size:0.85rem;color:#8B93A7;">📊 Comparado à média histórica '
        f'de <b>{nome_dia_semana_hoje}s</b> especificamente '
        f'({ritmo_historico_dia_semana_hoje:.1f} matéria(s)/dia)</span> '
        f'{icone_ajuda("Não a média geral do corpus — mídia costuma publicar menos ciência aos fins de semana, então comparar contra a média geral penalizaria domingos e sábados injustamente.")}',
        unsafe_allow_html=True
    )

    if resultado_tendencia_hoje:
        tendencias_hoje = resultado_tendencia_hoje["tendencias_diarias"]

        if len(tendencias_hoje) == 1:
            _, _, texto_tendencia = tendencias_hoje[0]
            st.markdown(
                f"""
                <div class="insight-box">
                    <div class="insight-title">📈 Tendência do dia</div>
                    {html.escape(texto_tendencia)}
                </div>
                """,
                unsafe_allow_html=True
            )
        elif len(tendencias_hoje) > 1:
            linhas_tendencia_html = "".join(
                f'<p style="margin:0 0 10px 0;"><b>{html.escape(veiculo)}:</b> {html.escape(texto)}</p>'
                for _, veiculo, texto in tendencias_hoje
            )
            st.markdown(
                f"""
                <div class="insight-box">
                    <div class="insight-title">📈 Tendência do dia (por veículo)</div>
                    {linhas_tendencia_html}
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.caption(
            "📈 Tendência do dia ainda não disponível para este dia/veículo."
        )

    col_titulo_sorteio, col_botao_sorteio = st.columns([4, 1])
    with col_botao_sorteio:
        sortear_novamente = st.button("🔀 Sortear outras", key="btn_sortear_materias")

    df_dia_completo = df_hoje_filtrado

    # Chave de cache inclui a seleção de veículos, não só a data — sem
    # isso, trocar o filtro de veículo não invalidava o sorteio, e podia
    # continuar mostrando matérias de fora do filtro atual (ou travado em
    # menos matérias do que realmente havia disponível para o novo filtro).
    chave_sorteio_atual = (data_mais_recente_str, tuple(sorted(veiculos_selecionados)))

    precisa_sortear = (
        sortear_novamente
        or "materias_aleatorias_chave" not in st.session_state
        or st.session_state.get("materias_aleatorias_chave") != chave_sorteio_atual
    )

    if precisa_sortear:
        n_amostra = min(3, len(df_dia_completo))
        st.session_state["materias_aleatorias"] = (
            df_dia_completo.sample(n=n_amostra) if n_amostra > 0 else df_dia_completo
        )
        st.session_state["materias_aleatorias_chave"] = chave_sorteio_atual

    materias_amostra = st.session_state.get("materias_aleatorias")
    n_mostradas = 0 if materias_amostra is None else len(materias_amostra)

    with col_titulo_sorteio:
        if n_mostradas == 3:
            st.markdown("##### 🎲 Três matérias do dia, para começar por algum lugar")
        elif n_mostradas > 0:
            st.markdown(
                f"##### 🎲 {n_mostradas} matéria(s) do dia, para começar por algum lugar"
            )
        else:
            st.markdown("##### 🎲 Matérias do dia, para começar por algum lugar")

    if materias_amostra is None or materias_amostra.empty:
        st.caption(
            "Nenhuma matéria disponível para sorteio neste dia, com o filtro de "
            "veículo atual."
        )
    else:
        cols_materias = st.columns(len(materias_amostra))
        for col, (_, materia) in zip(cols_materias, materias_amostra.iterrows()):
            with col:
                titulo_materia = html.escape(str(materia["Título"]) or "(sem título)")
                editoria_materia = html.escape(str(materia["Editoria"]) or "—")
                veiculo_materia = html.escape(str(materia.get("Veículo", "")) or "—")
                autor_materia = html.escape(materia["Autor"] or "Redação")
                url_materia = str(materia["URL"])
                imagem_materia = str(materia.get("Imagem", "")).strip()

                if url_materia.startswith("http"):
                    titulo_html = f'<a href="{url_materia}" target="_blank" style="color:#E8E4D8;text-decoration:none;">{titulo_materia}</a>'
                else:
                    titulo_html = titulo_materia

                if imagem_materia.startswith("http"):
                    # Miniatura à esquerda, largura fixa e altura contida —
                    # o texto (título, editoria, autor) fica ao lado, à
                    # direita, em vez de embaixo.
                    imagem_html = (
                        f'<img src="{html.escape(imagem_materia)}" alt="" '
                        f'style="width:96px;height:96px;object-fit:cover;'
                        f'border-radius:6px;flex-shrink:0;display:block;" '
                        f'onerror="this.style.display=\'none\';">'
                    )
                else:
                    imagem_html = ""

                st.markdown(
                    f"""
                    <div class="card-panorama" style="min-height:130px;display:flex;gap:12px;align-items:flex-start;">
                        {imagem_html}
                        <div style="flex:1;min-width:0;">
                            <div style="font-family:'SF Mono',Consolas,monospace;font-size:0.7rem;
                                        color:#8B93A7;text-transform:uppercase;letter-spacing:0.05em;
                                        margin-bottom:8px;">{editoria_materia} · {veiculo_materia}</div>
                            <div style="font-weight:600;margin-bottom:10px;line-height:1.4;">{titulo_html}</div>
                            <div style="font-size:0.8rem;color:#8B93A7;font-style:italic;">{autor_materia}</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    st.divider()

    # =========================
    # Parte 2: análise do período selecionado na barra lateral.
    # =========================
    mostrar_periodo_no_topo(periodo_label_sidebar)
    st.markdown(
        f"#### 🧬 Análise do período selecionado "
        f"{icone_ajuda('Enquadramento, área, abrangência, instituições e pessoas são agregados a partir da classificação já feita por matéria — filtrar aqui não tem custo adicional de IA. Só a tendência do período e a síntese abaixo usam uma chamada de IA nova (sob demanda).')}",
        unsafe_allow_html=True
    )

    if df_keywords is None:
        st.warning(
            "A classificação por IA das matérias ainda não está disponível "
            "para habilitar esta seção."
        )
    else:
        urls_filtradas_ia = set(df["URL"]) - {""}
        df_periodo_com_keywords = df[df["URL"].isin(urls_filtradas_ia)].merge(
            df_keywords[[
                "url", "frame_predominante", "area", "abrangencia",
                "instituicoes_lista", "pessoas_lista"
            ]],
            left_on="URL", right_on="url", how="inner"
        )
        resultado_classificacao = agregar_classificacoes_materias(df_periodo_com_keywords)

        if not resultado_classificacao:
            st.warning(
                "Nenhuma matéria do período/busca/veículo filtrados possui "
                "classificação de IA ainda."
            )
        else:
            rotulo_veiculo = (
                "todos os veículos" if len(veiculos_selecionados) == len(veiculos_disponiveis)
                else f"{len(veiculos_selecionados)} veículo(s) selecionado(s)"
            )
            st.caption(
                f"📊 {numero_br(resultado_classificacao['n_materias'])} matéria(s) "
                f"classificada(s) no período ({rotulo_veiculo})."
            )
            avisar_se_amostra_pequena(resultado_classificacao["n_materias"], "o período selecionado")

            col1, col2 = st.columns(2)

            with col1:
                card_ranking(
                    "Enquadramentos predominantes",
                    resultado_classificacao["enquadramentos_predominantes"],
                    "📰"
                )
                card_ranking(
                    "Áreas predominantes",
                    resultado_classificacao["areas_predominantes"],
                    "🔬"
                )

            with col2:
                card_ranking(
                    "Instituições mais visíveis",
                    resultado_classificacao["instituicoes_mais_visiveis"],
                    "🏛️"
                )
                card_ranking(
                    "Pessoas mais visíveis",
                    resultado_classificacao["pessoas_mais_visiveis"],
                    "👤"
                )

            card(
                "Abrangência predominante",
                [resultado_classificacao["abrangencia_predominante"]],
                "🌎"
            )

    st.divider()

    st.markdown(
        f"### Tendência do período "
        f"{icone_ajuda('Texto corrido, gerado por dia×veículo — não dá para agregar por contagem como os outros campos. A síntese abaixo usa 1 chamada de IA sob demanda, aproveitando só as frases diárias já existentes.')}",
        unsafe_allow_html=True
    )

    datas_no_filtro = set(df["Data"].unique())
    resultado_tendencia_periodo = agregar_analises_periodo(
        analises_diarias, datas_no_filtro, veiculos_no_filtro=veiculos_selecionados
    )

    if not resultado_tendencia_periodo:
        st.info(
            "Nenhuma tendência disponível ainda para os dias/veículos deste período."
        )
    else:
        chave_periodo = tuple(resultado_tendencia_periodo["tendencias_diarias"])

        if st.button("🧬 Gerar síntese do período", key="btn_sintese_periodo"):
            with st.spinner("Sintetizando tendência do período..."):
                try:
                    texto_sintese = sintetizar_tendencia_periodo(
                        resultado_tendencia_periodo["tendencias_diarias"]
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
                for data_item, veiculo_item, texto_item in resultado_tendencia_periodo["tendencias_diarias"]:
                    st.write(f"**{data_item}** ({veiculo_item}) — {texto_item}")


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
    mostrar_periodo_no_topo(periodo_label_sidebar)

    st.markdown(
        f"### 🕸️ Rede semântica de palavras-chave "
        f"{icone_ajuda('Grafo de coocorrência das palavras-chave extraídas por IA nas matérias do período filtrado — dois termos se conectam quando aparecem juntos na mesma matéria. Cores indicam comunidades detectadas automaticamente (algoritmo de Louvain). Atualiza dinamicamente com qualquer filtro.')}",
        unsafe_allow_html=True
    )

    if not REDE_SEMANTICA_DISPONIVEL:
        st.warning(
            "A rede semântica não está disponível neste ambiente no momento."
        )
    elif df_keywords is None:
        st.info(
            "A classificação por IA das matérias ainda não está disponível "
            "para habilitar a rede semântica."
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
                "extraídas. Amplie o filtro na barra lateral."
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