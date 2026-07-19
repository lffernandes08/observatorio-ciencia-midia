"""
Extração por matéria: palavras-chave, enquadramento (frame), área do
conhecimento, abrangência geográfica, instituições e pessoas centrais.

Antes, várias dessas classificações (enquadramento, área, abrangência,
"atores") eram feitas em analise_diaria.py, agregando várias matérias de
um dia inteiro numa única chamada de IA — o que forçava a IA a resumir
several textos diferentes de uma vez, sem ancoragem num texto específico.

Agora tudo isso é decidido matéria por matéria (mesma chamada que já
extraía palavras-chave, só que expandida) — mais preciso, e permite que
o app agregue esses dados livremente por qualquer período/veículo depois,
sem custo adicional de API (a agregação vira só contagem em Python).

Uso:
    python extrair_keywords.py
"""

import os
import json
import time
import pandas as pd
from openai import OpenAI

ARQUIVOS_FONTES = ["folha.csv", "cnn_brasil.csv", "bbc_brasil.csv", "g1_globo.csv"]
ARQUIVO_SAIDA = "materias_keywords.csv"
MODELO = "gpt-4o-mini"
INTERVALO_ENTRE_CHAMADAS = 1
LIMITE_TEXTO_MATERIA = 3000

COLUNAS_SAIDA = [
    "date", "ano", "section", "title", "url",
    "palavras_chave", "frame_predominante", "area", "abrangencia",
    "instituicoes", "pessoas"
]

# Mesma taxonomia de área que já existia em analise_diaria.py — não mudou.
# As 8 grandes áreas oficiais do CNPq — sem categoria coringa de propósito
# (nem "Interdisciplinar" nem "Outro"): uma opção fácil tende a virar a
# saída padrão da IA quando está em dúvida, enviesando a distribuição real
# na direção do coringa. A IA é instruída a sempre escolher a área mais
# central ao fato coberto, mesmo em casos limítrofes.
AREAS = [
    "Ciências Agrárias",
    "Ciências Biológicas",
    "Ciências da Saúde",
    "Ciências Exatas e da Terra",
    "Engenharias",
    "Ciências Humanas",
    "Ciências Sociais Aplicadas",
    "Linguística, Letras e Artes",
]

# Categoria à parte (não uma 9ª opção "fácil" misturada na lista acima) —
# só para quando genuinamente não dá para identificar nenhuma das 8 áreas.
# Diferente de "Multidisciplinar"/"Interdisciplinar": não deve ser usada só
# porque a matéria toca mais de uma área (nesse caso, ainda é pra escolher
# a mais central), reduzindo o risco de virar uma saída de conveniência.
AREA_NAO_IDENTIFICADA = "Não identificável"
AREAS_VALIDAS = AREAS + [AREA_NAO_IDENTIFICADA]

FRAMES = [
    "DESCOBERTA_CIENTIFICA", "INOVACAO_TECNOLOGICA", "PROMESSA_E_BENEFICIOS",
    "RISCO_E_AMEACA", "INCERTEZA_CIENTIFICA", "CONFLITO_E_CONTROVERSIA",
    "IMPACTO_SOCIAL", "POLITICA_CIENTIFICA_E_GOVERNANCA", "ETICA_E_MORALIDADE",
    "EDUCACAO_E_EXPLICACAO_CIENTIFICA", "ECONOMIA_E_MERCADO",
    "PERSONALIZACAO_E_HUMANIZACAO", "RESPONSABILIDADE_E_ATRIBUICAO",
    "COMPETICAO_E_PRESTIGIO", "SEM_FRAME_CIENTIFICO_IDENTIFICAVEL"
]

ABRANGENCIAS = ["NACIONAL", "INTERNACIONAL", "NAO_CONCLUSIVO"]

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def carregar_corpus():
    """Carrega e combina todas as fontes de matérias disponíveis. Fontes
    ausentes são ignoradas silenciosamente — não é erro não ter todas."""
    dataframes = []

    for caminho in ARQUIVOS_FONTES:
        if not os.path.exists(caminho):
            continue
        try:
            df_fonte = pd.read_csv(caminho, encoding="utf-8-sig")
            dataframes.append(df_fonte)
        except Exception as e:
            print(f"Aviso: erro ao ler {caminho}, ignorando esta fonte: {e}")

    if not dataframes:
        raise FileNotFoundError(
            f"Nenhuma fonte de dados encontrada (procurado: {', '.join(ARQUIVOS_FONTES)})."
        )

    return pd.concat(dataframes, ignore_index=True)


def extrair_json(texto):
    texto = texto.strip().replace("```json", "").replace("```", "").strip()
    return json.loads(texto)


def montar_prompt(row):
    titulo = row.get("title", "")
    texto = str(row.get("text", ""))[:LIMITE_TEXTO_MATERIA]

    return f"""
Você é pesquisador em jornalismo científico e análise de conteúdo. Analise a
matéria abaixo e produza uma classificação estruturada em 5 dimensões.

## 1. Palavras-chave

Extraia de 5 a 8 palavras-chave semânticas. Use termos substantivos e
relevantes — conceitos, temas, instituições, doenças, tecnologias, áreas
científicas, atores. Evite palavras genéricas como "estudo", "pesquisa",
"cientistas", "Brasil". Padronize em português do Brasil.

## 2. Enquadramento (frame)

Classifique o enquadramento predominante — a perspectiva que organiza a
narrativa e indica o principal significado da notícia.

1. DESCOBERTA_CIENTIFICA — novo conhecimento, resultado de pesquisa,
   descoberta de fenômeno/espécie/mecanismo. Não use se a pesquisa for só
   apoio secundário a risco, política ou controvérsia.
2. INOVACAO_TECNOLOGICA — desenvolvimento/aplicação de tecnologia, técnica,
   produto ou método. Diferente de DESCOBERTA_CIENTIFICA: aqui o central é
   a solução criada, não o conhecimento produzido.
3. PROMESSA_E_BENEFICIOS — ciência como esperança/progresso/benefício
   futuro ("poderá", "promete"). Use quando o benefício esperado for mais
   central que a descoberta em si.
4. RISCO_E_AMEACA — perigos, danos, vulnerabilidades, consequências
   negativas à saúde/ambiente/sociedade; alertas e necessidade de
   prevenção.
5. INCERTEZA_CIENTIFICA — limites do conhecimento, ausência de consenso,
   resultados preliminares, necessidade de mais pesquisa. Vem dos limites
   da evidência, não de uma disputa entre atores (isso seria
   CONFLITO_E_CONTROVERSIA).
6. CONFLITO_E_CONTROVERSIA — disputa entre cientistas/instituições/
   governos/grupos como eixo central. Não use só por haver opiniões
   diferentes mencionadas de passagem.
7. IMPACTO_SOCIAL — consequências para vida cotidiana, grupos sociais,
   desigualdades, qualidade de vida.
8. POLITICA_CIENTIFICA_E_GOVERNANCA — financiamento, regulação, orçamento,
   decisões institucionais/governamentais, gestão de universidades/
   agências.
9. ETICA_E_MORALIDADE — dilemas éticos, valores, direitos (edição
   genética, IA, experimentação, privacidade). Precisa de avaliação
   normativa (certo/errado), não só menção de risco.
10. EDUCACAO_E_EXPLICACAO_CIENTIFICA — objetivo principal é explicar um
    conceito/fenômeno ("o que é", "como funciona"), contextualização
    didática, combate a desinformação.
11. ECONOMIA_E_MERCADO — efeitos econômicos, patentes, investimentos,
    startups, competitividade. Mencionar valores financeiros não basta; a
    dimensão econômica precisa organizar a narrativa.
12. PERSONALIZACAO_E_HUMANIZACAO — ciência via trajetória/experiência de
    indivíduos (perfil de cientista, história de paciente). Só quando a
    experiência pessoal estrutura a matéria, não quando a pessoa é só
    fonte citada.
13. RESPONSABILIDADE_E_ATRIBUICAO — quem é responsável por causar/
    enfrentar/solucionar um problema (cobrança, negligência, dever de
    agir). Diferente de POLITICA_CIENTIFICA_E_GOVERNANCA: aqui o foco é
    atribuição de culpa/dever, lá é política e estrutura institucional.
14. COMPETICAO_E_PRESTIGIO — corrida/liderança/prestígio entre
    pesquisadores/instituições/países (rankings, prêmios, pioneirismo). Se
    a competição for só comercial, use ECONOMIA_E_MERCADO.
15. SEM_FRAME_CIENTIFICO_IDENTIFICAVEL — ciência aparece só
    incidentalmente, ou o texto não trata efetivamente de ciência/
    pesquisa/tecnologia/saúde baseada em evidências.

Um único frame predominante (obrigatório). Não classifique um frame só
porque uma palavra aparece — ele precisa organizar a narrativa. Priorize
título, lead e o que recebe mais espaço no texto. Diferencie tema de
enquadramento (ex: "mudanças climáticas" é tema; "risco e ameaça" pode ser
o frame). Não invente informações fora do texto.

## 3. Área do conhecimento

Escolha exatamente 1 item desta taxonomia — as 8 grandes áreas oficiais do
CNPq. Não existe opção "multidisciplinar": mesmo quando a matéria tocar
mais de uma área, escolha a que for mais central ao fato principal coberto
(o que está sendo descoberto/estudado/relatado), não uma área só porque
foi mencionada de passagem.
{chr(10).join("- " + item for item in AREAS)}

Use "{AREA_NAO_IDENTIFICADA}" APENAS quando o texto genuinamente não
permitir identificar nenhuma das 8 áreas acima (ex: texto curto/vago
demais, ou que não trata de nenhum assunto científico reconhecível). NÃO
use essa opção só porque a matéria toca mais de uma área — nesse caso,
ainda escolha a área mais central entre as 8.

## 4. Abrangência geográfica

Classifique a abrangência geográfica do FATO coberto — não o alcance do
veículo de imprensa, que é sempre nacional.

- NACIONAL — o fato central (descoberta, pesquisa, instituição, evento,
  política) está situado no Brasil, em qualquer escala (nacional,
  estadual ou municipal).
- INTERNACIONAL — o fato central está situado fora do Brasil, ou envolve
  instituição/pesquisa/evento estrangeiro (ex: Nasa, universidade
  estrangeira, OMS).
- NAO_CONCLUSIVO — não é possível determinar com base no texto.

Exemplo: missão da Nasa ou pesquisa em universidade estrangeira →
INTERNACIONAL, mesmo publicada por veículo brasileiro. Pesquisa da UFLA ou
política do Ministério da Saúde → NACIONAL.

## 5. Instituições e pessoas

Instituições: liste APENAS nomes próprios específicos de instituições,
empresas, agências, órgãos governamentais ou universidades citados como
parte central do fato coberto (ex: "USP", "Fiocruz", "Nasa", "OMS").
NÃO use categorias genéricas como "universidades brasileiras" ou
"governo" — se a matéria só mencionar a categoria genérica sem nomear a
instituição, não inclua.

Pessoas: liste APENAS nomes próprios específicos de indivíduos que sejam
protagonistas ou fontes centrais da matéria (pesquisador principal,
autoridade citada, paciente/beneficiário cuja história é central). NÃO
inclua pessoas mencionadas apenas de passagem. Use o nome como aparece no
texto (não abrevie nem tente completar nomes parciais).

## Texto para análise

Título: {titulo}

Texto:
{texto}

## Formato da resposta

Responda apenas com um objeto JSON válido, sem comentários, sem markdown,
exatamente nesta estrutura:

{{
  "palavras_chave": ["termo 1", "termo 2"],
  "frame_predominante": "NOME_EXATO_DO_FRAME",
  "area": "item da taxonomia de área",
  "abrangencia": "NACIONAL | INTERNACIONAL | NAO_CONCLUSIVO",
  "instituicoes": ["nome 1", "nome 2"],
  "pessoas": ["nome 1", "nome 2"]
}}
"""


def classificar_materia(row):
    prompt = montar_prompt(row)

    resposta = client.chat.completions.create(
        model=MODELO,
        messages=[
            {
                "role": "system",
                "content": (
                    "Você classifica matérias jornalísticas de ciência em "
                    "múltiplas dimensões, para análise de cobertura da mídia."
                )
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.1
    )

    dados = extrair_json(resposta.choices[0].message.content)

    # Validações leves: se o modelo devolver algo fora da taxonomia, não
    # trava o pipeline — só normaliza para os valores esperados quando dá,
    # e deixa passar como veio quando não reconhece (para não mascarar
    # erros de prompt silenciosamente).
    frame = str(dados.get("frame_predominante", "")).strip().upper()
    if frame not in FRAMES:
        frame = frame or "SEM_FRAME_CIENTIFICO_IDENTIFICAVEL"

    abrangencia = str(dados.get("abrangencia", "")).strip().upper()
    if abrangencia not in ABRANGENCIAS:
        abrangencia = abrangencia or "NAO_CONCLUSIVO"

    area = str(dados.get("area", "")).strip()
    # Só cai no fallback se vier vazio (a IA não respondeu nada) — se vier
    # algo fora da taxonomia mas não vazio, mantém como veio (não mascara
    # um erro de prompt), mas avisa no terminal pra ficar visível.
    if not area:
        area = AREA_NAO_IDENTIFICADA
    elif area not in AREAS_VALIDAS:
        print(f"  Aviso: área fora da taxonomia esperada: {area!r}")

    return {
        "palavras_chave": dados.get("palavras_chave", []),
        "frame_predominante": frame,
        "area": area,
        "abrangencia": abrangencia,
        "instituicoes": dados.get("instituicoes", []),
        "pessoas": dados.get("pessoas", []),
    }


def carregar_existentes():
    """Carrega o materias_keywords.csv já existente, se houver, no formato
    ATUAL (com as 5 dimensões). Um arquivo no formato antigo (só
    palavras_chave) não bate com COLUNAS_SAIDA e é tratado como ausente —
    force um reprocessamento completo, que é o esperado ao migrar para
    este formato expandido."""
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
        print(f"Aviso: {ARQUIVO_SAIDA} existente está num formato antigo "
              f"(faltando: {colunas_faltando}). Isso é esperado ao migrar "
              "para o novo formato com enquadramento/área/abrangência/"
              "instituições/pessoas — todas as matérias serão reprocessadas.")
        return pd.DataFrame(columns=COLUNAS_SAIDA)

    return existentes


def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("Configure OPENAI_API_KEY.")

    df = carregar_corpus()

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
            f"têm classificação completa em {ARQUIVO_SAIDA}."
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
            classificacao = classificar_materia(row)
        except Exception as e:
            print("  Erro:", e)
            classificacao = {
                "palavras_chave": [], "frame_predominante": "ERRO",
                "area": "ERRO", "abrangencia": "ERRO",
                "instituicoes": [], "pessoas": [],
            }

        resultados.append({
            "date": row["Data"],
            "ano": row["Ano"],
            "section": row.get("section", ""),
            "title": row.get("title", ""),
            "url": row.get("url", ""),
            "palavras_chave": json.dumps(classificacao["palavras_chave"], ensure_ascii=False),
            "frame_predominante": classificacao["frame_predominante"],
            "area": classificacao["area"],
            "abrangencia": classificacao["abrangencia"],
            "instituicoes": json.dumps(classificacao["instituicoes"], ensure_ascii=False),
            "pessoas": json.dumps(classificacao["pessoas"], ensure_ascii=False),
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