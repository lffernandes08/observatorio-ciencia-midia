"""
Scraper da editoria de Ciência do G1 (globo.com).

Diferente da CNN e da BBC, o G1 embute os dados da listagem como um JSON
estruturado dentro da própria página (num <script> chamado
"bstn-launcher-bundle"), e expõe uma função JavaScript pronta para lê-lo:
window.bstn.debugEmbedData(). Isso é usado como estratégia PRIMÁRIA de
extração aqui — via driver.execute_script(), sem precisar de seletor CSS
algum — porque é mais preciso que fazer parsing de HTML:
  - Datas vêm em ISO 8601 exato (não texto relativo tipo "Há 3 horas").
  - A editoria real de cada matéria vem no campo content.section (a
    página de "Ciência" mistura outras seções, como "Sul de Minas",
    "Fato ou Fake", "Blog Longevidade: modo de usar" — confirmado no
    HTML real inspecionado).
  - O tipo de cada item (content.type / item.type) já distingue matéria
    de texto ("materia") de vídeo ("video") e anúncio ("advertise") —
    só "materia" é coletado aqui.

Paginação: a página inicial (https://g1.globo.com/ciencia/) já traz um
JSON com um campo "nextPage" apontando para a próxima página de conteúdo
(ex: 4 — o G1 pré-carrega várias "páginas" internas na carga inicial,
então a numeração pula). As páginas seguintes usam a URL
https://g1.globo.com/ciencia/index/feed/pagina-N.ghtml, confirmada pelo
link "Mostrar mais" no HTML real. Em vez de assumir uma sequência fixa
(N, N+1, N+2...), o scraper SEGUE o valor de "nextPage" retornado a cada
página — mais robusto a qualquer particularidade da numeração interna do
G1. O checkpoint salvo reflete esse valor literal.

⚠️ A extração via JSON foi validada com dados reais da PRIMEIRA página.
Não há confirmação ainda de que as páginas seguintes (pagina-N.ghtml)
embutem o mesmo JSON da mesma forma — se o teste inicial encontrar
"pagina_vazia" logo na primeira página seguinte à inicial, isso é sinal
de que essa suposição precisa de ajuste.

Modos de uso:
    python g1_globo.py --auto
        Coleta incremental diária, com margem de segurança de 4 dias e
        catch-up automático de gaps (mesmo espírito da CNN e da BBC).

    python g1_globo.py --historico-dias 558
        Backfill: segue nextPage retomando de onde a última execução
        parou. Rode de novo (mesmo comando) para continuar.

    python g1_globo.py --historico-dias 558 --reiniciar
        Mesmo que acima, mas ignora o progresso salvo e recomeça da
        página inicial.
"""

import os
import re
import sys

# O Windows costuma redirecionar stdout/stderr para o console/log com a
# codificação cp1252 (Windows-1252), que não tem representação para vários
# símbolos usados nos prints deste script (✓, ✗) — isso derruba o script
# inteiro com UnicodeEncodeError bem no meio de uma extração, mesmo com a
# matéria já processada com sucesso. Força UTF-8 explicitamente; se algum
# caractere realmente não puder ser mostrado, troca por um substituto em
# vez de travar o processo inteiro.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
import time
import argparse
from datetime import datetime, timedelta

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from newsplease import NewsPlease
import pandas as pd


ARQUIVO_SAIDA = "g1_globo.csv"
ARQUIVO_PROGRESSO = "g1_globo_progresso.txt"
VEICULO = "G1"
EDITORIA_PADRAO = "ciência"

URL_PRIMEIRA_PAGINA = "https://g1.globo.com/ciencia/"
URL_PAGINA_TEMPLATE = "https://g1.globo.com/ciencia/index/feed/pagina-{numero}.ghtml"

TIPOS_ACEITOS = {"materia"}  # exclui "video" e "advertise"

PADRAO_DATA_ISO_INICIO = re.compile(r"^\d{4}-\d{2}-\d{2}")

# =========================
# Fallback via DOM: usado quando a página não embute o JSON esperado
# (window.bstn.debugEmbedData ausente) -- confirmado que acontece em
# páginas de continuação de feed mais profundas (pagina-N.ghtml com N
# grande), mesmo a página tendo conteúdo normal visível no navegador.
# =========================
SELETOR_CARD_DOM = "div.bastian-feed-item[data-type='materia']"
SELETOR_LINK_MOSTRAR_MAIS = "a[aria-label='Mostrar mais conteúdos']"
PADRAO_NUMERO_PAGINA_URL = re.compile(r"pagina-(\d+)\.ghtml")

MESES_PT = {
    "janeiro": 1, "fevereiro": 2, "março": 3, "abril": 4, "maio": 5, "junho": 6,
    "julho": 7, "agosto": 8, "setembro": 9, "outubro": 10, "novembro": 11, "dezembro": 12,
}


MAX_PAGINAS_POR_EXECUCAO = 40
INTERVALO_ENTRE_PAGINAS = 1.5
INTERVALO_ENTRE_MATERIAS = 0.5
MARGEM_SEGURANCA_DIAS = 4


def montar_url_pagina(numero_pagina):
    if numero_pagina is None or numero_pagina <= 1:
        return URL_PRIMEIRA_PAGINA
    return URL_PAGINA_TEMPLATE.format(numero=numero_pagina)


def carregar_urls_existentes():
    if not os.path.exists(ARQUIVO_SAIDA):
        return set()
    try:
        df_existente = pd.read_csv(ARQUIVO_SAIDA, encoding="utf-8-sig")
        return set(df_existente["url"].dropna().astype(str))
    except Exception:
        return set()


def obter_ultima_data_coletada():
    """Data mais recente já presente em g1_globo.csv — usada no modo
    --auto para catch-up de gaps, espelhando main.py/cnn_brasil.py/
    bbc_brasil.py."""
    if not os.path.exists(ARQUIVO_SAIDA):
        return None
    try:
        df_existente = pd.read_csv(ARQUIVO_SAIDA, encoding="utf-8-sig")
        datas = pd.to_datetime(df_existente["date"], format="%d/%m/%Y", errors="coerce").dropna()
        if datas.empty:
            return None
        return datas.max()
    except Exception:
        return None


def carregar_proxima_pagina():
    """Retorna None se não houver checkpoint (começa da página inicial)."""
    if not os.path.exists(ARQUIVO_PROGRESSO):
        return None
    try:
        with open(ARQUIVO_PROGRESSO, "r", encoding="utf-8") as f:
            valor = f.read().strip()
            return int(valor) if valor else None
    except Exception:
        return None


def salvar_proxima_pagina(pagina):
    if pagina is None:
        return
    with open(ARQUIVO_PROGRESSO, "w", encoding="utf-8") as f:
        f.write(str(pagina))


def parsear_data_iso(valor_iso):
    """Extrai só a data (AAAA-MM-DD) do início de um timestamp ISO 8601,
    ignorando hora e fuso — o G1 mistura formatos ('...Z' e '...-03:00'),
    mas ambos começam com AAAA-MM-DD, então isso lida com os dois."""
    if not valor_iso:
        return None
    match = PADRAO_DATA_ISO_INICIO.match(valor_iso)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(0), "%Y-%m-%d")
    except ValueError:
        return None


def parsear_data_relativa_ou_absoluta(texto):
    """Parser de data usado só no fallback via DOM (sem timestamp ISO
    disponível) -- o texto exibido é relativo ('Há 3 horas', 'Há 2 dias')
    para conteúdo recente, ou absoluto ('12 julho 2026') para mais
    antigo. Menos preciso que o ISO (só dá a data, não a hora exata),
    mas suficiente para o nível de granularidade usado no projeto."""
    if not texto:
        return None

    texto = texto.strip().lower()
    agora = datetime.now()

    match_relativo = re.match(r"há (\d+) (hora|dia)s?", texto)
    if match_relativo:
        quantidade = int(match_relativo.group(1))
        unidade = match_relativo.group(2)
        delta = timedelta(hours=quantidade) if unidade == "hora" else timedelta(days=quantidade)
        return (agora - delta).replace(hour=0, minute=0, second=0, microsecond=0)

    match_absoluto = re.match(r"(\d{1,2})\s+([a-zçã]+)\s+(\d{4})", texto)
    if match_absoluto:
        dia = int(match_absoluto.group(1))
        mes = MESES_PT.get(match_absoluto.group(2))
        ano = int(match_absoluto.group(3))
        if mes:
            try:
                return datetime(ano, mes, dia)
            except ValueError:
                return None

    return None


def extrair_dados_via_dom(driver):
    """Extração alternativa via seletores CSS, usada quando a página não
    embute o JSON esperado. O seletor de card já filtra só 'materia'
    diretamente via atributo (data-type='materia'), mesmo espírito da
    lista de permissão usada na extração via JSON."""
    try:
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, SELETOR_CARD_DOM))
        )
    except Exception:
        pass  # segue tentando mesmo assim -- se não aparecer nada, tratamos como página vazia de verdade

    cards_dom = driver.find_elements(By.CSS_SELECTOR, SELETOR_CARD_DOM)
    resultados = []

    for card in cards_dom:
        try:
            url_materia = card.get_attribute("data-mrf-link")

            try:
                titulo = card.find_element(By.CSS_SELECTOR, "h2 a").text.strip()
            except Exception:
                titulo = None

            try:
                texto_data = card.find_element(By.CSS_SELECTOR, ".feed-post-datetime").text
            except Exception:
                texto_data = None

            try:
                secao = card.find_element(By.CSS_SELECTOR, ".feed-post-metadata-section").text.strip().lower()
            except Exception:
                secao = EDITORIA_PADRAO

            data_dt = parsear_data_relativa_ou_absoluta(texto_data)

            if url_materia and titulo and data_dt:
                resultados.append({
                    "url": url_materia,
                    "title": titulo,
                    "date_dt": data_dt,
                    "section": secao or EDITORIA_PADRAO,
                })

        except Exception:
            continue

    proxima_pagina = None
    try:
        link_mostrar_mais = driver.find_element(By.CSS_SELECTOR, SELETOR_LINK_MOSTRAR_MAIS)
        href = link_mostrar_mais.get_attribute("href") or ""
        match = PADRAO_NUMERO_PAGINA_URL.search(href)
        if match:
            proxima_pagina = int(match.group(1))
    except Exception:
        pass

    return resultados, proxima_pagina


def carregar_pagina_com_retry(driver, url, tentativas=3, espera_entre_tentativas=8):
    """Tenta carregar a URL, com novas tentativas em caso de timeout de
    rede/conexão — mesmo mecanismo usado em cnn_brasil.py e bbc_brasil.py."""
    ultimo_erro = None
    for tentativa in range(1, tentativas + 1):
        try:
            driver.get(url)
            return True
        except Exception as e:
            ultimo_erro = e
            print(f"    Tentativa {tentativa}/{tentativas} falhou "
                  f"({e.__class__.__name__}), aguardando {espera_entre_tentativas}s...")
            time.sleep(espera_entre_tentativas)

    print(f"    Falha definitiva ao carregar a página após {tentativas} tentativas: {ultimo_erro}")
    return False


def extrair_dados_embutidos(driver):
    """Lê o JSON estruturado que o G1 já embute na página via
    window.bstn.debugEmbedData() — ver docstring do módulo."""
    try:
        WebDriverWait(driver, 15).until(
            lambda d: d.execute_script(
                "return !!(window.bstn && window.bstn.debugEmbedData);"
            )
        )
        return driver.execute_script("return window.bstn.debugEmbedData();")
    except Exception:
        return None


def coletar_pagina(driver, numero_pagina):
    """Retorna (lista_de_cards, proxima_pagina). Tenta primeiro extração
    via JSON (window.bstn.debugEmbedData(), mais precisa); se não vier
    nada útil, cai para extração via DOM (seletores CSS, menos precisa
    na data mas funciona em páginas onde o JSON não está disponível —
    confirmado que acontece em páginas de feed mais profundas).
    Retorna (None, None) em caso de falha de conexão."""
    url = montar_url_pagina(numero_pagina)

    if not carregar_pagina_com_retry(driver, url):
        return None, None

    dados = extrair_dados_embutidos(driver)
    resultados = []
    proxima_pagina = None
    motivo_fallback = None

    if not dados:
        motivo_fallback = "window.bstn.debugEmbedData() não retornou nada"
    elif "items" not in dados:
        motivo_fallback = "JSON veio sem a chave 'items' esperada"
    else:
        itens_brutos = dados.get("items", [])
        contagem_por_tipo = {}
        for item in itens_brutos:
            tipo = item.get("type", "desconhecido")
            contagem_por_tipo[tipo] = contagem_por_tipo.get(tipo, 0) + 1

        for item in itens_brutos:
            if item.get("type") not in TIPOS_ACEITOS:
                continue

            content = item.get("content", {})
            url_materia = content.get("url")
            titulo = content.get("title")
            secao_bruta = content.get("section")
            secao = secao_bruta.strip().lower() if secao_bruta else EDITORIA_PADRAO
            data_dt = parsear_data_iso(item.get("publication"))

            if url_materia and titulo and data_dt:
                resultados.append({
                    "url": url_materia,
                    "title": titulo.strip(),
                    "date_dt": data_dt,
                    "section": secao,
                })

        proxima_pagina = dados.get("nextPage")

        if not resultados:
            motivo_fallback = (
                f"JSON veio com {len(itens_brutos)} item(ns), mas nenhum é "
                f"'materia' com dados completos (tipos: {contagem_por_tipo or '(nenhum item)'})"
            )

    if resultados:
        return resultados, proxima_pagina

    print(f"    JSON indisponível/vazio nesta página ({motivo_fallback}) "
          "— tentando extração via DOM (fallback)...")

    resultados_dom, proxima_pagina_dom = extrair_dados_via_dom(driver)

    if resultados_dom:
        print(f"    Fallback via DOM funcionou: {len(resultados_dom)} matéria(s) encontrada(s).")
        return resultados_dom, proxima_pagina_dom

    print("    Fallback via DOM também não encontrou nenhuma matéria nesta página.")
    return [], proxima_pagina_dom or proxima_pagina


def coletar_e_salvar_paginas(data_limite_dt, max_paginas, pagina_inicial, urls_existentes):
    """Segue nextPage a partir de pagina_inicial, extraindo e salvando o
    conteúdo de cada página IMEDIATAMENTE após coletá-la (mesma lógica
    de resiliência de cnn_brasil.py/bbc_brasil.py: nada se perde se o
    processo cair no meio de um backfill longo)."""
    options = webdriver.ChromeOptions()
    driver = webdriver.Chrome(options=options)

    motivo_parada = "max_paginas"
    pagina_atual = pagina_inicial
    total_processado_nesta_execucao = 0
    total_no_arquivo = None

    try:
        driver.maximize_window()

        for _ in range(max_paginas):
            cards_pagina, proxima_pagina = coletar_pagina(driver, pagina_atual)

            if cards_pagina is None:
                motivo_parada = "erro_conexao"
                break

            if not cards_pagina:
                motivo_parada = "pagina_vazia"
                break

            data_mais_antiga_da_pagina = min(c["date_dt"] for c in cards_pagina)

            cards_novos = [c for c in cards_pagina if c["url"] not in urls_existentes]
            cards_no_periodo = [c for c in cards_novos if c["date_dt"] >= data_limite_dt]

            rotulo_pagina = pagina_atual if pagina_atual else 1
            print(f"  Página {rotulo_pagina}: {len(cards_pagina)} matéria(s) na listagem, "
                  f"{len(cards_no_periodo)} nova(s) dentro do período "
                  f"(mais antiga nesta página: {data_mais_antiga_da_pagina.strftime('%d/%m/%Y')})")

            if cards_no_periodo:
                df_pagina = extrair_conteudo(cards_no_periodo)
                total_no_arquivo = salvar_incremental(df_pagina)
                urls_existentes.update(c["url"] for c in cards_no_periodo)
                total_processado_nesta_execucao += len(cards_no_periodo)

            if data_mais_antiga_da_pagina < data_limite_dt:
                motivo_parada = "atingiu_data_limite"
                salvar_proxima_pagina(proxima_pagina)
                break

            if not proxima_pagina:
                motivo_parada = "sem_proxima_pagina"
                break

            salvar_proxima_pagina(proxima_pagina)
            pagina_atual = proxima_pagina

            time.sleep(INTERVALO_ENTRE_PAGINAS)

    finally:
        driver.quit()

    return motivo_parada, pagina_atual, total_processado_nesta_execucao, total_no_arquivo


def extrair_conteudo(cards_novos):
    textos = []

    for card in cards_novos:
        try:
            article = NewsPlease.from_url(card["url"])

            textos.append({
                "url": card["url"],
                "section": card.get("section", EDITORIA_PADRAO),
                "title": article.title or card["title"],
                "text": article.maintext,
                "description": article.description,
                "author": article.authors,
                "image_url": article.image_url,
                "date": card["date_dt"].strftime("%d/%m/%Y"),
                "veiculo": VEICULO,
            })
            print(f"  ✓ {article.title or card['title']}")

        except Exception as e:
            textos.append({
                "url": card["url"],
                "section": card.get("section", EDITORIA_PADRAO),
                "title": card["title"],
                "text": "ERRO!!!",
                "description": "ERRO!!!",
                "author": "ERRO!!!",
                "image_url": "ERRO!!!",
                "date": card["date_dt"].strftime("%d/%m/%Y"),
                "veiculo": VEICULO,
            })
            print(f"  ✗ Erro ao extrair {card['url']}: {e}")

        time.sleep(INTERVALO_ENTRE_MATERIAS)

    return pd.DataFrame(textos)


def salvar_incremental(df_novo):
    if df_novo.empty:
        return 0

    if os.path.exists(ARQUIVO_SAIDA):
        df_existente = pd.read_csv(ARQUIVO_SAIDA, encoding="utf-8-sig")
        df_combinado = pd.concat([df_existente, df_novo], ignore_index=True)
    else:
        df_combinado = df_novo

    antes = len(df_combinado)
    df_combinado = df_combinado.drop_duplicates(subset="url", keep="first")
    duplicadas = antes - len(df_combinado)

    df_combinado.to_csv(ARQUIVO_SAIDA, index=False, encoding="utf-8-sig")

    if duplicadas > 0:
        print(f"{duplicadas} duplicata(s) descartada(s) na deduplicação por URL.")

    return len(df_combinado)


def main():
    parser = argparse.ArgumentParser(description="Scraper da editoria de Ciência do G1.")
    parser.add_argument("--auto", action="store_true",
                         help="Coleta incremental diária, com margem de segurança de "
                              f"{MARGEM_SEGURANCA_DIAS} dias e catch-up automático de gaps.")
    parser.add_argument("--historico-dias", type=int,
                         help="Tenta coletar retroativamente até N dias atrás, retomando "
                              "de onde a última execução parou (backfill).")
    parser.add_argument("--max-paginas", type=int, default=MAX_PAGINAS_POR_EXECUCAO,
                         help=f"Teto de páginas percorridas nesta execução "
                              f"(padrão: {MAX_PAGINAS_POR_EXECUCAO}).")
    parser.add_argument("--reiniciar", action="store_true",
                         help="Ignora o progresso salvo e recomeça o backfill da página inicial.")
    args = parser.parse_args()

    if not args.auto and not args.historico_dias:
        print("Especifique --auto (coleta diária) ou --historico-dias N (backfill).")
        sys.exit(1)

    # Normalizado para meia-noite: sem isso, a hora exata em que o script
    # roda vaza pro cálculo de data_limite_dt, fazendo a comparação >=
    # excluir incorretamente matérias do próprio dia-limite (que têm hora
    # 00:00:00), mesmo estando dentro do período pedido.
    hoje_dt = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    if args.auto:
        limite_seguranca_dt = hoje_dt - timedelta(days=MARGEM_SEGURANCA_DIAS)
        ultima_data_coletada = obter_ultima_data_coletada()

        if ultima_data_coletada is None:
            data_limite_dt = limite_seguranca_dt
        else:
            inicio_catchup_dt = ultima_data_coletada + timedelta(days=1)
            data_limite_dt = min(inicio_catchup_dt, limite_seguranca_dt)

        pagina_inicial = None  # sempre começa da página inicial

        if ultima_data_coletada is not None and data_limite_dt < limite_seguranca_dt:
            print(
                f"Modo automático: gap detectado (última coleta em "
                f"{ultima_data_coletada.strftime('%d/%m/%Y')}) — estendendo "
                f"cobertura até {data_limite_dt.strftime('%d/%m/%Y')} em vez "
                f"da margem padrão de {MARGEM_SEGURANCA_DIAS} dias."
            )
        else:
            print(f"Modo automático: coletando até {data_limite_dt.strftime('%d/%m/%Y')} "
                  f"(margem de segurança de {MARGEM_SEGURANCA_DIAS} dias).")
    else:
        data_limite_dt = hoje_dt - timedelta(days=args.historico_dias)

        if args.reiniciar:
            pagina_inicial = None
            print("(--reiniciar) Ignorando progresso salvo, começando da página inicial.")
        else:
            pagina_inicial = carregar_proxima_pagina()

        print(f"Modo histórico: tentando coletar até {data_limite_dt.strftime('%d/%m/%Y')} "
              f"({args.historico_dias} dias atrás), retomando a partir da página "
              f"{pagina_inicial if pagina_inicial else 1}, com teto de "
              f"{args.max_paginas} página(s) nesta execução.")

    urls_existentes = carregar_urls_existentes()

    motivo_parada, ultima_pagina, total_processado, total_no_arquivo = coletar_e_salvar_paginas(
        data_limite_dt, args.max_paginas, pagina_inicial, urls_existentes
    )

    if total_processado == 0 and motivo_parada == "pagina_vazia" and pagina_inicial is None:
        print(
            "Nenhuma matéria foi encontrada logo na primeira página — o JSON esperado "
            "(window.bstn.debugEmbedData()) provavelmente não veio como esperado. "
            "Isso pode indicar que o G1 mudou a estrutura da página; inspecione o "
            "HTML/JS atual e ajuste a extração em extrair_dados_embutidos()."
        )
        sys.exit(1)

    print(f"\nColeta finalizada: {total_processado} matéria(s) nova(s) processada(s) "
          f"nesta execução (motivo da parada: {motivo_parada}, última página: "
          f"{ultima_pagina if ultima_pagina else 1}).")

    if total_no_arquivo is not None:
        print(f"Total agora em {ARQUIVO_SAIDA}: {total_no_arquivo} matéria(s).")

    if motivo_parada == "max_paginas" and args.historico_dias:
        print(
            f"⚠ Atingiu o teto de {args.max_paginas} página(s) antes de alcançar "
            f"{data_limite_dt.strftime('%d/%m/%Y')}. Rode o mesmo comando de novo "
            "para continuar — a próxima execução retoma direto de onde parou."
        )
    elif motivo_parada == "erro_conexao":
        print(
            "⚠ Parou por falha de conexão persistente (depois de 3 tentativas). "
            "Tudo que já foi processado até aqui já está salvo — rode o mesmo "
            "comando de novo pra continuar."
        )
    elif motivo_parada == "sem_proxima_pagina":
        print(
            "Chegou ao fim do conteúdo paginado disponível (sem mais 'nextPage')."
        )


if __name__ == "__main__":
    main()