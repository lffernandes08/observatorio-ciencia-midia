"""
Scraper de conteúdo de Ciência da CNN Brasil.

Usa a página de tema "https://www.cnnbrasil.com.br/tudo-sobre/ciencia/"
(e suas páginas seguintes .../pagina/2/, .../pagina/3/, ...), que tem
paginação de verdade via URL — bem mais robusta que a listagem da editoria
(.../ciencia/), que só pagina via um botão "Carregar mais" carregado por
JavaScript. Essa mudança resolve dois problemas observados na versão
anterior:

  1. Reset no meio da sessão: como cada página é uma navegação direta
     (driver.get), não existe estado de "carregado" acumulado no navegador
     para se perder — não há mais o risco de "a página voltar ao início".

  2. Retomada real de backfill: como as páginas são numeradas, o progresso
     de um backfill grande pode ser salvo como "última página processada"
     e retomado exatamente dali na próxima execução — antes, era preciso
     reclicar desde o começo a cada nova tentativa.

Importante: a página "tudo sobre Ciência" agrega matérias de qualquer
editoria marcada com o tema Ciência (vi matérias de "Saúde" junto com
"Ciência" na inspeção manual) — por isso a editoria de cada matéria é
capturada individualmente do card, em vez de fixada como "ciência".

Modos de uso:
    python cnn_brasil.py --auto
        Coleta incremental diária: percorre a partir da página 1 até
        cobrir a margem de segurança (mesmo espírito do --auto do main.py
        da Folha). Sempre começa da página 1 (conteúdo mais recente).

    python cnn_brasil.py --historico-dias 365
        Backfill: percorre páginas sequenciais retomando de onde a última
        execução parou (arquivo de progresso), até alcançar ~365 dias de
        histórico ou esgotar o teto de páginas desta execução. Rode de
        novo (mesmo comando) para continuar de onde parou.

    python cnn_brasil.py --historico-dias 365 --reiniciar
        Mesmo que acima, mas ignora o progresso salvo e recomeça da
        página 1.
"""

import os
import re
import sys
import time
import argparse
from datetime import datetime, timedelta

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from newsplease import NewsPlease
import pandas as pd


ARQUIVO_SAIDA = "cnn_brasil.csv"
ARQUIVO_PROGRESSO = "cnn_brasil_progresso.txt"
VEICULO = "CNN Brasil"
URL_BASE_PAGINADA = "https://www.cnnbrasil.com.br/tudo-sobre/ciencia"

SELETOR_TITULO_LINK = "[id^='titulo-noticia-'] a"   # robusto a diferença de heading (h2 na página de tema, h3 na de editoria)
XPATH_ANCESTRAL_CARD = "./ancestor::li[1]"
SELETOR_DATA_NO_CARD = "time"
SELETOR_EDITORIA_NO_CARD = "a[aria-label^='Ver mais sobre']"

FORMATO_DATETIME_ATRIBUTO = "%Y-%m-%d %H:%M:%S"
PADRAO_DATA_TEXTO = re.compile(r"(\d{2}/\d{2}/\d{4})")

MAX_PAGINAS_POR_EXECUCAO = 40
INTERVALO_ENTRE_PAGINAS = 1.5
INTERVALO_ENTRE_MATERIAS = 0.5
MARGEM_SEGURANCA_DIAS = 4


def montar_url_pagina(pagina):
    if pagina <= 1:
        return URL_BASE_PAGINADA  # o próprio site linka a página 1 sem barra final
    return f"{URL_BASE_PAGINADA}/pagina/{pagina}/"


def carregar_urls_existentes():
    if not os.path.exists(ARQUIVO_SAIDA):
        return set()
    try:
        df_existente = pd.read_csv(ARQUIVO_SAIDA, encoding="utf-8-sig")
        return set(df_existente["url"].dropna().astype(str))
    except Exception:
        return set()


def obter_ultima_data_coletada():
    """Data mais recente já presente em cnn_brasil.csv — usada no modo
    --auto para estender a cobertura além da margem de segurança fixa
    quando houver um gap maior (ex: PC ficou vários dias desligado),
    espelhando a mesma lógica do main.py da Folha."""
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


def carregar_ultima_pagina_processada():
    if not os.path.exists(ARQUIVO_PROGRESSO):
        return 0
    try:
        with open(ARQUIVO_PROGRESSO, "r", encoding="utf-8") as f:
            return int(f.read().strip())
    except Exception:
        return 0


def salvar_ultima_pagina_processada(pagina):
    with open(ARQUIVO_PROGRESSO, "w", encoding="utf-8") as f:
        f.write(str(pagina))


def parsear_data_card(valor_datetime_attr):
    if not valor_datetime_attr:
        return None
    try:
        return datetime.strptime(valor_datetime_attr.strip()[:19], FORMATO_DATETIME_ATRIBUTO)
    except ValueError:
        match = PADRAO_DATA_TEXTO.search(valor_datetime_attr)
        if match:
            try:
                return datetime.strptime(match.group(1), "%d/%m/%Y")
            except ValueError:
                return None
        return None


def carregar_pagina_com_retry(driver, url, tentativas=3, espera_entre_tentativas=8):
    """Tenta carregar a URL, com novas tentativas em caso de timeout de
    rede/conexão — coletas longas (centenas de páginas) têm chance real de
    esbarrar num timeout pontual do navegador ou da rede."""
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


def coletar_pagina(driver, numero_pagina):
    """Retorna: lista de cards (pode ser vazia = fim do conteúdo),
    ou None = falha de conexão (diferente de 'página vazia')."""
    url = montar_url_pagina(numero_pagina)

    if not carregar_pagina_com_retry(driver, url):
        return None

    try:
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, SELETOR_TITULO_LINK))
        )
    except Exception:
        return []

    links_titulo = driver.find_elements(By.CSS_SELECTOR, SELETOR_TITULO_LINK)
    resultados = []

    for link in links_titulo:
        try:
            url_materia = link.get_attribute("href")
            titulo = link.text.strip()

            if not url_materia or not titulo:
                continue

            try:
                card = link.find_element(By.XPATH, XPATH_ANCESTRAL_CARD)
            except Exception:
                card = link

            try:
                elemento_data = card.find_element(By.CSS_SELECTOR, SELETOR_DATA_NO_CARD)
                valor_datetime = elemento_data.get_attribute("datetime") or elemento_data.text
            except Exception:
                valor_datetime = None

            try:
                editoria = card.find_element(By.CSS_SELECTOR, SELETOR_EDITORIA_NO_CARD).text.strip()
            except Exception:
                editoria = "ciência"

            data_dt = parsear_data_card(valor_datetime)

            if url_materia and data_dt:
                resultados.append({
                    "url": url_materia,
                    "title": titulo,
                    "date_dt": data_dt,
                    "section": editoria.lower() if editoria else "ciência",
                })

        except Exception:
            continue

    return resultados


def coletar_e_salvar_paginas(data_limite_dt, max_paginas, pagina_inicial, urls_existentes):
    """Percorre páginas sequenciais, extraindo e salvando o conteúdo de
    cada uma IMEDIATAMENTE após coletá-la — não no final. Isso garante que,
    se o processo cair no meio (ex: timeout de rede numa página distante),
    tudo que já foi processado até ali continua salvo em cnn_brasil.csv."""
    options = webdriver.ChromeOptions()
    driver = webdriver.Chrome(options=options)

    motivo_parada = "max_paginas"
    pagina_atual = pagina_inicial
    total_processado_nesta_execucao = 0
    total_no_arquivo = None

    try:
        driver.maximize_window()

        for i in range(max_paginas):
            pagina_atual = pagina_inicial + i
            cards_pagina = coletar_pagina(driver, pagina_atual)

            if cards_pagina is None:
                motivo_parada = "erro_conexao"
                break

            if not cards_pagina:
                motivo_parada = "pagina_vazia"
                break

            data_mais_antiga_da_pagina = min(c["date_dt"] for c in cards_pagina)

            cards_novos = [c for c in cards_pagina if c["url"] not in urls_existentes]
            cards_no_periodo = [c for c in cards_novos if c["date_dt"] >= data_limite_dt]

            print(f"  Página {pagina_atual}: {len(cards_pagina)} matéria(s) na listagem, "
                  f"{len(cards_no_periodo)} nova(s) dentro do período "
                  f"(mais antiga nesta página: {data_mais_antiga_da_pagina.strftime('%d/%m/%Y')})")

            if cards_no_periodo:
                df_pagina = extrair_conteudo(cards_no_periodo)
                total_no_arquivo = salvar_incremental(df_pagina)
                urls_existentes.update(c["url"] for c in cards_no_periodo)
                total_processado_nesta_execucao += len(cards_no_periodo)

            # Checkpoint só avança DEPOIS de extrair e salvar com sucesso.
            salvar_ultima_pagina_processada(pagina_atual)

            if data_mais_antiga_da_pagina < data_limite_dt:
                motivo_parada = "atingiu_data_limite"
                break

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
                "section": card.get("section", "ciência"),
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
                "section": card.get("section", "ciência"),
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
    parser = argparse.ArgumentParser(description="Scraper de conteúdo de Ciência da CNN Brasil.")
    parser.add_argument("--auto", action="store_true",
                         help="Coleta incremental diária, com margem de segurança de "
                              f"{MARGEM_SEGURANCA_DIAS} dias. Sempre começa da página 1.")
    parser.add_argument("--historico-dias", type=int,
                         help="Tenta coletar retroativamente até N dias atrás, retomando "
                              "da última página processada (backfill).")
    parser.add_argument("--max-paginas", type=int, default=MAX_PAGINAS_POR_EXECUCAO,
                         help=f"Teto de páginas percorridas nesta execução "
                              f"(padrão: {MAX_PAGINAS_POR_EXECUCAO}).")
    parser.add_argument("--reiniciar", action="store_true",
                         help="Ignora o progresso salvo e recomeça o backfill da página 1.")
    args = parser.parse_args()

    if not args.auto and not args.historico_dias:
        print("Especifique --auto (coleta diária) ou --historico-dias N (backfill).")
        sys.exit(1)

    hoje_dt = datetime.now()

    if args.auto:
        limite_seguranca_dt = hoje_dt - timedelta(days=MARGEM_SEGURANCA_DIAS)
        ultima_data_coletada = obter_ultima_data_coletada()

        if ultima_data_coletada is None:
            data_limite_dt = limite_seguranca_dt
        else:
            inicio_catchup_dt = ultima_data_coletada + timedelta(days=1)
            # O mais antigo dos dois: cobre gaps grandes (PC desligado por
            # vários dias) E garante a margem de segurança mínima, mesmo
            # sem gap nenhum — mesma lógica do main.py da Folha.
            data_limite_dt = min(inicio_catchup_dt, limite_seguranca_dt)

        pagina_inicial = 1

        if ultima_data_coletada is not None and data_limite_dt < limite_seguranca_dt:
            print(
                f"Modo automático: gap detectado (última coleta em "
                f"{ultima_data_coletada.strftime('%d/%m/%Y')}) — estendendo "
                f"cobertura até {data_limite_dt.strftime('%d/%m/%Y')} em vez "
                f"da margem padrão de {MARGEM_SEGURANCA_DIAS} dias."
            )
        else:
            print(f"Modo automático: coletando até {data_limite_dt.strftime('%d/%m/%Y')} "
                  f"(margem de segurança de {MARGEM_SEGURANCA_DIAS} dias), a partir da página 1.")
    else:
        data_limite_dt = hoje_dt - timedelta(days=args.historico_dias)

        if args.reiniciar:
            pagina_inicial = 1
            print("(--reiniciar) Ignorando progresso salvo, começando da página 1.")
        else:
            pagina_inicial = carregar_ultima_pagina_processada() + 1

        print(f"Modo histórico: tentando coletar até {data_limite_dt.strftime('%d/%m/%Y')} "
              f"({args.historico_dias} dias atrás), retomando a partir da página "
              f"{pagina_inicial}, com teto de {args.max_paginas} página(s) nesta execução.")

    urls_existentes = carregar_urls_existentes()

    motivo_parada, ultima_pagina, total_processado, total_no_arquivo = coletar_e_salvar_paginas(
        data_limite_dt, args.max_paginas, pagina_inicial, urls_existentes
    )

    if total_processado == 0 and motivo_parada == "pagina_vazia" and pagina_inicial == 1:
        print(
            "Nenhuma matéria foi encontrada logo na primeira página. Os seletores "
            "provavelmente precisam ser ajustados — inspecione a página no navegador "
            "(F12) e atualize as constantes SELETOR_*/XPATH_* no topo deste arquivo."
        )
        sys.exit(1)

    print(f"\nColeta finalizada: {total_processado} matéria(s) nova(s) processada(s) "
          f"nesta execução (motivo da parada: {motivo_parada}, última página: {ultima_pagina}).")

    if total_no_arquivo is not None:
        print(f"Total agora em {ARQUIVO_SAIDA}: {total_no_arquivo} matéria(s).")

    if motivo_parada == "max_paginas" and args.historico_dias:
        print(
            f"⚠ Atingiu o teto de {args.max_paginas} página(s) antes de alcançar "
            f"{data_limite_dt.strftime('%d/%m/%Y')}. Rode o mesmo comando de novo "
            f"para continuar — a próxima execução retoma direto da página "
            f"{ultima_pagina + 1}, sem precisar refazer o que já foi processado."
        )
    elif motivo_parada == "erro_conexao":
        print(
            f"⚠ Parou por falha de conexão persistente na página {ultima_pagina + 1} "
            "(depois de 3 tentativas). Tudo que já foi processado até a página "
            f"{ultima_pagina} já está salvo — rode o mesmo comando de novo pra "
            "continuar dali."
        )


if __name__ == "__main__":
    main()