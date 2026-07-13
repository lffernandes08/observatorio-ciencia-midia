"""
Scraper do tópico de Ciência da BBC News Brasil.

Usa a página de tópico "https://www.bbc.com/portuguese/topics/cr50y580rjxt"
(e suas páginas seguintes ?page=2, ?page=3, ...) — paginação de verdade via
URL, no mesmo espírito da página "tudo-sobre" da CNN Brasil. Isso significa:

  - Sem risco de reset no meio da sessão (cada página é uma navegação
    direta, sem estado de "carregado" acumulado no navegador).
  - Backfill retomável de verdade: o progresso é salvo como "última página
    processada" e a próxima execução continua exatamente dali.

Particularidades desta fonte (confirmadas a partir do HTML real da página):
  - O contêiner da listagem tem um atributo estável: ul[data-testid=
    'topic-promos'] — mais confiável que as classes CSS hasheadas
    (ex: "css-psvf5b"), que podem mudar a qualquer build do site.
  - A data de publicação vem no atributo `datetime` de um <time> dentro de
    `.metadata-and-topic-data` (formato AAAA-MM-DD). Itens de vídeo/áudio
    têm um SEGUNDO <time>, com a DURAÇÃO no formato ISO 8601 (ex:
    "PT6M57S") — esse precisa ser ignorado, não é uma data.
  - Itens de vídeo/áudio são deliberadamente IGNORADOS nesta coleta (por
    pedido explícito) — são identificados pelo prefixo de acessibilidade
    "Vídeo, " ou "Áudio, " colado no início do título bruto.
  - Não há um rótulo de editoria por matéria nesta página (diferente da
    CNN Brasil) — como é uma página de tópico único, a editoria é fixada
    como "ciência" para todas as matérias coletadas por este script.

Modos de uso:
    python bbc_brasil.py --auto
        Coleta incremental diária, com margem de segurança de 4 dias e
        catch-up automático de gaps (mesmo espírito do --auto da Folha e
        da CNN Brasil).

    python bbc_brasil.py --historico-dias 558
        Backfill: percorre páginas sequenciais retomando de onde a última
        execução parou. Rode de novo (mesmo comando) para continuar.

    python bbc_brasil.py --historico-dias 558 --reiniciar
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


ARQUIVO_SAIDA = "bbc_brasil.csv"
ARQUIVO_PROGRESSO = "bbc_brasil_progresso.txt"
VEICULO = "BBC News Brasil"
EDITORIA = "ciência"  # página de tópico único; não há rótulo por matéria nesta fonte
URL_BASE_TOPICO = "https://www.bbc.com/portuguese/topics/cr50y580rjxt"

SELETOR_CARD = "ul[data-testid='topic-promos'] > li"
SELETOR_TITULO_LINK = "h2 a"
SELETOR_DATA_NO_CARD = ".metadata-and-topic-data time"

FORMATO_DATA_ATRIBUTO = "%Y-%m-%d"
PADRAO_DATA_ISO = re.compile(r"^\d{4}-\d{2}-\d{2}$")  # distingue de duração tipo "PT6M57S"
PADRAO_PREFIXO_MIDIA = re.compile(r"^(Vídeo|Áudio),")

MAX_PAGINAS_POR_EXECUCAO = 40
INTERVALO_ENTRE_PAGINAS = 1.5
INTERVALO_ENTRE_MATERIAS = 0.5
MARGEM_SEGURANCA_DIAS = 4


def montar_url_pagina(pagina):
    if pagina <= 1:
        return URL_BASE_TOPICO
    return f"{URL_BASE_TOPICO}?page={pagina}"


def carregar_urls_existentes():
    if not os.path.exists(ARQUIVO_SAIDA):
        return set()
    try:
        df_existente = pd.read_csv(ARQUIVO_SAIDA, encoding="utf-8-sig")
        return set(df_existente["url"].dropna().astype(str))
    except Exception:
        return set()


def obter_ultima_data_coletada():
    """Data mais recente já presente em bbc_brasil.csv — usada no modo
    --auto para estender a cobertura além da margem de segurança fixa
    quando houver um gap maior, espelhando main.py e cnn_brasil.py."""
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


def eh_video_ou_audio(titulo_bruto):
    """Itens de vídeo/áudio têm o texto de acessibilidade 'Vídeo, ' ou
    'Áudio, ' colado no início do título (span visually-hidden-text) —
    esses itens são deliberadamente ignorados nesta coleta."""
    return bool(PADRAO_PREFIXO_MIDIA.match(titulo_bruto.strip()))


def parsear_data_card(valor_datetime_attr):
    """Extrai a data a partir do atributo datetime do <time> de
    publicação. Ignora explicitamente valores em formato de duração
    (ex: 'PT6M57S'), que pertencem ao <time> do ícone de mídia, não à
    data de publicação."""
    if not valor_datetime_attr:
        return None
    valor = valor_datetime_attr.strip()
    if not PADRAO_DATA_ISO.match(valor):
        return None
    try:
        return datetime.strptime(valor, FORMATO_DATA_ATRIBUTO)
    except ValueError:
        return None


def carregar_pagina_com_retry(driver, url, tentativas=3, espera_entre_tentativas=8):
    """Tenta carregar a URL, com novas tentativas em caso de timeout de
    rede/conexão. Coletas longas (centenas de páginas) têm chance real de
    esbarrar num timeout pontual do navegador ou da rede — isso não
    significa que a página não existe, então vale tentar de novo antes
    de desistir."""
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
            EC.presence_of_element_located((By.CSS_SELECTOR, SELETOR_CARD))
        )
    except Exception:
        return []

    cards_dom = driver.find_elements(By.CSS_SELECTOR, SELETOR_CARD)
    resultados = []
    pulados_midia = 0

    for card in cards_dom:
        try:
            link = card.find_element(By.CSS_SELECTOR, SELETOR_TITULO_LINK)
            url_materia = link.get_attribute("href")
            titulo_bruto = link.text.strip()

            if not url_materia or not titulo_bruto:
                continue

            if eh_video_ou_audio(titulo_bruto):
                pulados_midia += 1
                continue

            try:
                elemento_data = card.find_element(By.CSS_SELECTOR, SELETOR_DATA_NO_CARD)
                valor_datetime = elemento_data.get_attribute("datetime")
            except Exception:
                valor_datetime = None

            data_dt = parsear_data_card(valor_datetime)

            if url_materia and data_dt:
                resultados.append({
                    "url": url_materia,
                    "title": titulo_bruto,
                    "date_dt": data_dt,
                })

        except Exception:
            continue

    if pulados_midia > 0:
        print(f"    ({pulados_midia} item(ns) de vídeo/áudio ignorado(s) nesta página)")

    return resultados


def coletar_e_salvar_paginas(data_limite_dt, max_paginas, pagina_inicial, urls_existentes):
    """Percorre páginas sequenciais, extraindo e salvando o conteúdo de
    cada uma IMEDIATAMENTE após coletá-la — não no final. Isso garante que,
    se o processo cair no meio (ex: timeout de rede numa página distante),
    tudo que já foi processado até ali continua salvo em bbc_brasil.csv, e
    o checkpoint de página reflete exatamente até onde os dados foram
    extraídos de verdade (não só até onde a listagem foi lida)."""
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

            # Checkpoint só avança DEPOIS de extrair e salvar com sucesso —
            # garante que "última página processada" significa "já está no CSV",
            # não só "já foi lida da listagem".
            salvar_ultima_pagina_processada(pagina_atual)

            if data_mais_antiga_da_pagina < data_limite_dt:
                motivo_parada = "atingiu_data_limite"
                break

            time.sleep(INTERVALO_ENTRE_PAGINAS)

    finally:
        driver.quit()

    return motivo_parada, pagina_atual, total_processado_nesta_execucao, total_no_arquivo


PADRAO_URL_DOMINIO = re.compile(r'(https?://|www\.|\.com\b|\.com\.br\b|\.org\b|\.net\b)', re.IGNORECASE)


def limpar_autores(lista_autores):
    """O news-please, em algumas matérias da BBC, captura um link de
    'compartilhar no Facebook' (que fica ao lado do nome do autor na
    página) como se fosse um autor a mais — ex: ['André Biernath',
    'www.facebook.com']. Filtra qualquer item que pareça URL/domínio em
    vez de nome de pessoa."""
    if not lista_autores:
        return lista_autores

    autores_limpos = [
        str(autor).strip() for autor in lista_autores
        if autor and not PADRAO_URL_DOMINIO.search(str(autor))
    ]

    # Se o filtro removeu tudo (ex: só havia lixo, sem nenhum nome real),
    # mantém a lista original — melhor mostrar o dado bruto do que
    # silenciosamente esconder que não há autor identificado.
    return autores_limpos if autores_limpos else lista_autores


def extrair_conteudo(cards_novos):
    textos = []

    for card in cards_novos:
        try:
            article = NewsPlease.from_url(card["url"])

            textos.append({
                "url": card["url"],
                "section": EDITORIA,
                "title": article.title or card["title"],
                "text": article.maintext,
                "description": article.description,
                "author": limpar_autores(article.authors),
                "image_url": article.image_url,
                "date": card["date_dt"].strftime("%d/%m/%Y"),
                "veiculo": VEICULO,
            })
            print(f"  ✓ {article.title or card['title']}")

        except Exception as e:
            textos.append({
                "url": card["url"],
                "section": EDITORIA,
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
    parser = argparse.ArgumentParser(description="Scraper do tópico de Ciência da BBC News Brasil.")
    parser.add_argument("--auto", action="store_true",
                         help="Coleta incremental diária, com margem de segurança de "
                              f"{MARGEM_SEGURANCA_DIAS} dias e catch-up automático de gaps.")
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
            "(F12) e atualize as constantes SELETOR_*/URL_BASE_TOPICO no topo deste arquivo."
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