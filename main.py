"""
Scraper da busca da Folha de S.Paulo
Versão .py para rodar no VSCode.

Suporta dois modos:
  - Manual/interativo (como sempre foi): rode `python main.py` sem argumentos
    e ele pergunta as datas no terminal.
  - Automático/diário: rode `python main.py --hoje` e ele coleta apenas as
    matérias do dia atual, sem pedir nada — pensado para rodar sozinho via
    Agendador de Tarefas do Windows.

Baseado no notebook original: Selenium - Folha.ipynb
"""

import os
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


ARQUIVO_SAIDA = "folha.csv"


# =========================
# 1. Datas da busca
# =========================

def determinar_intervalo_auto():
    """Calcula o intervalo a coletar no modo --auto: do dia seguinte ao
    último dia já presente em folha.csv até hoje. Isso faz a coleta se
    autocorrigir sozinha se o computador ficar dias desligado — na próxima
    execução, ela cobre tudo que ficou faltando, não só o dia atual."""
    hoje_dt = datetime.now()

    if not os.path.exists(ARQUIVO_SAIDA):
        hoje = hoje_dt.strftime("%d/%m/%Y")
        return hoje, hoje

    try:
        df_existente = pd.read_csv(ARQUIVO_SAIDA, encoding="utf-8-sig")
        datas_existentes = pd.to_datetime(
            df_existente["date"], format="%d/%m/%Y", errors="coerce"
        ).dropna()
    except Exception:
        datas_existentes = pd.Series([], dtype="datetime64[ns]")

    if datas_existentes.empty:
        hoje = hoje_dt.strftime("%d/%m/%Y")
        return hoje, hoje

    ultimo_dia = datas_existentes.max()
    inicio_dt = ultimo_dia + timedelta(days=1)

    if inicio_dt.date() > hoje_dt.date():
        return None, None  # já está em dia, nada a coletar

    return inicio_dt.strftime("%d/%m/%Y"), hoje_dt.strftime("%d/%m/%Y")


def obter_datas():
    """Determina o intervalo de datas a coletar.

    Prioridade:
      1. --ini/--fim informados explicitamente na linha de comando.
      2. --auto: calcula automaticamente do último dia salvo até hoje —
         cobre lacunas sozinho se a coleta ficou dias sem rodar. Este é o
         modo recomendado para a automação diária (Agendador de Tarefas).
      3. --hoje: usa apenas a data atual como início e fim.
      4. Nenhum argumento: pergunta interativamente no terminal, como
         sempre funcionou.
    """
    parser = argparse.ArgumentParser(
        description="Scraper da busca de ciência da Folha de S.Paulo."
    )
    parser.add_argument("--ini", help="Data inicial no formato DD/MM/AAAA")
    parser.add_argument("--fim", help="Data final no formato DD/MM/AAAA")
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Coleta do dia seguinte ao último salvo em folha.csv até hoje "
             "(cobre gaps automaticamente; recomendado para automação diária)."
    )
    parser.add_argument(
        "--hoje",
        action="store_true",
        help="Coleta apenas o dia de hoje."
    )
    args = parser.parse_args()

    if args.auto:
        ini_auto, fim_auto = determinar_intervalo_auto()

        if ini_auto is None:
            print("Já está tudo atualizado — nenhuma coleta necessária hoje.")
            sys.exit(0)

        if ini_auto == fim_auto:
            print(f"Modo automático: coletando o dia {ini_auto}.")
        else:
            print(f"Modo automático: coletando de {ini_auto} a {fim_auto} "
                  "(cobrindo dias que ficaram pendentes desde a última execução).")

        return ini_auto, fim_auto

    if args.hoje:
        hoje = datetime.now().strftime("%d/%m/%Y")
        print(f"Coletando apenas o dia de hoje ({hoje}).")
        return hoje, hoje

    if args.ini and args.fim:
        print(f"Datas informadas via linha de comando: {args.ini} a {args.fim}.")
        return args.ini, args.fim

    ini_date = input("Initial date (DD/MM/YYYY): ")
    fin_date = input("Final date (DD/MM/YYYY): ")
    return ini_date, fin_date


ini_date, fin_date = obter_datas()

ini = ini_date.split("/")
fin = fin_date.split("/")

url = (
    "https://search.folha.uol.com.br/search?q=a*&periodo=personalizado&sd="
    + ini[0]
    + "%2F"
    + ini[1]
    + "%2F"
    + ini[2]
    + "&ed="
    + fin[0]
    + "%2F"
    + fin[1]
    + "%2F"
    + fin[2]
    + "&site=sitefolha&site%5B%5D=online%2Fciencia&sort=asc"
)


# =========================
# 2. Abre o navegador
# =========================
options = webdriver.ChromeOptions()

# Selenium Manager: dispensa informar manualmente o caminho do chromedriver.
driver = webdriver.Chrome(options=options)

driver.get(url)
driver.maximize_window()

print("Título da página:", driver.title)


# =========================
# 3. Coleta URLs nas páginas de busca
# =========================
# A busca da Folha apresenta um erro ao paginar por seta/elemento oculto.
# Por isso, o código clica diretamente nos números das páginas.
data_dict = {"section": [], "url": []}


def capturar_pagina_atual():
    blocks = driver.find_elements(By.ID, "view-view")
    links = driver.find_elements(By.CSS_SELECTOR, "div.c-headline__content > a")

    for block, link in zip(blocks, links):
        text = block.text
        extract = text.strip().split("\n")
        data_dict["section"].append(extract[0])
        data_dict["url"].append(link.get_attribute("href"))


# Captura o conteúdo já carregado ANTES de qualquer clique. Necessário porque
# quando os resultados cabem numa página só (comum em coletas de 1 dia), o
# widget de paginação não mostra nenhum link numerado clicável — sem essa
# captura inicial, a coleta ficaria vazia mesmo com matérias na tela.
try:
    WebDriverWait(driver, 8).until(
        EC.presence_of_element_located((By.ID, "view-view"))
    )
    capturar_pagina_atual()
except Exception:
    pass  # nenhum resultado nesse intervalo

pages_list = [str(num) for num in range(10000)]
c = 1

while True:
    try:
        element = WebDriverWait(driver, 20).until(
            EC.element_to_be_clickable((By.LINK_TEXT, pages_list[c]))
        )
        driver.execute_script("arguments[0].click();", element)

        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.ID, "view-view"))
        )
        capturar_pagina_atual()

        print(f"Página {c} coletada")
        c += 1

    except Exception:
        break

# Fecha o navegador depois de coletar os links.
driver.quit()


df_folha = pd.DataFrame(data_dict)

# Protege contra dupla contagem, caso o primeiro clique em página "1"
# recapture o mesmo conteúdo já lido antes do clique.
if not df_folha.empty:
    df_folha = df_folha.drop_duplicates(subset="url").reset_index(drop=True)

print(f"\nTotal de links coletados: {len(df_folha)}")


# =========================
# 4. Extrai o conteúdo das matérias
# =========================
urls = df_folha["url"]
texts = []

for url in urls:
    try:
        url = url.rstrip("\n")
        story = {}
        article = NewsPlease.from_url(url)

        story["date"] = article.date_publish
        story["title"] = article.title
        story["text"] = article.maintext
        story["description"] = article.description
        story["author"] = article.authors
        story["image_url"] = article.image_url

        texts.append(story)
        print(article.title)

    except Exception:
        texts.append(
            {
                "date": "ERRO!!!",
                "title": "ERRO!!!",
                "text": "ERRO!!!",
                "description": "ERRO!!!",
                "author": "ERRO!!!",
                "image_url": "ERRO!!!",
            }
        )
        print("Não foi possível extrair")


y = pd.DataFrame(texts)
df_folha = pd.concat([df_folha, y], axis=1)


# =========================
# 5. Padroniza campos
# =========================
if not df_folha.empty:
    df_folha["section"] = (
        df_folha["section"]
        .str.replace("FOLHA DE S.PAULO - ", "", regex=False)
        .str.lower()
    )

    # A conversão de data pode falhar se houver matérias com erro.
    df_folha["date"] = pd.to_datetime(df_folha["date"], errors="coerce")
    df_folha["date"] = df_folha["date"].dt.strftime("%d/%m/%Y")


# =========================
# 6. Salva resultado (acrescentando ao histórico existente)
# =========================
# Diferente da versão original (que sobrescrevia o arquivo inteiro a cada
# execução), aqui o resultado desta coleta é somado ao que já existe em
# folha.csv — essencial para uso diário, onde cada execução deve ACRESCENTAR
# o dia novo, não substituir todo o histórico já coletado.
if os.path.exists(ARQUIVO_SAIDA):
    df_existente = pd.read_csv(ARQUIVO_SAIDA, encoding="utf-8-sig")
    df_combinado = pd.concat([df_existente, df_folha], ignore_index=True)

    if "url" in df_combinado.columns:
        antes = len(df_combinado)
        df_combinado = df_combinado.drop_duplicates(subset="url", keep="first")
        duplicadas = antes - len(df_combinado)
        if duplicadas > 0:
            print(f"{duplicadas} matéria(s) duplicada(s) (já existente) ignorada(s).")
else:
    df_combinado = df_folha

df_combinado.to_csv(ARQUIVO_SAIDA, index=False, encoding="utf-8-sig")

print(f"\nArquivo salvo: {ARQUIVO_SAIDA} ({len(df_combinado)} matéria(s) no total).")
print("Concluído.")