"""
Script pontual: coleta somente o AUTOR de cada matéria do G1 já presente
no backup (g1_globo_backup_v2.csv, coletado com o texto antigo via
news-please) e grava um CSV novo com todo o resto (texto, data, título,
imagem etc.) mantido como estava, só a coluna author corrigida.

Bem mais rápido que recoletar tudo do zero, porque:
- não precisa paginar pela listagem de novo (as URLs já são conhecidas)
- não precisa esperar o corpo inteiro da matéria renderizar -- só a
  tag de autor (schema.org), que aparece bem mais cedo no carregamento
  da página

Uso:
    python corrigir_autores_g1.py

Se for interrompido no meio (Ctrl+C, queda de conexão etc.), rode o
mesmo comando de novo -- ele retoma a partir de onde parou.
"""

import os
import sys
import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup

ARQUIVO_ENTRADA = "g1_globo_backup_v2.csv"
ARQUIVO_SAIDA = "g1_globo.csv"
ARQUIVO_PROGRESSO = "corrigir_autores_g1_progresso.txt"
SALVAR_A_CADA = 20  # grava o CSV a cada N matérias, não a cada uma (mais rápido)

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def carregar_progresso():
    if os.path.exists(ARQUIVO_PROGRESSO):
        with open(ARQUIVO_PROGRESSO, encoding="utf-8") as f:
            return int(f.read().strip())
    return 0


def salvar_progresso(indice):
    with open(ARQUIVO_PROGRESSO, "w", encoding="utf-8") as f:
        f.write(str(indice))


def criar_driver():
    """Cria uma sessão do Chrome configurada para ser bem mais rápida
    neste caso de uso específico: só precisamos da tag de autor
    (schema.org), que já está pronta no HTML bem antes da página
    terminar de carregar por completo. Por padrão, driver.get() só
    retorna depois que TUDO carrega -- anúncios, player de vídeo,
    scripts de rastreamento (vimos vários no HTML do G1). A estratégia
    'eager' faz o Selenium devolver o controle assim que o DOM estiver
    pronto, sem esperar esses recursos pesados. Desativar imagens
    reduz ainda mais o tempo de carregamento, já que não precisamos
    delas para nada aqui."""
    options = webdriver.ChromeOptions()
    options.page_load_strategy = "eager"
    options.add_experimental_option(
        "prefs", {"profile.managed_default_content_settings.images": 2}
    )
    return webdriver.Chrome(options=options)


def reiniciar_driver(driver_antigo):
    try:
        driver_antigo.quit()
    except Exception:
        pass
    print("    Reiniciando sessão do navegador...")
    return criar_driver()


def extrair_autor(driver, url, tentativas=3):
    """Carrega a matéria e extrai só o autor via schema.org. Tenta de
    novo se a página não carregar (rede) ou se a tag de autor não
    aparecer a tempo (pode ser matéria sem autor mesmo, ou carregamento
    lento -- por isso tenta mais de uma vez antes de aceitar vazio)."""
    for tentativa in range(1, tentativas + 1):
        try:
            driver.get(url)
        except Exception as e:
            if tentativa == tentativas:
                raise
            print(f"    Falha ao carregar (tentativa {tentativa}/{tentativas}): {e}")
            time.sleep(3)
            continue

        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, "span[itemprop='author'] meta[itemprop='name']")
                )
            )
        except Exception:
            pass  # segue mesmo assim -- pode ser matéria sem autor

        soup = BeautifulSoup(driver.page_source, "lxml")
        autor_meta = soup.select_one("span[itemprop='author'] meta[itemprop='name']")
        if autor_meta and autor_meta.get("content"):
            return autor_meta["content"].strip()

        if tentativa < tentativas:
            time.sleep(2)

    return ""  # não encontrou autor após todas as tentativas -- provavelmente não tem


def main():
    if not os.path.exists(ARQUIVO_ENTRADA):
        print(f"Arquivo de entrada não encontrado: {ARQUIVO_ENTRADA}")
        return

    df = pd.read_csv(ARQUIVO_ENTRADA, encoding="utf-8-sig")

    if "author" not in df.columns:
        df["author"] = ""
    # Se a coluna vier vazia/NaN, o pandas infere tipo float64 e quebra
    # ao tentar escrever uma string nela mais adiante -- força para
    # object (texto) desde já, independente do que veio no CSV.
    df["author"] = df["author"].astype(object)

    total = len(df)
    inicio = carregar_progresso()

    if inicio >= total:
        print("Já concluído (checkpoint aponta para o fim do arquivo).")
        return

    if inicio > 0:
        print(f"Retomando a partir do índice {inicio} (de {total})")
        df_ja_processado = pd.read_csv(ARQUIVO_SAIDA, encoding="utf-8-sig")
        df.loc[:inicio - 1, "author"] = df_ja_processado.loc[:inicio - 1, "author"]

    driver = criar_driver()

    try:
        for i in range(inicio, total):
            url = df.at[i, "url"]
            try:
                autor = extrair_autor(driver, url)
                df.at[i, "author"] = autor
                print(f"  [{i + 1}/{total}] {autor or '(sem autor encontrado)'} — {url}")

            except Exception as e:
                print(f"  [{i + 1}/{total}] Falha de conexão, reiniciando navegador: {e}")
                driver = reiniciar_driver(driver)
                try:
                    autor = extrair_autor(driver, url)
                    df.at[i, "author"] = autor
                    print(f"  [{i + 1}/{total}] {autor or '(sem autor encontrado)'} (após reiniciar)")
                except Exception as e2:
                    print(f"  [{i + 1}/{total}] ✗ Erro definitivo em {url}: {e2}")

            if (i + 1) % SALVAR_A_CADA == 0:
                df.to_csv(ARQUIVO_SAIDA, index=False, encoding="utf-8-sig")
                salvar_progresso(i + 1)

    finally:
        df.to_csv(ARQUIVO_SAIDA, index=False, encoding="utf-8-sig")
        salvar_progresso(min(i + 1, total))
        driver.quit()

    print(f"\nConcluído: {ARQUIVO_SAIDA} atualizado com {total} matéria(s).")


if __name__ == "__main__":
    main()