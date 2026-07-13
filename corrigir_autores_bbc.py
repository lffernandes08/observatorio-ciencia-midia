"""
Corrige o campo 'author' já salvo em bbc_brasil.csv, removendo fragmentos
de URL/domínio que o news-please capturou junto com o nome do autor de
verdade em algumas matérias (ex: "['André Biernath', 'www.facebook.com']"
vira "['André Biernath']").

Isso corrige dados JÁ coletados — a correção equivalente no scraper
(bbc_brasil.py) evita que o problema aconteça de novo em coletas futuras,
mas não conserta sozinha o que já está no CSV.

Uso:
    python corrigir_autores_bbc.py
"""

import re
import ast
import pandas as pd

ARQUIVO = "bbc_brasil.csv"

PADRAO_URL_DOMINIO = re.compile(r'(https?://|www\.|\.com\b|\.com\.br\b|\.org\b|\.net\b)', re.IGNORECASE)


def limpar_valor_autor(valor):
    """Recebe o valor bruto da célula 'author' (pode ser string de lista
    Python, ex: "['André Biernath', 'www.facebook.com']", ou um nome
    solto) e retorna a versão limpa, sem itens que pareçam URL/domínio."""
    texto = str(valor).strip()

    if texto.startswith("[") and texto.endswith("]"):
        try:
            lista = ast.literal_eval(texto)
            if not isinstance(lista, list):
                lista = [texto]
        except (ValueError, SyntaxError):
            lista = [texto]
    else:
        lista = [texto]

    autores_limpos = [
        str(item).strip() for item in lista
        if item and not PADRAO_URL_DOMINIO.search(str(item))
    ]

    # Se filtrou tudo (só havia lixo, sem nome real), mantém o valor
    # original — melhor preservar o dado bruto do que apagar silenciosamente.
    if not autores_limpos:
        return valor

    if len(autores_limpos) == 1:
        return autores_limpos[0]

    return str(autores_limpos)


def main():
    df = pd.read_csv(ARQUIVO, encoding="utf-8-sig")

    if "author" not in df.columns:
        print(f"Coluna 'author' não encontrada em {ARQUIVO}. Nada a fazer.")
        return

    valores_antes = df["author"].astype(str).tolist()
    df["author"] = df["author"].apply(limpar_valor_autor)
    valores_depois = df["author"].astype(str).tolist()

    alteracoes = [
        (antes, depois)
        for antes, depois in zip(valores_antes, valores_depois)
        if antes != depois
    ]

    if not alteracoes:
        print("Nenhuma linha precisou de correção — o campo 'author' já está limpo.")
        return

    df.to_csv(ARQUIVO, index=False, encoding="utf-8-sig")

    print(f"{len(alteracoes)} linha(s) corrigida(s). Exemplos:")
    for antes, depois in alteracoes[:5]:
        print(f"  {antes!r} -> {depois!r}")

    print(f"\nArquivo salvo: {ARQUIVO}")


if __name__ == "__main__":
    main()