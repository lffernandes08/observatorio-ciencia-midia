import pandas as pd

df = pd.read_csv("bbc_brasil.csv", encoding="utf-8-sig")
datas = pd.to_datetime(df["date"], format="%d/%m/%Y")

print(f"Total de matérias: {len(df)}")
print(f"Período: {datas.min().strftime('%d/%m/%Y')} a {datas.max().strftime('%d/%m/%Y')}")

# Checagem de qualidade: nenhum texto deveria começar com resquício de metadado
suspeitos = df[df["text"].astype(str).str.contains(
    r"^(Author,|Tempo de leitura)", regex=True, na=False
)]
print(f"\nMatérias com possível vazamento de metadado no início do texto: {len(suspeitos)}")
if len(suspeitos) > 0:
    print(suspeitos[["date", "title"]].head(10).to_string(index=False))