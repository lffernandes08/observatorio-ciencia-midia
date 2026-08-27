# Observatório de Ciência na Mídia — contexto do projeto

Dashboard Streamlit que acompanha a cobertura de ciência na mídia brasileira
(Folha, CNN Brasil, BBC News Brasil, G1), combinando estatística, PLN e
classificação por IA. Projeto pessoal de Luiz, da Secom/Jornal UFG.

Deploy: Streamlit Community Cloud, conectado ao branch `main` do GitHub —
qualquer `git push` reimplanta o app automaticamente (30-60s), sem passo
manual. Local: Windows, Task Scheduler roda `pipeline_diario.bat` às 22h.

## Stack

- Python + Selenium (scrapers) + BeautifulSoup (parsing) + pandas
- OpenAI API para classificação por IA (6 dimensões: palavras-chave, frame,
  área, abrangência, instituições, pessoas)
- Streamlit (dashboard) + Plotly (gráficos) + Cytoscape.js (grafo de rede,
  via `components.html`)

## Estrutura de arquivos

- **Scrapers**: `main.py` (Folha), `cnn_brasil.py`, `bbc_brasil.py`,
  `g1_globo.py`
- **Pipeline de IA**: `extrair_keywords.py` (classificação por matéria),
  `analise_diaria.py` (tendência do dia por dia×veículo)
- **Manutenção** (scripts pontuais, não fazem parte do pipeline regular):
  `corrigir_autores_bbc.py`, `corrigir_autores_g1.py`, `corrigir_datas.py`,
  `invalidar_dias.py`, `reclassificar_areas.py`
- **Diagnóstico**: `contar_materias_por_dia.py`, `verificar_gaps.py`,
  `verificar_datas_suspeitas.py`, `verificar_resultado_bbc.py`,
  `testar_reextracao_bbc.py`
- **App**: `app.py`
- **Dados**: `folha.csv`, `cnn_brasil.csv`, `bbc_brasil.csv`, `g1_globo.csv`,
  `materias_keywords.csv`, `analise_diaria.jsonl`
- **Orquestração**: `pipeline_diario.bat` (6 passos: os 4 scrapers em modo
  `--auto`, depois `extrair_keywords.py`, depois `analise_diaria.py`)

## Convenções estabelecidas

- Nomes de variáveis/funções em português, comentários explicando o
  **porquê** de decisões não óbvias (não só o quê).
- **Retry de conexão + reinício de driver** em toda coleta longa: Chrome
  trava/morre em execuções de centenas de páginas. Padrão:
  `carregar_pagina_com_retry()` pra timeouts pontuais,
  `FalhaConexaoDriver` + `reiniciar_driver()` quando a sessão morre de vez
  (reinicia o Chrome do zero e tenta a mesma matéria de novo, sem perder o
  restante da fila).
- **Checkpoints incrementais** (`*_progresso.txt`) salvos durante a
  execução — qualquer script longo deve poder ser interrompido e retomado
  sem reprocessar do zero.
- **Data-limite sempre normalizada pra meia-noite** antes de comparar —
  `datetime.now()` sozinho vaza a hora da execução pro cálculo, causando
  exclusão incorreta de matérias no dia-limite exato (bug já corrigido em
  BBC/CNN/G1, mas o padrão vale pra qualquer código novo de data).
- **Validação pós-extração com retry** antes de aceitar conteúdo raspado
  (`_texto_parece_valido()` ou equivalente) — melhor tentar de novo do que
  salvar silenciosamente um texto incompleto/contaminado.
- **CSVs com colunas de texto**: forçar `.astype(object)` depois de
  `pd.read_csv()` se a coluna pode vir vazia — pandas infere `float64`/NaN
  em colunas totalmente vazias, e quebra ao tentar escrever string nelas
  depois.
- **Deploy**: só um `git push` no `main`; nunca precisa mexer no painel do
  Streamlit Cloud a menos que o app trave (aí sim, botão "Reboot app").

## Decisões de design do app.py

- **Tema único escuro** (sem modo claro) — paleta grafite levemente
  esverdeada: `ink #151817`, `ink2 #1D211F`, `paper #E8E4D8`, accent teal
  `#5FBFA0`, amber `#E8A33D`. Definida em `.streamlit/config.toml` (uma
  única tabela `[theme]`) e espelhada em `obter_cores_tema()` no `app.py`.
- `obter_cores_tema()` existe porque os componentes renderizados via
  `components.html()` (gauge/velocímetro, sismógrafo, rede semântica)
  rodam em **iframe isolado** — não herdam `var(--text-color)` etc. do
  Streamlit, então o Python precisa resolver e injetar a cor certa direto
  no HTML gerado.
- **Navegação**: botões horizontais no topo do conteúdo (não sidebar, não
  `st.tabs()`). Decisão deliberada: `st.tabs()` executa o código de
  **todas** as abas a cada rerun (só esconde visualmente as inativas) —
  penalizaria a performance de um app com seções pesadas. Implementado com
  `st.button()` + `st.session_state`, replicando o comportamento de só
  rodar a seção ativa que o antigo `st.sidebar.radio()` tinha.
- **"Cobertura do dia"** é a seção padrão/primeira (era "Análise IA") — é
  o que o usuário vê ao carregar a página, sem precisar navegar.
- **Emojis removidos** quase por completo (pedido explícito: "dão um ar
  amador"). As poucas exceções funcionais (indicador de frescor dos
  dados, aviso de amostra pequena) viraram indicador CSS (bolinha
  colorida com glow), não emoji.
- **Animações discretas**: fade-in com leve subida (cards/blocos), entrada
  escalonada por coluna (métricas), fade suave (gráficos Plotly),
  crescimento de barra (rankings, via variável CSS `--largura-final`
  animada de 0 até o valor final). Sempre curtas (~0.4-0.5s), `ease-out`,
  nunca bounce/elástico. Respeitam `prefers-reduced-motion`.

## Lições aprendidas (não repetir)

- **Chrome trava em coletas longas** (centenas de páginas em sequência) —
  sempre ter retry + reinício de driver, nunca assumir que uma sessão
  Selenium aguenta o histórico inteiro sem intervenção.
- **BBC vazava metadado** (autor/"tempo de leitura") dentro do corpo do
  texto em ~35% das matérias — causa raiz: condição de espera do Selenium
  satisfeita cedo demais (por um `<p>` de legenda de foto, não o corpo
  real). Corrigido com seletor mais específico
  (`main div[dir='ltr'] > p`) + validação pós-extração.
- **G1 usava `news-please`**, que não extraía o autor corretamente nessa
  estrutura — trocado por extração via dados estruturados schema.org
  (`span[itemprop='author'] > meta[itemprop='name']`), mais confiável.
- **Bug de fronteira de data**: `hoje_dt = datetime.now()` (sem normalizar
  a hora) fazia a comparação `>=` excluir matérias do próprio dia-limite —
  afetava BBC, CNN e G1 igualmente (mesmo padrão de código copiado entre
  os três).
- **`st.context.theme.type`** existe desde o Streamlit 1.46 — usar com
  fallback (`try/except`), pode não resolver certo no primeiro render de
  uma sessão com tema customizado.
- **`key=` de widget vira classe CSS `st-key-<chave>`** — não pode ter
  espaço/acento na chave (quebra a classe). Usar slug quando o rótulo
  visível tem espaço/acento.
- **`st.tabs()` roda o código de todas as abas a cada rerun** — evitar em
  apps com seções computacionalmente pesadas; preferir botões +
  `session_state` quando só uma seção deve executar por vez.
- Ao editar `.py` com `str_replace`, sempre revalidar com `py_compile` e,
  quando possível, testar a lógica isolada (extrair função via `ast`,
  rodar com stubs) antes de considerar a mudança concluída — vários bugs
  nesta sessão só apareceram em testes isolados, não na leitura do código.

## .gitignore

Não versionar: `*_progresso.txt`, `lotes_concluidos.txt`, `logs/`, `lib/`
(artefato gerado automaticamente pelo pyvis pro grafo de rede),
`*_backup*.csv`, `*_backup*.jsonl`.

## Pendências conhecidas

- Decisão consciente (não é bug): a classificação por IA das matérias da
  BBC **não foi reprocessada** depois da reextração de texto que corrigiu
  o vazamento de metadado — impacto avaliado como pequeno demais pra
  justificar o custo de API. Se um dia isso incomodar, o padrão é um
  script pontual tipo `reclassificar_areas.py`, filtrando só as URLs da
  BBC.
