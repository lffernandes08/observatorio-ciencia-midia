@echo off
REM ============================================================
REM Pipeline diario do Observatorio de Ciencia na Midia
REM Roda: coleta do dia -> extracao de keywords -> analise de IA do dia
REM Pensado para ser chamado pelo Agendador de Tarefas do Windows.
REM ============================================================

REM A pasta do projeto é detectada automaticamente a partir de onde este
REM arquivo .bat está salvo (evita problemas de codificação com acentos
REM em caminhos digitados manualmente, como "CÓDIGOS").
cd /d "%~dp0"

REM Arquivo de log com carimbo de data/hora, para conferir depois se rodou.
REM Usa PowerShell para obter a data no formato AAAA-MM-DD, independente da
REM configuração regional do Windows (o %date% do cmd varia de máquina para
REM máquina — em português, por exemplo, pode vir com o dia da semana na
REM frente, o que quebraria uma extração por posição de caractere).
if not exist logs mkdir logs

for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyy-MM-dd"') do set HOJE=%%i
set LOGFILE=logs\pipeline_%HOJE%.log

echo ============================================== >> %LOGFILE%
echo Execucao iniciada em %date% %time% >> %LOGFILE%
echo ============================================== >> %LOGFILE%

echo [1/5] Coletando materias da Folha... >> %LOGFILE%
python main.py --auto >> %LOGFILE% 2>&1

echo [2/5] Coletando materias da CNN Brasil... >> %LOGFILE%
python cnn_brasil.py --auto >> %LOGFILE% 2>&1

echo [3/5] Coletando materias da BBC News Brasil... >> %LOGFILE%
python bbc_brasil.py --auto >> %LOGFILE% 2>&1

echo [4/5] Extraindo palavras-chave... >> %LOGFILE%
python extrair_keywords.py >> %LOGFILE% 2>&1

echo [5/5] Gerando analise diaria de IA... >> %LOGFILE%
python analise_diaria.py >> %LOGFILE% 2>&1

echo Pipeline concluido em %date% %time% >> %LOGFILE%
echo. >> %LOGFILE%