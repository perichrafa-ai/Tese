Análise de Dados: Impacto da Gestão da Pandemia nas Eleições Municipais de 2020

Este repositório contém um conjunto de scripts em Python para analisar o impacto da gestão da pandemia de COVID-19 nos resultados das reeleições municipais de 2020 no Brasil. O projeto foi estruturado em módulos independentes, cada um focado em uma etapa específica da análise, desde a estatística descritiva até a estimação de modelos de regressão complexos com interações.

Principais Características
Pipeline Automatizado: Cada script é uma ferramenta "tudo-em-um" que executa uma análise completa com um único comando.

Reprodutibilidade Garantida: Os scripts gerenciam automaticamente seu próprio ambiente virtual (.venv) e instalam as dependências exatas listadas no requirements.txt. Não é necessária nenhuma configuração manual.

Detecção Inteligente de Colunas: Os scripts são robustos a pequenas variações nos nomes das colunas do seu arquivo de dados, tentando encontrar as variáveis necessárias por meio de palavras-chave.

Saídas Organizadas: Todos os resultados (tabelas em .csv e .xlsx, e gráficos em .png) são salvos automaticamente em um diretório de saída (./output/).

Pré-requisitos
Python 3.7 ou superior instalado em seu sistema.

O arquivo de dados Base_VALIDADA_E_PRONTA.csv localizado no mesmo diretório dos scripts.

Como Executar as Análises
Não é preciso ativar o ambiente virtual ou instalar pacotes manualmente. Basta executar o script desejado a partir do seu terminal, e ele cuidará de todo o processo de configuração antes de iniciar a análise.

Parte 1: Análise Descritiva e Modelos Principais
Este é o script principal da análise. Ele gera um panorama completo dos dados, incluindo estatísticas descritivas, distribuições de frequência, e executa os modelos de regressão centrais da pesquisa (Logit e OLS), além de um diagnóstico de multicolinearidade (VIF).

Script: master parte 1.py

Para executar:

Windows (PowerShell / CMD)

py "master parte 1.py" --csv "./Base_VALIDADA_E_PRONTA.csv" --outdir "./output"

macOS / Linux

python3 "master parte 1.py" --csv "./Base_VALIDADA_E_PRONTA.csv" --outdir "./output"

Principais Saídas Geradas:

Relatórios Descritivos: Tabelas e gráficos de frequência para todas as variáveis.

Tabela_3_0_Diagnostico_Multicolinearidade_VIF.csv: Teste de VIF para as variáveis independentes.

Tabela_3_1_Determinantes_Reeleicao_Logit.csv: Resultados dos modelos Logit para a probabilidade de reeleição.

Tabela_3_2_Determinantes_DeltaVantagem_OLS.csv: Resultados dos modelos OLS para a variação na vantagem eleitoral.

fig_3_7_coeficientes_ols_completo.png: Gráfico de coeficientes (coefplot) do modelo OLS completo.

Parte 2: Análise de Interações (Gestão vs. Abstenção e Vantagem vs. Competição)
Este script foca em testar o efeito de moderação de duas interações específicas:

Como o aumento da abstenção (ΔAbstenção) modera o efeito do Índice de Gestão na probabilidade de reeleição.

Como a vantagem eleitoral de 2016 (Vantagem2016) modera o efeito da variação na competição (ΔNEC) sobre a performance eleitoral em 2020.

Script: master parte 2.py

Para executar:

Windows (PowerShell / CMD)

py "master parte 2.py" --csv "./Base_VALIDADA_E_PRONTA.csv" --outdir "./output"

macOS / Linux

python3 "master parte 2.py" --csv "./Base_VALIDADA_E_PRONTA.csv" --outdir "./output"

Principais Saídas Geradas:

Tabela_3_4a_logit_gestao_x_deltaabst.csv: Coeficientes do modelo Logit com interação.

Tabela_3_4b_marginais_logit_gestao_x_deltaabst.csv: Efeitos marginais médios (AMEs) do modelo.

Tabela_3_4c_ols_v2016_x_dnec_log.csv: Coeficientes do modelo OLS com interação.

Tabela_3_4d_robustez_competicao.csv: Testes de robustez para o modelo OLS.

fig_pred_logit_gestao_x_deltaabst.png: Gráfico do efeito predito da interação Gestão × ΔAbstenção.

fig_pred_ols_v2016_x_dnec_log.png: Gráfico do efeito predito da interação Vantagem × ΔCompetição.

Parte 3: Teste das Hipóteses H3 e H4 (Moderação por Porte e Vantagem Prévia)
Este script foi desenvolvido para testar as hipóteses H3 e H4, que investigam como os efeitos da Gestão e da Severidade da crise são moderados por:

H3: O porte do município (medido pelo ln(População)).

H4: A vantagem eleitoral do incumbente em 2016 (Vantagem2016).

Script: master parte 3.py

Para executar:

Windows (PowerShell / CMD)

py "master parte 3.py" --csv "./Base_VALIDADA_E_PRONTA.csv" --outdir "./output"

macOS / Linux

python3 "master parte 3.py" --csv "./Base_VALIDADA_E_PRONTA.csv" --outdir "./output"

Principais Saídas Geradas:

Sumários de regressão (.txt) e tabelas de coeficientes (.csv) para cada hipótese e variável dependente (Logit e OLS).

Gráficos de predição para visualizar os efeitos de moderação, como H3_logit_lnpop.png e H4_logit_v2016.png.

H3H4_relatorio_componentes.txt: Um relatório de transparência que lista quais colunas do CSV foram usadas para construir os índices de Gestão e Severidade.

Customização e Solução de Problemas
Parâmetros do CSV
Se seu arquivo CSV utiliza um formato diferente do padrão brasileiro, você pode ajustar a leitura com os seguintes parâmetros:

--encoding: Codificação do arquivo (padrão: latin-1).

--sep: Separador de colunas (padrão: ;).

--decimal: Separador decimal (padrão: ,).

Mapeamento Manual de Colunas
Se um script não conseguir encontrar uma coluna essencial, você pode especificá-la manualmente com o parâmetro --map. Este argumento pode ser usado várias vezes.

Exemplo:

py "master parte 1.py" --csv "dados.csv" --map Reeleicao="Reeleito (S/N)" --map LnPop="log_pop_2019"

Solução de Problemas
Caso encontre algum erro inesperado durante a execução, a solução mais simples geralmente é deletar a pasta .venv que foi criada no diretório. Ao rodar o script novamente, ele irá recriar o ambiente virtual e reinstalar as dependências do zero, resolvendo a maioria dos problemas.
