# Responsabilização Eleitoral em Tempos de Crise: Os Limites do Voto Retrospectivo na Gestão Municipal da Pandemia de COVID-19

**Autor:** Rafael Azevedo Perich

## Descrição do Projeto

Esta tese investiga se e como a gestão municipal da COVID-19 afetou a reeleição de prefeitos nas eleições de 2020 no Brasil. O argumento central é que crises sistêmicas produzem uma "névoa da crise" que recalibra a responsabilização: sinais de desempenho tornam-se ruidosos e de difícil atribuição, levando eleitores a recorrerem a heurísticas políticas relativamente estáveis, como a força política prévia do incumbente e a estrutura da competição eleitoral.

Este repositório contém os dados e os scripts de análise utilizados para produzir os resultados apresentados na tese.

## Conteúdo do Repositório

* `Base_VALIDADA_E_PRONTA.csv`: A base de dados consolidada com N=139 municípios, contendo variáveis eleitorais, socioeconômicas e de gestão da pandemia.
* `tese_master.py`: O script em Python utilizado para realizar todo o pré-processamento, engenharia de variáveis e análise estatística (descritiva e regressões).

## Dicionário de Variáveis (Principais)

Abaixo estão descritas as principais variáveis utilizadas nos modelos de regressão. Para os nomes exatos das colunas no script, consulte o arquivo `tese_master.py`.

| Variável | Descrição |
| :--- | :--- |
| **reeleito** | Variável dependente (binária). `1` se o prefeito incumbente foi reeleito em 2020, `0` caso contrário. |
| **delta_vantagem**| Variável dependente (contínua). Mede a variação na vantagem eleitoral do incumbente (em pontos percentuais) entre 2016 e 2020. |
| **vantagem_2016**| Controle político. Vantagem eleitoral (em p.p.) que o incumbente obteve na eleição de 2016. |
| **nec_2020** | Controle político. Número Efetivo de Candidatos na eleição de 2020. Mede o nível de competição. |
| **npi_mascara_norm** | Variável de gestão (H1). Medida normalizada (0 a 1) da precocidade na adoção da obrigatoriedade do uso de máscaras. |
| **infra_hosp_campanha** | Variável de gestão (H1, binária). `1` se o município instalou um hospital de campanha, `0` caso contrário. |
| **stress_sobrecarga_uti**| Variável de severidade (H2, binária). `1` se o município reportou sobrecarga de leitos/UTI, `0` caso contrário. |
| **delta_ifdm_emprego** | Variável de severidade (H2). Variação no indicador de Emprego & Renda do IFDM/FIRJAN entre 2016 e 2020. |
| **ln_pop** | Controle estrutural. Logaritmo natural da população do município. |
| **pib_pc** | Controle estrutural. Produto Interno Bruto per capita do município. |

## Como Replicar a Análise

1.  **Requisitos:** É necessário ter o Python 3 instalado, juntamente com as bibliotecas listadas no início do script `tese_master.py` (principalmente `pandas`, `statsmodels` e `scikit-learn`).
2.  **Execução:** Coloque o script `tese_master.py` e o banco de dados `Base_VALIDADA_E_PRONTA.csv` na mesma pasta. Execute o script a partir do seu terminal com o comando: `python tese_master.py`.
3.  **Resultados:** Os resultados das análises de regressão serão impressos diretamente no console e na pasta que os arquivos estão localizados.

## Como Citar

Se utilizar estes dados ou a análise em sua pesquisa, por favor, cite a tese da seguinte forma:

PERICH, Rafael Azevedo. *Responsabilização Eleitoral em Tempos de Crise: Os Limites do Voto Retrospectivo na Gestão Municipal da Pandemia de COVID-19 a partir de Casos Brasileiros*. Tese (Doutorado em Ciência Política) – Universidade Federal do Paraná, Curitiba, 2025.
