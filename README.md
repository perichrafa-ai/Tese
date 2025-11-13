# Responsabilização Eleitoral em Tempos de Crise (Tese de Doutorado, 2025)

**Autor:** Rafael Azevedo Perich
**Orientador:** Prof. Dr. Adriano Nervo Codato
**Instituição:** Universidade Federal do Paraná (UFPR)

Este repositório contém os dados e scripts de análise da tese de doutorado "RESPONSABILIZAÇÃO ELEITORAL EM TEMPOS DE CRISE: OS LIMITES DO VOTO RETROSPECTIVO NA GESTÃO MUNICIPAL DA PANDEMIA DE COVID-19 A PARTIR DE CASOS BRASILEIROS".

## Resumo

Esta tese investiga se e como a gestão municipal da COVID-19 afetou a reeleição de prefeitos nas eleições de 2020 no Brasil. Com base em uma base original de 139 municípios e em modelos logísticos e lineares, os resultados indicam ausência de efeito eleitoral tanto das políticas sanitárias implementadas (por exemplo, uso de máscaras, restrição ao comércio, hospitais de campanha) quanto da severidade local da pandemia. Em contraste, fatores políticos pré-existentes (notadamente a vantagem eleitoral acumulada pelo incumbente e a estrutura de competição) aparecem como os principais organizadores do resultado.

## Argumento Central: A "Névoa da Crise"

O argumento central da tese é que a pandemia produziu uma **"névoa da crise"**: um ambiente informacional ruidoso, politizado e de alta incerteza.

Neste cenário, os mecanismos tradicionais de *accountability* falharam:

1.  **Avaliação de Desempenho Fraca:** A gestão da crise (ações de saúde, NPIs) e a severidade (estresse hospitalar) **não tiveram efeito eleitoral** estatisticamente significativo, em média.

2.  **Recurso a Heurísticas Políticas:** Sem um sinal claro de desempenho, os eleitores recorreram a heurísticas mais estáveis e pré-existentes.

3.  **Fatores Políticos Prevaleceram:** Os principais preditores da reeleição e da variação da vantagem eleitoral foram a **vantagem acumulada do incumbente (Vantagem 2016)** e, principalmente, a **estrutura da competição eleitoral (NEC 2020)**.

Uma descoberta-chave de heterogeneidade (H3) é que em **municípios maiores** (com maior `ln_pop`), a severidade da crise *esteve*, de facto, associada a uma **punição eleitoral** mais intensa ao incumbente.

## Estrutura do Repositório

* `Perich, Rafael - Tese.docx`: O documento final completo da tese.
* `Base_VALIDADA_E_PRONTA.csv`: A base de dados principal (N=139) utilizada para todas as análises.

### Scripts de Análise

* `master.py`: (Idêntico a `tese_master.py`) Pipeline principal para geração das figuras descritivas (Figuras 3.1 a 3.7) e modelos de moderação H3 (ln(Pop)).
* `tese_master2.py`: Script dedicado aos modelos de interação centrais da tese (H5a/Modelo 27B: *Gestão × ΔAbstenção* e H5b/Modelo 27C: *Vantagem2016 × ΔCompetição*).
* `testes_robustez.py`: Script para execução de todos os testes de robustez mencionados no Apêndice (Specification Curve, Testes de Placebo, Probit, LPM, WLS).
* `analise_modelos_logit_ols.py`: Script focado na estimação e visualização dos modelos principais (Logit Modelo 3 e OLS Modelo 6), gerando as Tabelas 5 e 6 da tese.
* `scripts_depuracao/` (Pasta Sugerida):
    * `check_columns.py`: Utilitário para normalizar nomes de colunas.
    * `check_missing.py`: Utilitário para verificar dados ausentes nas variáveis-chave.
    * `debug_27C.py`: Script de depuração para validar a amostra do Modelo 27C.

## Como Reproduzir os Resultados

Para executar as análises e gerar as figuras e tabelas, siga estes passos:

1.  **Clone o Repositório:**
    ```bash
    git clone [https://github.com/perichrafa-ai/Tese.git](https://github.com/perichrafa-ai/Tese.git)
    cd Tese
    ```

2.  **Crie um Ambiente Virtual (Recomendado):**
    ```bash
    python -m venv .venv
    # No Windows
    .\.venv\Scripts\activate
    # No macOS/Linux
    source .venv/bin/activate
    ```

3.  **Instale as Dependências:**
    Os scripts `master.py` e `tese_master2.py` possuem um gerenciador de dependências embutido. Eles criarão o arquivo `requirements.txt` e instalarão os pacotes necessários (como `pandas`, `statsmodels`, `matplotlib`) automaticamente na primeira execução.

4.  **Execute os Scripts de Análise:**
    Os resultados serão salvos na pasta `/output` (criada automaticamente).

    ```bash
    # Para gerar as figuras descritivas (Fig 3.1-3.7) e H3
    python master.py --csv "Base_VALIDADA_E_PRONTA.csv" --outdir "./output"

    # Para gerar as figuras dos modelos de interação H5 (27B e 27C)
    python tese_master2.py --run-figures-all --base "Base_VALIDADA_E_PRONTA.csv" --outdir "./output"
    
    # Para executar os testes de robustez (gera figuras de placebo e spec curve)
    python testes_robustez.py
    
    # Para inspecionar os modelos principais (Logit e OLS)
    python analise_modelos_logit_ols.py
    ```

## Citação

Se você utilizar esta base de dados ou os resultados desta pesquisa, por favor, cite:

PERICH, Rafael Azevedo. **Responsabilização Eleitoral em Tempos de Crise: Os limites do voto retrospectivo na gestão municipal da pandemia de COVID-19 a partir de casos brasileiros**. Tese (Doutorado em Ciência Política) – Universidade Federal do Paraná, Curitiba, 2025.
