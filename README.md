# Tese — Responsabilização eleitoral em tempos de crise: os limites do voto retrospectivo na gestão municipal da pandemia de Covid-19 a partir de casos brasileiros

[![Python](https://img.shields.io/badge/Python-3.10–3.12-informational)](https://www.python.org/)
[![OS](https://img.shields.io/badge/Windows%20%7C%20macOS%20%7C%20Linux-ok)](#)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Releases](https://img.shields.io/badge/Download-Releases-brightgreen)](./releases)
[![Data](https://img.shields.io/badge/Dataset-CSV-important)](#)
<!-- [![DOI](https://img.shields.io/badge/DOI-em_breve-lightgrey)](#) -->

Este repositório contém **scripts Python** para rodar, de ponta a ponta, as análises automatizadas da tese
**“RESPONSABILIZAÇÃO ELEITORAL EM TEMPOS DE CRISE: OS LIMITES DO VOTO RETROSPECTIVO NA GESTÃO MUNICIPAL DA PANDEMIA DE COVID-19 A PARTIR DE CASOS BRASILEIROS”**.
O pipeline lê o CSV validado, reconstrói variáveis quando necessário, roda modelos (Logit/OLS), calcula VIF, AMEs,
e gera **tabelas e figuras** exatamente como especificado no texto.

> **Arquivos relevantes**
>
> - `Base_VALIDADA_E_PRONTA.csv` — base integrada usada em todas as rotinas.
> - `master parte 1.py` — _pipeline principal_ (descritivas, VIF, Logit, OLS, coefplot).
> - `master parte 2.py` — _interações finais_ (robustez + AMEs; Tabelas 3.4a–d + 2 figuras).
> - `master parte 3.py` — _H3/H4_ (autodetecção de colunas, construção de índices, predições).
> - `tese.pdf` — versão atual da tese (referência metodológica e numérica).

---

## Índice
- [Requisitos](#requisitos)
- [Instalação rápida](#instalação-rápida)
- [Como executar](#como-executar)
- [Parâmetros importantes](#parâmetros-importantes)
- [Saídas (tabelas & figuras)](#saídas-tabelas--figuras)
- [Reprodutibilidade e solução de problemas](#reprodutibilidade-e-solução-de-problemas)
- [Estrutura de pastas](#estrutura-de-pastas)
- [Referência metodológica](#referência-metodológica)
- [Releases & versão dos scripts](#releases--versão-dos-scripts)
- [Citação & licença](#citação--licença)

---

## Requisitos

- **Python** 3.10–3.12.
- Nenhuma permissão administrativa; tudo roda no diretório do projeto.
- Windows, macOS ou Linux.

> Os scripts **1** e **3** criam/gerenciam automaticamente a **venv local** (`.venv`) e instalam dependências fixadas.
> O script **2** assume que o ambiente já está pronto (execute **antes** o `master parte 1.py`).

---

## Instalação rápida

> **Windows (PowerShell)** — recomendado usar o lançador `py`:
```powershell
# Rode o script 1 para bootstrap do ambiente (.venv + requirements)
py ".\master parte 1.py" --csv ".\Base_VALIDADA_E_PRONTA.csv" --outdir ".\output" --encoding latin-1 --sep ";" --decimal ","
```

> **macOS / Linux (bash/zsh)**:
```bash
python3 "./master parte 1.py" --csv "./Base_VALIDADA_E_PRONTA.csv" --outdir "./output" --encoding latin-1 --sep ";" --decimal ","
```

Isso prepara a venv (`.venv/`) e baixa as dependências (pandas, numpy, statsmodels, matplotlib, seaborn, openpyxl, etc.).

---

## Como executar

> **Pipeline principal (descritivas + VIF + Logit + OLS + coefplot)**
```powershell
# Windows
py ".\master parte 1.py" --csv ".\Base_VALIDADA_E_PRONTA.csv" --outdir ".\output" --encoding latin-1 --sep ";" --decimal ","
```
```bash
# macOS / Linux
python3 "./master parte 1.py" --csv "./Base_VALIDADA_E_PRONTA.csv" --outdir "./output" --encoding latin-1 --sep ";" --decimal ","
```

> **Interações finais (robustez + AMEs; Tabelas 3.4a–d + 2 figuras)**
```powershell
py ".\master parte 2.py" --csv ".\Base_VALIDADA_E_PRONTA.csv" --outdir ".\output" --encoding latin-1 --sep ";" --decimal ","
```
```bash
python3 "./master parte 2.py" --csv "./Base_VALIDADA_E_PRONTA.csv" --outdir "./output" --encoding latin-1 --sep ";" --decimal ","
```

> **H3/H4 (autodetecção + construção de índices + predições)**
```powershell
py ".\master parte 3.py" --csv ".\Base_VALIDADA_E_PRONTA.csv" --outdir ".\output" --encoding latin-1 --sep ";" --decimal "," --ols-hc1
```
```bash
python3 "./master parte 3.py" --csv "./Base_VALIDADA_E_PRONTA.csv" --outdir "./output" --encoding latin-1 --sep ";" --decimal "," --ols-hc1
```

---

## Parâmetros importantes

Os scripts aceitam *flags* alinhadas ao formato brasileiro do CSV (encoding/sep/decimal) e mapeamento de colunas.
Alguns parâmetros úteis:

### Comuns
- `--csv` caminho do CSV; `--outdir` pasta de saída; `--encoding` (padrão: `latin-1`); `--sep` (padrão: `;`);
  `--decimal` (padrão: `,`); `--map` para renomear colunas específicas na leitura.

**Exemplo de `--map`:**
```bash
--map Reeleicao="Reeleito (0/1)" --map Vantagem2016="Vantagem do Incumbente no primeiro turno 2016" --map DeltaIFDM_Emprego="Δ IFDM Emprego & Renda (2020-2016)"
```

### Específicos do `master parte 1.py`
- `--no-xlsx` desativa exportação `.xlsx` (mantém `.csv`).
- `--max-num-graphs-per-cat` (padrão: 3), `--top-k-categories` (se aplicável).
- `--check-delta-bins` habilita inspeções adicionais.
- `--ifdm_xmin`, `--ifdm_xmax`, `--ifdm_bw_adjust` controlam janelas e suavização de densidade para IFDM.

### Específicos do `master parte 3.py`
- `--ols-hc1` força OLS com erros‑padrão robustos HC1 (se preferir ao cluster por Estado).
- `--upgrade` (quando disponível) atualiza _wheels_ estáveis dentro da venv.

> O `master parte 2.py` não gerencia venv; execute o `master parte 1.py` primeiro.

---

## Saídas (tabelas & figuras)

Tudo é salvo em `./output/` (criado automaticamente). Você encontrará:

- **Tabelas (.csv e, quando disponível, .xlsx)**: VIF (3.0), Logit (3.1), OLS (3.2), Interações (3.4a–d) e sumários H3/H4.
- **Figuras (.png)** sem títulos embutidos (apenas eixos/legenda): coefplot principal, gráficos de predição e AMEs.
- Nomes de arquivos seguem padrão consistente, facilitando referência cruzada com a tese.

> **Boas práticas**: manter o nome da base no nome do arquivo gerado quando pertinente (ex.: `coefplot_Base_VALIDADA_E_PRONTA.png`).

---

## Reprodutibilidade e solução de problemas

- **Ambiente quebrado?** Apague `.venv/` e execute novamente o `master parte 1.py`.
- **ImportError (pandas/statsmodels)?** Certifique‑se de estar usando `py` (Windows) ou `python3` (macOS/Linux) conforme acima.
- **CSV fora do padrão BR?** Ajuste `--encoding`, `--sep` e `--decimal`.
- **PowerShell**: o `master parte 3.py` ignora carets `^` acidentais herdados de comandos quebrados.
- **Logs ruidosos do Matplotlib**: suprimidos nos scripts; gráficos exportados em PNG de alta resolução.

---

## Estrutura de pastas

```
.
├── Base_VALIDADA_E_PRONTA.csv
├── master parte 1.py
├── master parte 2.py
├── master parte 3.py
├── tese.pdf
└── output/
```

> Você pode usar outra estrutura, mas garanta que `--csv` e `--outdir` apontem para os caminhos corretos.

---

## Referência metodológica

A especificação de variáveis, modelos, hipóteses e a lista canônica de tabelas/figuras está em **`tese.pdf`** (raiz do projeto).

---

## Releases & versão dos scripts

Use a aba **[Releases](./releases)** para publicar pacotes versionados do `output/` (tabelas e figuras) e _snapshots_ de código.
Recomendação de versionamento: `vMAJOR.MINOR.PATCH` alinhado aos cabeçalhos internos dos scripts
(p.ex., `master_v18`, `master_2_v5`, `master_h3_h4_v3`).

---



---

> **Resumo**: execute o **script 1** para preparar ambiente e gerar as saídas básicas, rode o **script 2** para as interações finais,
e finalize com o **script 3** para H3/H4 com autodetecção e predições. Todas as tabelas e figuras aparecem em `output/`.
