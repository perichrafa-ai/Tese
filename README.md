# Pipeline da Tese — Reprodutibilidade e Execução dos Scripts

Este repositório contém **três scripts Python** que automatizam a leitura, o pré-processamento e as análises estatísticas dos dados da tese, além de gerar **tabelas e figuras** exatamente como descrito no trabalho. O repositório inclui também o CSV final utilizado nas análises.

> **Arquivos relevantes**
>
> - `Base_VALIDADA_E_PRONTA.csv` — base integrada usada em todas as rotinas.
> - `master parte 1.py` — _pipeline principal_ (descritivas + VIF + Logit + OLS + coefplot).
> - `master parte 2.py` — _interações finais_ (robustez + AMEs + 2 gráficos; Tabelas 3.4a–d).
> - `master parte 3.py` — _H3/H4 com autodetecção_ (construção dos índices; predições).
> - `tese.pdf` — versão atual da tese, para referência metodológica e numérica.

Os scripts criam e usam automaticamente uma **venv local** (`.venv`) e instalam as dependências **com versões fixas** via `requirements.txt`, garantindo resultados reprodutíveis em Windows, macOS e Linux.

---

## 1) Pré-requisitos

- **Python**: 3.10–3.12 (Windows: o lançador `py` costuma apontar para a versão correta).
- **Permissões**: nenhuma permissão administrativa é necessária; tudo roda no diretório do projeto.
- **SO**: Windows, macOS ou Linux.

> Se você já tem um `.venv` mas está quebrado, **apague a pasta `.venv/`** e rode novamente qualquer script — a venv é recriada.

---

## 2) Estrutura mínima de pastas

```
.
├── Base_VALIDADA_E_PRONTA.csv
├── master parte 1.py
├── master parte 2.py
├── master parte 3.py
├── tese.pdf
└── output/              # será criado automaticamente (se não existir)
```

---

## 3) Uso rápido (one‑liners por SO)

> **Windows (PowerShell):**
```powershell
# Pipeline principal (descritivas + VIF + Logit + OLS + coefplot)
py ".\master parte 1.py" --csv ".\Base_VALIDADA_E_PRONTA.csv" --outdir ".\output" --encoding latin-1 --sep ";" --decimal ","

# Interações finais (robustez + AMEs; gera Tabelas 3.4a–d e 2 gráficos)
py ".\master parte 2.py" --csv ".\Base_VALIDADA_E_PRONTA.csv" --outdir ".\output" --encoding latin-1 --sep ";" --decimal ","

# H3/H4 com autodetecção (constrói índices e gera predições; opção HC1 no OLS)
py ".\master parte 3.py" --csv ".\Base_VALIDADA_E_PRONTA.csv" --outdir ".\output" --encoding latin-1 --sep ";" --decimal "," --ols-hc1
```

> **macOS / Linux (bash/zsh):**
```bash
# Troque 'python3' por 'python' se preferir
python3 "./master parte 1.py" --csv "./Base_VALIDADA_E_PRONTA.csv" --outdir "./output" --encoding latin-1 --sep ";" --decimal ","

python3 "./master parte 2.py" --csv "./Base_VALIDADA_E_PRONTA.csv" --outdir "./output" --encoding latin-1 --sep ";" --decimal ","

python3 "./master parte 3.py" --csv "./Base_VALIDADA_E_PRONTA.csv" --outdir "./output" --encoding latin-1 --sep ";" --decimal "," --ols-hc1
```

As **opções** `--encoding latin-1`, `--sep ";"` e `--decimal ","` refletem a convenção brasileira do CSV. Altere **somente** se o arquivo tiver outro formato.

---

## 4) O que cada script faz

### `master parte 1.py` — *pipeline principal (master_v18)*
- Gera **estatísticas descritivas**, **VIF** (Tabela 3.0), **Logit** de reeleição (Tabela 3.1, com erros‑padrão robustos HC1) e **OLS** para Δ Vantagem (Tabela 3.2).
- Exporta **coefplot** do Modelo 6 (Figura 3.7).
- Silencia logs ruidosos do Matplotlib (“categorical units”), evita `SettingWithCopyWarning` e **converte eixos numéricos** explicitamente para `float`.
- **Parâmetros extras** úteis:
  - `--no-xlsx` — desativa exportação `.xlsx` (mantém `.csv`).
  - `--max-num-graphs-per-cat` (padrão: 3) e `--top-k-categories` (padrão: 30) — controlam seleção de categorias para gráficos descritivos.
  - `--check-delta-bins` — inspeções adicionais sobre deltas.
  - `--ifdm_xmin/--ifdm_xmax/--ifdm_bw_adjust` — ajustes para curvas de densidade do IFDM.

### `master parte 2.py` — *interações finais (master_2_v5)*
- Foca nas **interações finais** com **robustez** e **efeitos marginais médios (AMEs)**.
- **Corrige** o erro `DiscreteMargins` (usa `summary_frame()` internamente).
- **Autodetecta** colunas a partir de _keywords_ do CSV e constrói as variáveis necessárias.
- Produz **Tabelas 3.4a–3.4d** e **dois gráficos** finais, sem títulos embutidos (apenas eixos/legenda).

### `master parte 3.py` — *H3/H4 + índices internos (master_h3_h4_v3)*
- **Descobre** automaticamente os nomes de colunas no CSV (ex.: `"Estado"` em vez de `"UF"`, `"reeleito"`, `"delta_vantagem (2020-2016)"`, `"População"`/`"ln_pop"`).
- **Constrói os índices de Gestão e Severidade** a partir das variáveis presentes (máscara, comércio, leitos, tendas, hospital, testagem; sobrecarga/transferência/24h).
- **Tolerante a PowerShell** (remove carets `^` errôneos quando o comando é quebrado em várias linhas).
- Saídas: **quatro sumários/tabelas** (H3/H4 × Logit/OLS) e **figuras de predição**.
- **Opção**: `--ols-hc1` para OLS com HC1 (padrão: cluster por Estado, se disponível).

---

## 5) Convenções do CSV e mapeamento de colunas

Os scripts tentam **autodetectar** os nomes de colunas por _keywords_. Caso sua coluna tenha um nome diferente, use `--map` para **forçar o mapeamento**:

```bash
# Exemplos:
--map Reeleicao="Reeleito (0/1)" --map Vantagem2016="Vantagem do Incumbente no primeiro turno 2016" --map DeltaIFDM_Emprego="Δ IFDM Emprego & Renda (2020-2016)" --map DeltaIFDM_Saude="IFDM Saúde – variação (2016-2020)"
```

> Dicas importantes:
> - **Reeleição**: valores binários são interpretados de maneira robusta (`Sim/Não`, `1/0`, `True/False` etc.).
> - **Abstenção 2016/2020** e **NEC 2016/2020**: usados para calcular Δ Abstenção (p.p.) e Δ NEC (diff/log).
> - **Separadores**: o leitor reconhece **ponto** como separador de milhar e **vírgula** como separador decimal quando `--decimal ","`.
> - **População**: se não houver `ln_pop`, ele é **reconstruído** a partir de `População`.

---

## 6) Saídas geradas

Tudo é salvo em `./output/` (criado automaticamente). Você encontrará:

- **Tabelas (.csv e, quando possível, .xlsx)** de VIF (3.0), Logit (3.1), OLS (3.2) e interações (3.4a–d), além de sumários para H3/H4.
- **Figuras (.png)** sem títulos embutidos (apenas eixos/legenda), incluindo o **coefplot** principal e **gráficos de predição/AMEs**.
- Os arquivos são numerados de forma consistente para facilitar a referência cruzada com a tese.

---

## 7) Dicas de reprodutibilidade e solução de problemas

- **Venv quebrada?** Apague `.venv/` e rode de novo qualquer script — a venv é recriada e os _wheels_ estáveis são reinstalados.
- **Erro de `pandas`/`statsmodels` não encontrado?** Os scripts chamam `pip` dentro da venv. Certifique-se de usar `py` (Windows) ou `python3` (macOS/Linux).
- **CSV com formato diferente?** Ajuste `--encoding`, `--sep` e `--decimal` de acordo com o arquivo.
- **PowerShell e `^`**: o script `master parte 3.py` ignora carets `^` acidentalmente colados ao comando.
- **Logs do Matplotlib**: silenciados automaticamente (`mpl.set_loglevel("warning")`).

---

## 8) Referência metodológica

A especificação dos modelos, variáveis e hipóteses, bem como a lista canônica de tabelas e figuras, encontra-se detalhada em `tese.pdf` (na raiz do repositório).

---

## 9) Licença e citação

- **Licença**: defina uma licença na raiz do repositório (`LICENSE`). Sugerido: MIT ou CC BY‑NC, conforme sua preferência.
- **Como citar**: inclua a referência da tese e, se houver, DOI/handle do repositório de dados.

---

## 10) Resumo dos scripts (visão rápida)

| Script              | Foco principal                                   | Saídas‑chave                                   | Extras |
|---------------------|---------------------------------------------------|------------------------------------------------|--------|
| `master parte 1.py` | Descritivas, VIF, Logit (Reeleição), OLS (ΔV)    | Tabelas 3.0–3.2; Figura 3.7 (coefplot)         | Flags de gráficos e IFDM |
| `master parte 2.py` | Interações finais + AMEs + robustez               | Tabelas 3.4a–3.4d; 2 gráficos finais           | Corrige `DiscreteMargins` |
| `master parte 3.py` | H3/H4 com autodetecção + construção de índices    | 4 sumários (H3/H4 × Logit/OLS); predições     | `--ols-hc1`; tolerante a `^` |

---

> **Reprodutibilidade não é mágica, é método**: use os _one‑liners_, mantenha o CSV no padrão BR (`latin-1`, `;`, `,`), e registre tudo em `output/`. Assim, qualquer pessoa consegue chegar às mesmas tabelas e figuras da tese.

