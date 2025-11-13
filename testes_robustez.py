"""
Testes de Robustez - Modelos de Reeleição e Delta Vantagem
Implementa: Modelos Alternativos, Specification Curve, Testes de Placebo e Análise de Sensibilidade
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from scipy import stats
import warnings
from itertools import combinations
warnings.filterwarnings('ignore')

# Configuração
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

print("=" * 80)
print("TESTES DE ROBUSTEZ - ANALISE COMPLETA")
print("=" * 80)

# =============================================================================
# CARREGAR E PREPARAR DADOS (Mesmo processo do script principal)
# =============================================================================

print("\n" + "=" * 80)
print("1. CARREGANDO E PREPARANDO DADOS...")
print("=" * 80)

df = pd.read_csv('Base_VALIDADA_E_PRONTA.csv', encoding='latin-1', sep=';', decimal=',')
print(f"Dados carregados: {df.shape[0]} linhas, {df.shape[1]} colunas")

# Conversões binárias
sim_nao_cols = [
    'Foi instalado hospital de campanha durante a pandemia da COVID-19 no município',
    'Houve a instalação de tendas de triagem para o combate da COVID-19, no município',
    'O número de leitos foi ampliado para atender à demanda por internação no município em virtude da COVID-19',
    'O número de internações ultrapassou a capacidade de leitos e de unidades de tratamento intensivo (UTI) públicos ou privados con',
    'Nos casos de internação por COVID-19, houve necessidade de referenciar o(s) paciente(s) para outro município',
    'Durante o período da pandemia do COVID-19 foi necessário manter pessoas por mais de 24 horas em unidades sem internação'
]

for col in sim_nao_cols:
    if col in df.columns:
        df[col] = df[col].map({'Sim': 1, 'Não': 0, 'SIM': 1, 'NÃO': 0, 'sim': 1, 'não': 0})

if 'Gênero' in df.columns:
    df['Gênero'] = df['Gênero'].map({'FEMININO': 1, 'Feminino': 1, 'feminino': 1,
                                      'MASCULINO': 0, 'Masculino': 0, 'masculino': 0})

if 'reeleito' in df.columns:
    if df['reeleito'].dtype == 'object':
        df['reeleito'] = df['reeleito'].map({'Sim': 1, 'Não': 0, 'SIM': 1, 'NÃO': 0,
                                              'sim': 1, 'não': 0, 1: 1, 0: 0})

# Converter População e PIB
if 'População' in df.columns:
    if df['População'].dtype == 'object':
        df['População'] = df['População'].astype(str).str.replace('.', '').str.replace(',', '.').astype(float)
    df['ln_pop'] = np.log(df['População'])

pib_col = 'Produto Interno Bruto per capita,  a preços correntes (R$ 1,00)'
if pib_col in df.columns:
    if df[pib_col].dtype == 'object':
        df[pib_col] = df[pib_col].astype(str).str.replace('.', '').str.replace(',', '.').astype(float)

# Criar dummies de região
if 'Região do País' in df.columns:
    region_dummies = pd.get_dummies(df['Região do País'], prefix='Regiao', drop_first=True)
    df = pd.concat([df, region_dummies], axis=1)
    dummy_cols = region_dummies.columns.tolist()
else:
    dummy_cols = []

# Selecionar colunas para o modelo
model_cols = [
    'reeleito',
    'Dias da eleição - Máscara (Normalizado)',
    'Dias da eleição - Restrição ao comércio (Normalizado)',
    'Foi instalado hospital de campanha durante a pandemia da COVID-19 no município',
    'Houve a instalação de tendas de triagem para o combate da COVID-19, no município',
    'O número de leitos foi ampliado para atender à demanda por internação no município em virtude da COVID-19',
    'O número de internações ultrapassou a capacidade de leitos e de unidades de tratamento intensivo (UTI) públicos ou privados con',
    'Nos casos de internação por COVID-19, houve necessidade de referenciar o(s) paciente(s) para outro município',
    'Durante o período da pandemia do COVID-19 foi necessário manter pessoas por mais de 24 horas em unidades sem internação',
    'delta_firjan_saude_16_20',
    'delta_firjan_emprego_16_20',
    'Vantagem do Incumbente no primeiro turno 2016',
    'Número efetivo de candidatos de 2020',
    'ln_pop',
    pib_col,
    'Gênero',
    'delta_vantagem (2020-2016)',
    'População',
    'Estado'  # Para testes de placebo por UF
] + dummy_cols

existing_cols = [col for col in model_cols if col in df.columns]
df_clean = df[existing_cols].dropna().copy()

print(f"Observacoes apos limpeza: {len(df_clean)}")

# Definir fórmulas
independent_vars = [
    "Q('Dias da eleição - Máscara (Normalizado)')",
    "Q('Dias da eleição - Restrição ao comércio (Normalizado)')",
    "Q('Foi instalado hospital de campanha durante a pandemia da COVID-19 no município')",
    "Q('Houve a instalação de tendas de triagem para o combate da COVID-19, no município')",
    "Q('O número de leitos foi ampliado para atender à demanda por internação no município em virtude da COVID-19')",
    "Q('O número de internações ultrapassou a capacidade de leitos e de unidades de tratamento intensivo (UTI) públicos ou privados con')",
    "Q('Nos casos de internação por COVID-19, houve necessidade de referenciar o(s) paciente(s) para outro município')",
    "Q('Durante o período da pandemia do COVID-19 foi necessário manter pessoas por mais de 24 horas em unidades sem internação')",
    "delta_firjan_saude_16_20",
    "delta_firjan_emprego_16_20",
    "Q('Vantagem do Incumbente no primeiro turno 2016')",
    "Q('Número efetivo de candidatos de 2020')",
    "ln_pop",
    "Q('Produto Interno Bruto per capita,  a preços correntes (R$ 1,00)')",
    "Gênero"
]

for col in dummy_cols:
    if ' ' in col or '-' in col:
        independent_vars.append(f"Q('{col}')")
    else:
        independent_vars.append(col)

logit_formula = "reeleito ~ " + " + ".join(independent_vars)
ols_formula = "Q('delta_vantagem (2020-2016)') ~ " + " + ".join(independent_vars)

print("\n[OK] Dados preparados com sucesso")

# =============================================================================
# 1. MODELOS ALTERNATIVOS
# =============================================================================

print("\n" + "=" * 80)
print("2. TESTANDO FAMILIAS DE MODELOS ALTERNATIVAS")
print("=" * 80)

# 1.1. Modelo Probit
print("\n" + "-" * 80)
print("2.1. MODELO PROBIT (Alternativa ao Logit)")
print("-" * 80)

try:
    probit_model = smf.probit(formula=logit_formula, data=df_clean).fit(cov_type='HC1', disp=False, maxiter=100)
    print("[OK] Modelo Probit estimado com sucesso")

    print("\nSumario do Modelo Probit:")
    print(probit_model.summary())

    print(f"\nProbit AIC: {probit_model.aic:.4f}")
    print(f"Probit BIC: {probit_model.bic:.4f}")

    # AMEs do Probit
    try:
        probit_ames = probit_model.get_margeff(at='overall', method='dydx')
        print("\nAverage Marginal Effects (AMEs) - Probit:")
        print(probit_ames.summary())
    except Exception as e:
        print(f"[AVISO] Nao foi possivel calcular AMEs do Probit: {e}")

except Exception as e:
    print(f"[ERRO] ao estimar Modelo Probit: {e}")

# 1.2. Modelo Linear de Probabilidade (LPM)
print("\n" + "-" * 80)
print("2.2. MODELO LINEAR DE PROBABILIDADE (LPM)")
print("-" * 80)

try:
    lpm_model = smf.ols(formula=logit_formula, data=df_clean).fit(cov_type='HC1')
    print("[OK] Modelo LPM estimado com sucesso")

    print("\nSumario do Modelo LPM:")
    print(lpm_model.summary())

    print(f"\nLPM AIC: {lpm_model.aic:.4f}")
    print(f"LPM BIC: {lpm_model.bic:.4f}")

except Exception as e:
    print(f"[ERRO] ao estimar Modelo LPM: {e}")

# 1.3. OLS Ponderado por População (WLS)
print("\n" + "-" * 80)
print("2.3. MODELO WLS PONDERADO POR POPULACAO")
print("-" * 80)

try:
    wls_model = smf.wls(formula=ols_formula, data=df_clean, weights=df_clean['População']).fit(cov_type='HC1')
    print("[OK] Modelo WLS estimado com sucesso")

    print("\nSumario do Modelo WLS:")
    print(wls_model.summary())

    print(f"\nWLS AIC: {wls_model.aic:.4f}")
    print(f"WLS BIC: {wls_model.bic:.4f}")

except Exception as e:
    print(f"[ERRO] ao estimar Modelo WLS: {e}")

# =============================================================================
# 2. SPECIFICATION CURVE ANALYSIS
# =============================================================================

print("\n" + "=" * 80)
print("3. SPECIFICATION CURVE ANALYSIS")
print("=" * 80)

# Definir grupos de controles
controles_base = [
    "Q('Vantagem do Incumbente no primeiro turno 2016')",
    "Q('Número efetivo de candidatos de 2020')"
]

controles_estruturais = ['ln_pop', "Q('Produto Interno Bruto per capita,  a preços correntes (R$ 1,00)')", "Gênero"]

controles_regiao = [f"Q('{col}')" if ' ' in col or '-' in col else col for col in dummy_cols]

controles_firjan = ['delta_firjan_saude_16_20', 'delta_firjan_emprego_16_20']

# Variáveis de interesse (Gestão/Severidade)
vars_interesse = [
    "Q('Dias da eleição - Máscara (Normalizado)')",
    "Q('Dias da eleição - Restrição ao comércio (Normalizado)')",
    "Q('Foi instalado hospital de campanha durante a pandemia da COVID-19 no município')",
    "Q('Houve a instalação de tendas de triagem para o combate da COVID-19, no município')",
    "Q('O número de leitos foi ampliado para atender à demanda por internação no município em virtude da COVID-19')",
    "Q('O número de internações ultrapassou a capacidade de leitos e de unidades de tratamento intensivo (UTI) públicos ou privados con')",
    "Q('Nos casos de internação por COVID-19, houve necessidade de referenciar o(s) paciente(s) para outro município')",
    "Q('Durante o período da pandemia do COVID-19 foi necessário manter pessoas por mais de 24 horas em unidades sem internação')"
]

# Gerar especificações
especificacoes = []
grupos_controles = {
    'estruturais': controles_estruturais,
    'regiao': controles_regiao,
    'firjan': controles_firjan
}

print("\nGerando especificacoes...")

# Todas as combinações de grupos de controles
grupo_names = list(grupos_controles.keys())
for r in range(len(grupo_names) + 1):
    for combo in combinations(grupo_names, r):
        controles = controles_base.copy()
        spec_name = "base"

        for grupo in combo:
            controles.extend(grupos_controles[grupo])
            spec_name += f"+{grupo}"

        # Adicionar variáveis de interesse
        all_vars = vars_interesse + controles

        formula_logit = "reeleito ~ " + " + ".join(all_vars)
        formula_ols = "Q('delta_vantagem (2020-2016)') ~ " + " + ".join(all_vars)

        especificacoes.append({
            'nome': spec_name,
            'formula_logit': formula_logit,
            'formula_ols': formula_ols,
            'n_controles': len(controles)
        })

print(f"Total de especificacoes geradas: {len(especificacoes)}")

# Estimar modelos para cada especificação
resultados_spec = []

print("\nEstimando modelos para cada especificacao...")

for i, spec in enumerate(especificacoes, 1):
    print(f"  Especificacao {i}/{len(especificacoes)}: {spec['nome']}", end='\r')

    try:
        # Logit
        logit_spec = smf.logit(formula=spec['formula_logit'], data=df_clean).fit(disp=False, maxiter=50)

        # OLS
        ols_spec = smf.ols(formula=spec['formula_ols'], data=df_clean).fit(cov_type='HC1')

        # Extrair coeficientes das variáveis de interesse
        for var in vars_interesse:
            var_name = var.replace("Q('", "").replace("')", "")

            # Logit
            if var in logit_spec.params.index:
                resultados_spec.append({
                    'especificacao': spec['nome'],
                    'modelo': 'Logit',
                    'variavel': var_name,
                    'coef': logit_spec.params[var],
                    'pvalue': logit_spec.pvalues[var],
                    'significativo': logit_spec.pvalues[var] < 0.05,
                    'n_controles': spec['n_controles']
                })

            # OLS
            if var in ols_spec.params.index:
                resultados_spec.append({
                    'especificacao': spec['nome'],
                    'modelo': 'OLS',
                    'variavel': var_name,
                    'coef': ols_spec.params[var],
                    'pvalue': ols_spec.pvalues[var],
                    'significativo': ols_spec.pvalues[var] < 0.05,
                    'n_controles': spec['n_controles']
                })

    except Exception as e:
        continue

print(f"\n[OK] {len(resultados_spec)} resultados coletados")

# Criar DataFrame com resultados
df_spec = pd.DataFrame(resultados_spec)

# Visualização: Specification Curve
print("\nCriando visualizacao da Specification Curve...")

# Focar em NEC_2020
var_foco = "Número efetivo de candidatos de 2020"
df_nec = df_spec[df_spec['variavel'] == var_foco].copy()

if len(df_nec) > 0:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10),
                                     gridspec_kw={'height_ratios': [3, 1]})

    # Panel superior: Coeficientes
    for modelo in ['Logit', 'OLS']:
        data = df_nec[df_nec['modelo'] == modelo].sort_values('coef')
        x = range(len(data))

        colors = ['red' if sig else 'lightgray' for sig in data['significativo']]

        ax1.scatter(x, data['coef'], c=colors, label=modelo, s=60, alpha=0.7)
        ax1.plot(x, data['coef'], alpha=0.3)

    ax1.axhline(0, color='black', linestyle='--', linewidth=1)
    ax1.set_ylabel('Coeficiente Estimado', fontsize=12, fontweight='bold')
    ax1.set_title(f'Specification Curve: {var_foco}', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Panel inferior: Indicadores das especificações
    ax2.set_xlabel('Especificacao (ordenada por coeficiente)', fontsize=12)
    ax2.set_ylabel('N Controles', fontsize=10)
    ax2.set_ylim([0, df_nec['n_controles'].max() + 2])

    plt.tight_layout()
    plt.savefig('specification_curve_nec2020.png', dpi=300, bbox_inches='tight')
    print("[OK] Grafico salvo: specification_curve_nec2020.png")
    plt.close()

# Salvar resumo estatístico
print("\nResumo da Specification Curve (NEC 2020):")
for modelo in ['Logit', 'OLS']:
    data = df_nec[df_nec['modelo'] == modelo]
    if len(data) > 0:
        print(f"\n{modelo}:")
        print(f"  Media dos coeficientes: {data['coef'].mean():.4f}")
        print(f"  Desvio padrao: {data['coef'].std():.4f}")
        print(f"  Min: {data['coef'].min():.4f}, Max: {data['coef'].max():.4f}")
        print(f"  % significativo (p<0.05): {data['significativo'].mean()*100:.1f}%")

# =============================================================================
# 3. TESTES DE PLACEBO
# =============================================================================

print("\n" + "=" * 80)
print("4. TESTES DE PLACEBO")
print("=" * 80)

# 3.1. Placebo por Permutação (Dentro das UFs)
print("\n" + "-" * 80)
print("4.1. TESTE DE PLACEBO POR PERMUTACAO (DENTRO DAS UFs)")
print("-" * 80)

n_permutacoes = 250
coefs_placebo_logit = []
coefs_placebo_ols = []

print(f"\nExecutando {n_permutacoes} permutacoes...")

for i in range(n_permutacoes):
    if (i + 1) % 100 == 0:
        print(f"  Permutacao {i+1}/{n_permutacoes}", end='\r')

    # Criar cópia dos dados
    df_perm = df_clean.copy()

    # Permutar reeleito e delta_vantagem dentro de cada UF
    for uf in df_perm['Estado'].unique():
        mask = df_perm['Estado'] == uf
        df_perm.loc[mask, 'reeleito'] = np.random.permutation(df_perm.loc[mask, 'reeleito'].values)
        df_perm.loc[mask, 'delta_vantagem (2020-2016)'] = np.random.permutation(
            df_perm.loc[mask, 'delta_vantagem (2020-2016)'].values
        )

    try:
        # Logit
        logit_perm = smf.logit(formula=logit_formula, data=df_perm).fit(disp=False, maxiter=30)
        if "Q('Número efetivo de candidatos de 2020')" in logit_perm.params.index:
            coefs_placebo_logit.append(logit_perm.params["Q('Número efetivo de candidatos de 2020')"])

        # OLS
        ols_perm = smf.ols(formula=ols_formula, data=df_perm).fit()
        if "Q('Número efetivo de candidatos de 2020')" in ols_perm.params.index:
            coefs_placebo_ols.append(ols_perm.params["Q('Número efetivo de candidatos de 2020')"])
    except:
        continue

print(f"\n[OK] Permutacoes concluidas: {len(coefs_placebo_logit)} Logit, {len(coefs_placebo_ols)} OLS")

# Estimar modelos reais para comparação
logit_real = smf.logit(formula=logit_formula, data=df_clean).fit(disp=False, maxiter=100)
ols_real = smf.ols(formula=ols_formula, data=df_clean).fit(cov_type='HC1')

coef_real_logit = logit_real.params["Q('Número efetivo de candidatos de 2020')"]
coef_real_ols = ols_real.params["Q('Número efetivo de candidatos de 2020')"]

# Visualização
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Logit
ax1.hist(coefs_placebo_logit, bins=50, alpha=0.7, color='gray', edgecolor='black')
ax1.axvline(coef_real_logit, color='red', linewidth=3, label=f'Coef. Real: {coef_real_logit:.3f}')
ax1.set_xlabel('Coeficiente (NEC 2020)', fontsize=12)
ax1.set_ylabel('Frequencia', fontsize=12)
ax1.set_title('Teste de Placebo - Modelo Logit', fontsize=13, fontweight='bold')
ax1.legend()
ax1.grid(alpha=0.3)

# OLS
ax2.hist(coefs_placebo_ols, bins=50, alpha=0.7, color='gray', edgecolor='black')
ax2.axvline(coef_real_ols, color='red', linewidth=3, label=f'Coef. Real: {coef_real_ols:.3f}')
ax2.set_xlabel('Coeficiente (NEC 2020)', fontsize=12)
ax2.set_ylabel('Frequencia', fontsize=12)
ax2.set_title('Teste de Placebo - Modelo OLS', fontsize=13, fontweight='bold')
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('teste_placebo_permutacao.png', dpi=300, bbox_inches='tight')
print("[OK] Grafico salvo: teste_placebo_permutacao.png")
plt.close()

# P-valor empírico
p_value_logit = np.mean(np.abs(coefs_placebo_logit) >= np.abs(coef_real_logit))
p_value_ols = np.mean(np.abs(coefs_placebo_ols) >= np.abs(coef_real_ols))

print(f"\nP-valor empirico (Logit): {p_value_logit:.4f}")
print(f"P-valor empirico (OLS): {p_value_ols:.4f}")

# 3.2. Placebo Temporal
print("\n" + "-" * 80)
print("4.2. TESTE DE PLACEBO TEMPORAL (Pre-2020)")
print("-" * 80)

# Verificar se dados de 2012 estão disponíveis
colunas_2012 = [col for col in df.columns if '2012' in str(col)]
if len(colunas_2012) > 0:
    print(f"[INFO] Encontradas {len(colunas_2012)} colunas com dados de 2012:")
    for col in colunas_2012[:5]:
        print(f"  - {col}")
    print("[INFO] Implementacao do placebo temporal requer mais investigacao dos dados")
else:
    print("[AVISO] Dados de 2012 nao encontrados na base.")
    print("        Teste de placebo temporal nao pode ser realizado.")

# =============================================================================
# 4. SENSIBILIDADE A CONFUNDIDORES NÃO OBSERVADOS
# =============================================================================

print("\n" + "=" * 80)
print("5. ANALISE DE SENSIBILIDADE A CONFUNDIDORES NAO OBSERVADOS")
print("=" * 80)

print("\n[INFO] Analise simplificada de robustez do coeficiente NEC_2020")

# Focar no modelo OLS (mais simples para análise de viés)
var_interesse_sens = "Q('Número efetivo de candidatos de 2020')"

if var_interesse_sens in ols_real.params.index:
    coef_nec = ols_real.params[var_interesse_sens]
    se_nec = ols_real.bse[var_interesse_sens]
    t_stat = coef_nec / se_nec

    print(f"\nModelo OLS - NEC 2020:")
    print(f"  Coeficiente: {coef_nec:.4f}")
    print(f"  Erro padrao: {se_nec:.4f}")
    print(f"  Estatistica t: {t_stat:.4f}")

    # Calcular o R² parcial do tratamento
    # Usar uma aproximação: partial R² ≈ t² / (t² + df)
    df_resid = ols_real.df_resid
    partial_r2 = (t_stat ** 2) / (t_stat ** 2 + df_resid)

    print(f"\nR² parcial (NEC 2020 com outcome): {partial_r2:.4f}")

    # Robustness Value: quão forte um confundidor precisa ser
    # Para tornar o efeito insignificante (t < 1.96), precisaríamos reduzir |t| de t_stat para 1.96
    # Isso é uma aproximação simplificada

    critical_t = 1.96
    if abs(t_stat) > critical_t:
        # Proporção do efeito que precisa ser "explicada" por confundidor
        prop_explicar = 1 - (critical_t / abs(t_stat))

        print(f"\nAnalise de Robustez (simplificada):")
        print(f"  Um confundidor nao observado precisaria explicar {prop_explicar*100:.1f}%")
        print(f"  do efeito atual de NEC_2020 para tornar o resultado nao-significativo.")
        print(f"\n  Interpretacao: Quanto maior essa porcentagem, mais robusto e o resultado.")

        # Aproximação do RV (Robustness Value)
        # RV ≈ sqrt(partial_r2 * (1 - prop_explicar))
        rv_approx = np.sqrt(partial_r2 * (1 - prop_explicar))
        print(f"\n  Robustness Value (aprox.): {rv_approx:.4f}")
        print(f"  Um confundidor com R² parcial > {rv_approx:.4f} com o tratamento")
        print(f"  E com o outcome poderia invalidar a conclusao.")
    else:
        print("\n[AVISO] Coeficiente ja e nao-significativo no nivel convencional.")

print("\n" + "=" * 80)
print("TESTES DE ROBUSTEZ CONCLUIDOS!")
print("=" * 80)

print("\nArquivos gerados:")
print("  - specification_curve_nec2020.png")
print("  - teste_placebo_permutacao.png")
