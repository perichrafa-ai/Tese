"""
Análise Estatística - Modelos Logit e OLS
Mantém os nomes originais das colunas usando Q() nas fórmulas statsmodels
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from pathlib import Path

# Configuração
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# =============================================================================
# 1. CARREGAMENTO E PREPARAÇÃO DOS DADOS
# =============================================================================

print("=" * 80)
print("CARREGANDO DADOS...")
print("=" * 80)

# Carregar o arquivo
df = pd.read_csv(
    'Base_VALIDADA_E_PRONTA.csv',
    encoding='latin-1',
    sep=';',
    decimal=','
)

print(f"Dados carregados: {df.shape[0]} linhas, {df.shape[1]} colunas")
print("\nPrimeiras colunas:")
print(df.columns.tolist()[:10])

# =============================================================================
# 2. CONVERSÃO BINÁRIA
# =============================================================================

print("\n" + "=" * 80)
print("CONVERTENDO VARIÁVEIS BINÁRIAS...")
print("=" * 80)

# Colunas Sim/Não para converter
sim_nao_cols = [
    'Foi instalado hospital de campanha durante a pandemia da COVID-19 no município',
    'Houve a instalação de tendas de triagem para o combate da COVID-19, no município',
    'O número de leitos foi ampliado para atender à demanda por internação no município em virtude da COVID-19',
    'O número de internações ultrapassou a capacidade de leitos e de unidades de tratamento intensivo (UTI) públicos ou privados con',
    'Nos casos de internação por COVID-19, houve necessidade de referenciar o(s) paciente(s) para outro município',
    'Durante o período da pandemia do COVID-19 foi necessário manter pessoas por mais de 24 horas em unidades sem internação'
]

# Converter Sim/Não para 1/0
for col in sim_nao_cols:
    if col in df.columns:
        df[col] = df[col].map({'Sim': 1, 'Não': 0, 'SIM': 1, 'NÃO': 0, 'sim': 1, 'não': 0})
        print(f"[OK] {col[:60]}... convertida")
    else:
        print(f"[AVISO] Coluna nao encontrada: {col}")

# Converter Gênero (assumindo FEMININO=1, MASCULINO=0)
if 'Gênero' in df.columns:
    df['Gênero'] = df['Gênero'].map({'FEMININO': 1, 'Feminino': 1, 'feminino': 1,
                                      'MASCULINO': 0, 'Masculino': 0, 'masculino': 0})
    print("[OK] Genero convertido (FEMININO=1, MASCULINO=0)")
else:
    print("[AVISO] Coluna 'Genero' nao encontrada")

# Converter reeleito (se necessário)
if 'reeleito' in df.columns:
    if df['reeleito'].dtype == 'object':
        df['reeleito'] = df['reeleito'].map({'Sim': 1, 'Não': 0, 'SIM': 1, 'NÃO': 0,
                                              'sim': 1, 'não': 0, 1: 1, 0: 0})
        print("[OK] reeleito convertido")

# =============================================================================
# 3. VARIÁVEL LOG DE POPULAÇÃO
# =============================================================================

print("\n" + "=" * 80)
print("CRIANDO VARIAVEL LOG DE POPULACAO...")
print("=" * 80)

if 'População' in df.columns:
    # Converter População para numérico (remover pontos de milhar e usar vírgula como decimal)
    if df['População'].dtype == 'object':
        df['População'] = df['População'].astype(str).str.replace('.', '').str.replace(',', '.').astype(float)
    df['ln_pop'] = np.log(df['População'])
    print(f"[OK] ln_pop criada (min={df['ln_pop'].min():.2f}, max={df['ln_pop'].max():.2f})")
else:
    print("[AVISO] Coluna 'Populacao' nao encontrada")

# Converter PIB per capita para numérico
pib_col = 'Produto Interno Bruto per capita,  a preços correntes (R$ 1,00)'
if pib_col in df.columns:
    if df[pib_col].dtype == 'object':
        df[pib_col] = df[pib_col].astype(str).str.replace('.', '').str.replace(',', '.').astype(float)
        print(f"[OK] PIB per capita convertido para numerico")
else:
    print("[AVISO] Coluna PIB per capita nao encontrada")

# =============================================================================
# 4. VARIÁVEIS CATEGÓRICAS (DUMMIES DE REGIÃO)
# =============================================================================

print("\n" + "=" * 80)
print("CRIANDO DUMMIES DE REGIÃO...")
print("=" * 80)

if 'Região do País' in df.columns:
    # Criar dummies
    region_dummies = pd.get_dummies(df['Região do País'], prefix='Regiao', drop_first=True)

    # Adicionar ao dataframe
    df = pd.concat([df, region_dummies], axis=1)

    # Armazenar nomes das colunas dummy
    dummy_cols = region_dummies.columns.tolist()
    print(f"[OK] Dummies criadas: {len(dummy_cols)} categorias")
    for col in dummy_cols:
        print(f"  - {col}")
else:
    print("[AVISO] Coluna 'Regiao do Pais' nao encontrada")
    dummy_cols = []

# =============================================================================
# 5. TRATAMENTO DE NULOS
# =============================================================================

print("\n" + "=" * 80)
print("TRATAMENTO DE VALORES AUSENTES...")
print("=" * 80)

# Lista de todas as colunas que serão usadas nos modelos
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
    'Produto Interno Bruto per capita,  a preços correntes (R$ 1,00)',
    'Gênero',
    'delta_vantagem (2020-2016)'
] + dummy_cols

# Verificar colunas existentes
existing_cols = [col for col in model_cols if col in df.columns]
missing_cols = [col for col in model_cols if col not in df.columns]

if missing_cols:
    print("[AVISO] COLUNAS NAO ENCONTRADAS:")
    for col in missing_cols:
        print(f"  - {col}")

print(f"\nObservações antes de remover nulos: {len(df)}")

# Verificar nulos nas colunas existentes
null_counts = df[existing_cols].isnull().sum()
if null_counts.sum() > 0:
    print("\nValores ausentes por coluna:")
    for col in existing_cols:
        if null_counts[col] > 0:
            print(f"  - {col[:60]}...: {null_counts[col]} nulos")

    # Remover linhas com nulos
    df_clean = df[existing_cols].dropna()
    print(f"\nObservacoes apos remover nulos: {len(df_clean)}")
else:
    print("[OK] Nenhum valor ausente encontrado")
    df_clean = df[existing_cols].copy()

# =============================================================================
# 6. DEFINIÇÃO DAS FÓRMULAS DOS MODELOS
# =============================================================================

print("\n" + "=" * 80)
print("VERIFICANDO DADOS...")
print("=" * 80)

# Verificar preditores
predictor_cols = [col for col in existing_cols if col not in ['reeleito', 'delta_vantagem (2020-2016)']]

print(f"Numero de observacoes: {len(df_clean)}")
print(f"Numero de preditores: {len(predictor_cols)}")
print(f"Razao obs/preditores: {len(df_clean)/len(predictor_cols):.1f}")

# Verificar variância
print("\nVerificando variancias...")
zero_var_cols = []
for col in predictor_cols:
    if df_clean[col].var() < 1e-10:
        zero_var_cols.append(col)

if zero_var_cols:
    print(f"[AVISO] Encontradas {len(zero_var_cols)} variaveis com variancia ~0:")
    for col in zero_var_cols:
        print(f"  - {col[:60]}...")
else:
    print("[OK] Todas as variaveis tem variancia > 0")

print("\n" + "=" * 80)
print("DEFININDO FORMULAS DOS MODELOS...")
print("=" * 80)

# Variáveis independentes com Q() para nomes complexos
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

# Adicionar dummies de região com Q() se necessário
for col in dummy_cols:
    if ' ' in col or '-' in col:
        independent_vars.append(f"Q('{col}')")
    else:
        independent_vars.append(col)

# Fórmula do Logit (Model 3)
logit_formula = "reeleito ~ " + " + ".join(independent_vars)
print("\nFórmula Logit (Model 3):")
print(logit_formula)

# Fórmula do OLS (Model 6)
ols_formula = "Q('delta_vantagem (2020-2016)') ~ " + " + ".join(independent_vars)
print("\nFórmula OLS (Model 6):")
print(ols_formula)

# =============================================================================
# 7. ESTIMAÇÃO DOS MODELOS
# =============================================================================

print("\n" + "=" * 80)
print("ESTIMANDO MODELOS...")
print("=" * 80)

# Modelo Logit
print("\nEstimando Modelo Logit...")
try:
    logit_model = smf.logit(formula=logit_formula, data=df_clean).fit(cov_type='HC1', disp=False, maxiter=100)
    print("[OK] Modelo Logit estimado com sucesso")
except Exception as e:
    print(f"[ERRO] ao estimar Modelo Logit: {e}")
    print("[INFO] Tentando sem robust standard errors...")
    try:
        logit_model = smf.logit(formula=logit_formula, data=df_clean).fit(disp=False, maxiter=100)
        print("[OK] Modelo Logit estimado (sem HC1)")
    except Exception as e2:
        print(f"[ERRO] Falha novamente: {e2}")
        print("[INFO] Possivel separacao perfeita ou multicolinearidade nos dados")
        logit_model = None

# Modelo OLS
print("\nEstimando Modelo OLS...")
try:
    ols_model = smf.ols(formula=ols_formula, data=df_clean).fit(cov_type='HC1')
    print("[OK] Modelo OLS estimado com sucesso")
except Exception as e:
    print(f"[ERRO] ao estimar Modelo OLS: {e}")
    ols_model = None

# =============================================================================
# 8. CÁLCULO DE EFEITOS MARGINAIS (LOGIT)
# =============================================================================

if logit_model is not None:
    print("\n" + "=" * 80)
    print("CALCULANDO AVERAGE MARGINAL EFFECTS (AMEs)...")
    print("=" * 80)

    try:
        logit_ames = logit_model.get_margeff(at='overall', method='dydx')
        print("[OK] AMEs calculados com sucesso")

        # Mostrar sumário
        print("\n" + "=" * 80)
        print("SUMARIO DOS AVERAGE MARGINAL EFFECTS")
        print("=" * 80)
        print(logit_ames.summary())
    except Exception as e:
        print(f"[ERRO] ao calcular AMEs: {e}")
        logit_ames = None
else:
    logit_ames = None

# =============================================================================
# 9. VISUALIZAÇÕES
# =============================================================================

# 9.1 Forest Plot (Logit AMEs)
if logit_ames is not None:
    print("\n" + "=" * 80)
    print("CRIANDO FOREST PLOT DOS EFEITOS MARGINAIS...")
    print("=" * 80)

    try:
        # Obter DataFrame dos AMEs
        ames_df = logit_ames.summary_frame()
        ames_df['Variable'] = ames_df.index

        # Filtrar para variáveis de Gestão e Severidade
        gestao_severidade_keywords = [
            'hospital de campanha',
            'tendas de triagem',
            'leitos foi ampliado',
            'internações ultrapassou',
            'referenciar',
            'manter pessoas',
            'Máscara',
            'Restrição ao comércio'
        ]

        # Filtrar linhas
        mask = ames_df['Variable'].apply(
            lambda x: any(keyword.lower() in x.lower() for keyword in gestao_severidade_keywords)
        )
        filtered_ames = ames_df[mask].copy()

        if len(filtered_ames) > 0:
            # Criar o plot
            fig, ax = plt.subplots(figsize=(12, len(filtered_ames) * 0.6 + 2))

            # Ordenar por efeito marginal
            filtered_ames = filtered_ames.sort_values('dy/dx')

            # Plot
            ax.errorbar(
                x=filtered_ames['dy/dx'],
                y=range(len(filtered_ames)),
                xerr=[
                    filtered_ames['dy/dx'] - filtered_ames['Conf. Int. Low'],
                    filtered_ames['Cont. Int. Hi.'] - filtered_ames['dy/dx']
                ],
                fmt='o',
                markersize=8,
                capsize=5,
                capthick=2,
                linewidth=2,
                color='steelblue'
            )

            # Linha vertical em x=0
            ax.axvline(0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)

            # Labels
            ax.set_yticks(range(len(filtered_ames)))

            # Truncar nomes longos
            labels = []
            for var in filtered_ames['Variable']:
                if len(var) > 60:
                    labels.append(var[:57] + '...')
                else:
                    labels.append(var)
            ax.set_yticklabels(labels)

            ax.set_xlabel('Average Marginal Effect (dy/dx)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Variáveis', fontsize=12, fontweight='bold')
            ax.set_title('Efeitos Marginais Médios - Variáveis de Gestão e Severidade\n(Modelo Logit)',
                        fontsize=14, fontweight='bold', pad=20)

            # Grid
            ax.grid(axis='x', alpha=0.3)

            plt.tight_layout()
            plt.savefig('logit_ames_forest_plot.png', dpi=300, bbox_inches='tight')
            print("[OK] Forest plot salvo: logit_ames_forest_plot.png")
            plt.close()
        else:
            print("[AVISO] Nenhuma variavel de gestao/severidade encontrada nos AMEs")

    except Exception as e:
        print(f"[ERRO] ao criar Forest Plot: {e}")

# 9.2 Gráfico de Probabilidades Previstas (NEC_2020)
if logit_model is not None:
    print("\n" + "=" * 80)
    print("CRIANDO GRÁFICO DE PROBABILIDADES PREVISTAS...")
    print("=" * 80)

    try:
        nec_col = 'Número efetivo de candidatos de 2020'

        if nec_col in df_clean.columns:
            # Range de valores
            nec_range = np.linspace(df_clean[nec_col].min(), df_clean[nec_col].max(), 100)

            # Criar DataFrame de predição
            predict_df = pd.DataFrame()

            # Para variáveis contínuas: usar a média
            continuous_vars = [
                'Dias da eleição - Máscara (Normalizado)',
                'Dias da eleição - Restrição ao comércio (Normalizado)',
                'delta_firjan_saude_16_20',
                'delta_firjan_emprego_16_20',
                'Vantagem do Incumbente no primeiro turno 2016',
                'ln_pop',
                'Produto Interno Bruto per capita,  a preços correntes (R$ 1,00)'
            ]

            for var in continuous_vars:
                if var in df_clean.columns:
                    predict_df[var] = [df_clean[var].mean()] * 100

            # Para variáveis binárias: usar a moda
            binary_vars = [
                'Foi instalado hospital de campanha durante a pandemia da COVID-19 no município',
                'Houve a instalação de tendas de triagem para o combate da COVID-19, no município',
                'O número de leitos foi ampliado para atender à demanda por internação no município em virtude da COVID-19',
                'O número de internações ultrapassou a capacidade de leitos e de unidades de tratamento intensivo (UTI) públicos ou privados con',
                'Nos casos de internação por COVID-19, houve necessidade de referenciar o(s) paciente(s) para outro município',
                'Durante o período da pandemia do COVID-19 foi necessário manter pessoas por mais de 24 horas em unidades sem internação',
                'Gênero'
            ]

            for var in binary_vars:
                if var in df_clean.columns:
                    predict_df[var] = [df_clean[var].mode()[0]] * 100

            # Para dummies de região: usar a moda
            for var in dummy_cols:
                if var in df_clean.columns:
                    predict_df[var] = [df_clean[var].mode()[0]] * 100

            # Adicionar a variável de interesse
            predict_df[nec_col] = nec_range

            # Calcular probabilidades previstas
            predicted_probs = logit_model.predict(predict_df)

            # Criar o gráfico
            fig, ax = plt.subplots(figsize=(10, 6))

            ax.plot(nec_range, predicted_probs, linewidth=2.5, color='darkblue')

            ax.set_xlabel('Número Efetivo de Candidatos (2020)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Probabilidade de Reeleição', fontsize=12, fontweight='bold')
            ax.set_title('Probabilidade Prevista de Reeleição vs. NEC 2020\n(Modelo Logit)',
                        fontsize=14, fontweight='bold', pad=20)

            ax.grid(alpha=0.3)
            ax.set_ylim([0, 1])

            plt.tight_layout()
            plt.savefig('logit_pred_prob_nec.png', dpi=300, bbox_inches='tight')
            print("[OK] Grafico de probabilidades salvo: logit_pred_prob_nec.png")
            plt.close()
        else:
            print(f"[AVISO] Coluna '{nec_col}' nao encontrada")

    except Exception as e:
        print(f"[ERRO] ao criar grafico de probabilidades: {e}")

# =============================================================================
# 10. OUTPUT DOS SUMÁRIOS
# =============================================================================

print("\n" + "=" * 80)
print("SUMÁRIOS DOS MODELOS")
print("=" * 80)

if logit_model is not None:
    print("\n" + "=" * 80)
    print("MODELO LOGIT (MODEL 3) - REELEICAO")
    print("=" * 80)
    print(logit_model.summary())
    print("\n" + "-" * 80)
    print("CRITERIOS DE INFORMACAO - MODELO LOGIT")
    print("-" * 80)
    print(f"Logit AIC: {logit_model.aic:.4f}")
    print(f"Logit BIC: {logit_model.bic:.4f}")

if ols_model is not None:
    print("\n" + "=" * 80)
    print("MODELO OLS (MODEL 6) - DELTA VANTAGEM")
    print("=" * 80)
    try:
        print(ols_model.summary())
    except Exception as e:
        print(f"[AVISO] Nao foi possivel gerar o sumario completo: {e}")
        print("\nCoeficientes:")
        print(ols_model.params)
        print("\nP-valores:")
        print(ols_model.pvalues)
        print(f"\nR-quadrado: {ols_model.rsquared:.4f}")
        print(f"N observacoes: {ols_model.nobs}")

    print("\n" + "-" * 80)
    print("CRITERIOS DE INFORMACAO - MODELO OLS")
    print("-" * 80)
    print(f"OLS AIC: {ols_model.aic:.4f}")
    print(f"OLS BIC: {ols_model.bic:.4f}")

print("\n" + "=" * 80)
print("ANÁLISE CONCLUÍDA!")
print("=" * 80)
print("\nArquivos gerados:")
print("  - logit_ames_forest_plot.png")
print("  - logit_pred_prob_nec.png")
