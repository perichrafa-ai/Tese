import pandas as pd
import numpy as np
import unicodedata

def normalize(text):
    nfd = unicodedata.normalize('NFD', str(text))
    return ''.join([ch for ch in nfd if unicodedata.category(ch) != 'Mn']).lower().strip()

def br_to_float(s, decimal=','):
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors='coerce')
    temp = s.astype(str).str.strip()
    if decimal == ',':
        temp = temp.str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
    return pd.to_numeric(temp, errors='coerce')

def to01(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    if s in ('sim', 'yes', '1', '1.0', 'true', 's'):
        return 1.0
    if s in ('não', 'nao', 'no', '0', '0.0', 'false', 'n'):
        return 0.0
    try:
        v = float(s)
        return 1.0 if v > 0.5 else 0.0
    except:
        return np.nan

def to_binary(series):
    return series.map(to01)

def find_by_keywords(df, *keywords):
    for col in df.columns:
        col_norm = normalize(str(col))
        if all(normalize(kw) in col_norm for kw in keywords):
            return col
    return None

df = pd.read_csv('Base_VALIDADA_E_PRONTA.csv', encoding='latin-1', sep=';', decimal=',')

print("=" * 60)
print("SIMULATING MODEL 27C PREPARATION")
print("=" * 60)

# Locate columns
col_delta_vant = find_by_keywords(df, 'delta', 'vantagem')
col_vantagem2016 = find_by_keywords(df, 'vantagem', '2016')
col_delta_comp = find_by_keywords(df, 'delta', 'nec')

print(f"DeltaVantagem: {col_delta_vant}")
print(f"Vantagem2016: {col_vantagem2016}")
print(f"DeltaCompeticao: {col_delta_comp}")

# Prepare variables
Y = br_to_float(df[col_delta_vant], ',')
Vantagem2016_raw = br_to_float(df[col_vantagem2016], ',')
DeltaComp_raw = br_to_float(df[col_delta_comp], ',')

# Center
Vantagem2016_c = Vantagem2016_raw - Vantagem2016_raw.mean()
DeltaComp_c = DeltaComp_raw - DeltaComp_raw.mean()

# Build DataFrame
df_work = pd.DataFrame({
    'DeltaVantagem': Y,
    'Vantagem2016_c': Vantagem2016_c,
    'DeltaCompeticao_c': DeltaComp_c,
    'V2016_x_DeltaComp': Vantagem2016_c * DeltaComp_c
})

print(f"\nAfter main variables: N={len(df_work.dropna())}")

# Add controls
controles_candidatos = ['ln_pop', 'pib_pc', 'genero', 'ifdm']
for ctrl in controles_candidatos:
    col = find_by_keywords(df, ctrl)
    if col:
        print(f"\nAdding control '{ctrl}': {col}")
        if ctrl == 'genero':
            gen_bin = to_binary(df[col])
            print(f"  Genero values before binary: {df[col].unique()}")
            print(f"  Genero after binary: {gen_bin.value_counts(dropna=False)}")
            print(f"  Missing after binary: {gen_bin.isna().sum()}")
            df_work[ctrl] = gen_bin
        else:
            val = br_to_float(df[col], ',')
            print(f"  Values (first 5): {val.head().values}")
            print(f"  Missing: {val.isna().sum()}")
            df_work[ctrl] = val

        print(f"  After adding '{ctrl}': N={len(df_work.dropna())}")

print(f"\n" + "=" * 60)
print(f"FINAL: N after dropna = {len(df_work.dropna())}")
print("=" * 60)

# Show which rows have NAs
print("\nMissing values per column:")
print(df_work.isna().sum())

print("\nRows with any NA:")
has_na = df_work.isna().any(axis=1)
print(f"Total rows with NA: {has_na.sum()}")

if has_na.sum() > 0 and has_na.sum() < 20:
    print("\nShowing rows with NAs:")
    print(df_work[has_na])
