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

def find_by_keywords(df, *keywords):
    for col in df.columns:
        col_norm = normalize(str(col))
        if all(normalize(kw) in col_norm for kw in keywords):
            return col
    return None

df = pd.read_csv('Base_VALIDADA_E_PRONTA.csv', encoding='latin-1', sep=';', decimal=',')

print("=" * 60)
print("CHECKING MODEL 27C VARIABLES")
print("=" * 60)

# Check what columns would be used for Model 27C
col_delta_vant = find_by_keywords(df, 'delta', 'vantagem') or find_by_keywords(df, 'delta_vantagem')
col_vantagem2016 = find_by_keywords(df, 'vantagem', '2016')
col_delta_comp = find_by_keywords(df, 'delta', 'nec')
col_ln_pop = find_by_keywords(df, 'ln_pop')
col_genero = find_by_keywords(df, 'genero')
col_ifdm = find_by_keywords(df, 'ifdm', '2016')

print(f"delta_vantagem: {col_delta_vant}")
print(f"vantagem2016: {col_vantagem2016}")
print(f"delta_nec: {col_delta_comp}")
print(f"ln_pop: {col_ln_pop}")
print(f"genero: {col_genero}")
print(f"ifdm 2016: {col_ifdm}")

print("\n" + "=" * 60)
print("MISSING VALUES CHECK")
print("=" * 60)

if col_delta_vant:
    Y = br_to_float(df[col_delta_vant], ',')
    print(f"DeltaVantagem: {Y.isna().sum()} missing / {len(Y)} total")

if col_vantagem2016:
    V2016 = br_to_float(df[col_vantagem2016], ',')
    print(f"Vantagem2016: {V2016.isna().sum()} missing / {len(V2016)} total")

if col_delta_comp:
    DComp = br_to_float(df[col_delta_comp], ',')
    print(f"DeltaNEC: {DComp.isna().sum()} missing / {len(DComp)} total")

if col_ln_pop:
    LnPop = br_to_float(df[col_ln_pop], ',')
    print(f"ln_pop: {LnPop.isna().sum()} missing / {len(LnPop)} total")

if col_genero:
    print(f"genero: {df[col_genero].isna().sum()} missing / {len(df)} total")
    print(f"  Unique values: {df[col_genero].unique()}")

if col_ifdm:
    IFDM = br_to_float(df[col_ifdm], ',')
    print(f"IFDM 2016: {IFDM.isna().sum()} missing / {len(IFDM)} total")

print("\n" + "=" * 60)
print("CHECKING ABSTENCAO FOR MODEL 27B")
print("=" * 60)

# Check abstencao columns
col_abst_2016 = find_by_keywords(df, 'abstencao', '2016')
col_abst_2020 = find_by_keywords(df, 'abstencao', '2020')

print(f"Abstencao 2016: {col_abst_2016}")
print(f"Abstencao 2020: {col_abst_2020}")

if col_abst_2016 and col_abst_2020:
    Abst2016 = br_to_float(df[col_abst_2016], ',')
    Abst2020 = br_to_float(df[col_abst_2020], ',')
    DeltaAbst = Abst2020 - Abst2016
    print(f"Abstencao 2016: {Abst2016.isna().sum()} missing")
    print(f"Abstencao 2020: {Abst2020.isna().sum()} missing")
    print(f"Delta Abstencao: {DeltaAbst.isna().sum()} missing")
