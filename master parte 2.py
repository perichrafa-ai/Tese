#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
master_2_v5.py — Interações finais + 2 gráficos (com robustez e AMEs)
--------------------------------------------------------------------
Correções desta versão:
- **FIX** no erro `DiscreteMargins` (usa `summary_frame()` ao invés de `margeff_index`).
- Descoberta/Construção automática das variáveis a partir dos nomes do seu CSV.
- Gera APENAS as saídas desta seção (duas figuras + tabelas 3.4a–d).
- Sem títulos dentro das figuras (apenas eixos/legenda).

Como rodar (uma linha):
  py .\\master_2_v5.py --csv ".\\Base_VALIDADA_E_PRONTA.csv" --outdir ".\\output" --encoding latin-1 --sep ";" --decimal ","
"""
from __future__ import annotations

import argparse, logging, re, unicodedata
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# ------------------------------ CLI -----------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Interações finais + 2 gráficos (H5a e V2016×ΔNEC)')
    p.add_argument('--csv', type=Path, required=True)
    p.add_argument('--outdir', type=Path, default=Path('./output'))
    p.add_argument('--encoding', default='latin-1')
    p.add_argument('--sep', default=';')
    p.add_argument('--decimal', default=',')
    p.add_argument('--map', action='append', default=[], help='Mapeia nomes: CHAVE="nome exato no CSV" (repetível)')
    return p.parse_args()

args = parse_args()
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
log = logging.getLogger('master2-v5')

# --------------------------- Utils ------------------------------------

def normalize(text: str) -> str:
    if text is None: return ''
    t = unicodedata.normalize('NFKD', str(text))
    t = ''.join(ch for ch in t if not unicodedata.combining(ch))
    t = re.sub(r'[^0-9a-zA-Z]+', '', t)
    return t.lower()

def br_to_float(s: pd.Series, decimal: str=',') -> pd.Series:
    x = s.astype(str).str.strip()
    x = (x.str.replace('%','', regex=False)
           .str.replace('\u00A0','', regex=False)
           .str.replace(' ', '', regex=False))
    x = x.str.replace(r"\.(?=\d{3}(\D|$))", '', regex=True)
    if decimal != '.':
        x = x.str.replace(decimal, '.', regex=False)
    return pd.to_numeric(x, errors='coerce')

def to01(x):
    if pd.isna(x): return np.nan
    if isinstance(x,(int,float)): return float(x>0)
    s = str(x).strip().lower()
    if s in {'1','sim','yes','true','verdadeiro'}: return 1.0
    if s in {'0','nao','não','no','false','falso'}: return 0.0
    try:
        return float(float(s.replace(',', '.'))>0)
    except Exception:
        return np.nan

# procura coluna por keywords (todas precisam aparecer no nome normalizado)

def find_by_keywords(df: pd.DataFrame, *keywords: str) -> Optional[str]:
    keys = [normalize(k) for k in keywords if k]
    for c in df.columns:
        nc = normalize(c)
        if all(k in nc for k in keys):
            return c
    return None

# --------------------------- Leitura ----------------------------------

log.info(f'Lendo CSV: {args.csv.name}')
df = pd.read_csv(args.csv, encoding=args.encoding, sep=args.sep, decimal=args.decimal)
log.info(f'Dimensão: {df.shape[0]} × {df.shape[1]}')

outdir = args.outdir; outdir.mkdir(parents=True, exist_ok=True)

# ---------------------- Resolver colunas chave -------------------------

# Reeleição (usa "reeleito")
col_ree = find_by_keywords(df, 'reeleito') or find_by_keywords(df, 'reeleicao')
if col_ree is None:
    raise SystemExit('[ERRO] Não encontrei a coluna de reeleição (ex.: "reeleito").')
Reeleicao = df[col_ree].map(to01).astype(float)

# Vantagem 2016
col_v2016 = find_by_keywords(df, 'vantagem', '2016') or 'Vantagem do Incumbente no primeiro turno 2016'
if col_v2016 not in df.columns:
    col_v2016 = find_by_keywords(df, 'vantagemdoincumbentenoprimeiroturno2016')
if col_v2016 is None:
    raise SystemExit('[ERRO] Não encontrei Vantagem 2016.')
V2016 = br_to_float(df[col_v2016], args.decimal)

# Δ Vantagem (2020-2016)
col_dv = find_by_keywords(df, 'delta', 'vantagem') or 'delta_vantagem (2020-2016)'
if col_dv not in df.columns:
    col_dv = find_by_keywords(df, 'deltavantagem20202016')
if col_dv is None:
    raise SystemExit('[ERRO] Não encontrei delta_vantagem (2020-2016).')
DeltaV = br_to_float(df[col_dv], args.decimal)

# Abstenção 2016/2020 → Δ Abstenção (p.p.)
col_abs16 = find_by_keywords(df, 'abstencao','2016')
col_abs20 = find_by_keywords(df, 'abstencao','2020')
if col_abs16 is None or col_abs20 is None:
    raise SystemExit('[ERRO] Não encontrei Abstenção 2016/2020 para calcular Δ Abstenção.')
DeltaAbst = br_to_float(df[col_abs20], args.decimal) - br_to_float(df[col_abs16], args.decimal)

# Δ Competição (usa NEC 2016/2020; também aceita delta_nec)
col_nec16 = find_by_keywords(df, 'numero efetivo de candidatos', '2016')
col_nec20 = find_by_keywords(df, 'numero efetivo de candidatos', '2020')
col_delta_nec = find_by_keywords(df, 'delta_nec')
if col_nec16 is not None and col_nec20 is not None:
    nec16 = br_to_float(df[col_nec16], args.decimal)
    nec20 = br_to_float(df[col_nec20], args.decimal)
    DNEC_log  = np.log(nec20) - np.log(nec16)
    DNEC_diff = nec20 - nec16
elif col_delta_nec is not None:
    dd = br_to_float(df[col_delta_nec], args.decimal)
    # diff é direto; log: aproximação via NEC_2016/2020 não disponível → usa diff como proxy centrado
    DNEC_diff = dd
    DNEC_log  = dd - dd.mean()
else:
    raise SystemExit('[ERRO] Não encontrei NEC 2016/2020 nem delta_nec (2020-2016).')

# Índice de Gestão (compõe de colunas disponíveis)
parts = []
used = []
# binárias
bin_defs = [
    ('Obrigatoriedade de máscara', ['obrigatoriedade','mascara']),
    ('Hospital de campanha', ['hospital','campanha']),
    ('Tendas de triagem', ['tendas','triagem']),
    ('Leitos ampliados', ['leitos','ampliado']),
    ('Testagem (PCR/Sorológico)', ['condicoes','testes','pcr']),
]
for label, keys in bin_defs:
    c = find_by_keywords(df, *keys)
    if c:
        v = df[c].map(to01).astype(float)
        parts.append(v); used.append(f'{label} ← {c}')
# contínuas normalizadas
cont_defs = [
    ('Dias máscara (norm.)', ['dias','eleicao','mascara','normalizado']),
    ('Dias comércio (norm.)', ['dias','eleicao','restricao','comercio','normalizado']),
]
for label, keys in cont_defs:
    c = find_by_keywords(df, *keys)
    if c:
        v = br_to_float(df[c], args.decimal)
        vr = v.copy()
        if pd.notna(vr).sum()>0 and (vr.max() or 0)>1:
            vr = (vr - vr.min())/(vr.max()-vr.min())
        parts.append(vr); used.append(f'{label} ← {c}')

if not parts:
    raise SystemExit('[ERRO] Nenhum componente para o Índice de Gestão foi encontrado (máscara, comércio, leitos, tendas, hospital, testagem).')
Gestao_Index = pd.concat(parts, axis=1).mean(axis=1, skipna=True)

# Centragem
Gestao_c    = Gestao_Index - Gestao_Index.mean()
DeltaAbst_c = DeltaAbst - DeltaAbst.mean()
V2016_c     = V2016 - V2016.mean()
DNEC_log_c  = DNEC_log - DNEC_log.mean()

# UF e Pop para TOP5 (robustez)
col_uf = find_by_keywords(df, 'estado') or find_by_keywords(df, 'uf')
UF = df[col_uf].astype(str) if col_uf else pd.Series(['ALL']*len(df))
col_pop = find_by_keywords(df, 'populacao')
if col_pop is None:
    col_ln = find_by_keywords(df, 'ln_pop')
    POP = np.exp(br_to_float(df[col_ln], args.decimal)) if col_ln else pd.Series(np.nan, index=df.index)
else:
    POP = br_to_float(df[col_pop], args.decimal)

# ------------------------------ MODELOS --------------------------------

# LOGIT: Reeleição ~ Gestão_c * ΔAbst_c
Xlog = pd.DataFrame({
    'const': 1.0,
    'Gestao_Index_c': Gestao_c,
    'delta_abstencao_pp_c': DeltaAbst_c,
    'Gestao_Index_c:delta_abstencao_pp_c': Gestao_c*DeltaAbst_c,
})
maskL = Xlog.notna().all(axis=1) & Reeleicao.notna()
res_logit = sm.Logit(Reeleicao[maskL], Xlog[maskL]).fit(disp=False, cov_type='HC1')

# OLS: ΔVantagem ~ V2016_c * ΔNEC_log_c
Xols = pd.DataFrame({
    'const': 1.0,
    'vantagem2016_c': V2016_c,
    'delta_competicao_log_c': DNEC_log_c,
    'vantagem2016_c:delta_competicao_log_c': V2016_c*DNEC_log_c,
})
maskO = Xols.notna().all(axis=1) & DeltaV.notna()
res_ols = sm.OLS(DeltaV[maskO], Xols[maskO]).fit(cov_type='HC1')

# ----------------------------- TABELAS ---------------------------------

outdir.mkdir(parents=True, exist_ok=True)

# 3.4a — coeficientes do LOGIT
T34a = pd.DataFrame({
    'variavel': res_logit.params.index,
    'coef': res_logit.params.values,
    'se': res_logit.bse.values,
    't_z': res_logit.tvalues,
    'pvalue': res_logit.pvalues,
})
T34a.to_csv(outdir/'Tabela_3_4a_logit_gestao_x_deltaabst.csv', index=False, encoding='utf-8')

# 3.4b — AMEs (usa summary_frame para ser compatível com versões do statsmodels)
try:
    mfx = res_logit.get_margeff(at='overall', method='dydx')
    sf = mfx.summary_frame()
    # renomear colunas para o formato do texto (tolerante a variantes)
    rename_map = {
        'dy/dx': 'dy/dx',
        'Std. Err.': 'Std, Err,',
        'Std. err': 'Std, Err,',
        'z': 'z',
        'P>|z|': 'Pr(>|z|)',
        '[0.025': 'Conf, Int, Low',
        '0.975]': 'Cont, Int, Hi,'
    }
    cols_new = {c: rename_map.get(c, c) for c in sf.columns}
    T34b = sf.rename(columns=cols_new).reset_index().rename(columns={'index':'Unnamed: 0'})
    # garantir apenas as colunas esperadas
    keep = ['Unnamed: 0','dy/dx','Std, Err,','z','Pr(>|z|)','Conf, Int, Low','Cont, Int, Hi,']
    for k in keep:
        if k not in T34b.columns:
            T34b[k] = np.nan
    T34b = T34b[keep]
    T34b.to_csv(outdir/'Tabela_3_4b_marginais_logit_gestao_x_deltaabst.csv', index=False, encoding='utf-8')
except Exception as e:
    log.warning(f'AMEs não exportadas (prosseguindo mesmo assim): {e}')

# 3.4c — OLS (coeficientes)
T34c = pd.DataFrame({
    'variavel': res_ols.params.index,
    'coef': res_ols.params.values,
    'se': res_ols.bse.values,
    't_z': res_ols.tvalues,
    'pvalue': res_ols.pvalues,
})
T34c.to_csv(outdir/'Tabela_3_4c_ols_v2016_x_dnec_log.csv', index=False, encoding='utf-8')

# 3.4d — Robustez (FULL/TOP5 × log/diff × HC1/cluster-UF)
UF_series = UF
# Seleção TOP5 por UF
if POP.isna().all():
    top5_idx = df.index  # se faltar população, não restringe
else:
    top5_idx = (df.assign(__ord__=POP)
                  .sort_values([col_uf if col_uf else '__ord__','__ord__'], ascending=[True, False])
                  .groupby(UF_series, group_keys=False).head(5).index)

def fit_take(Y, v, d, groups=None, cov_type='HC1'):
    X = pd.DataFrame({'const':1.0,'vantagem2016_c':v,'delta_comp':d,'vantagem2016_c:delta_comp':v*d})
    m = X.notna().all(axis=1) & Y.notna()
    if groups is not None:
        res = sm.OLS(Y[m], X[m]).fit(cov_type='cluster', cov_kwds={'groups': groups[m]})
    else:
        res = sm.OLS(Y[m], X[m]).fit(cov_type=cov_type)
    par = 'vantagem2016_c:delta_comp'
    return int(res.nobs), float(res.params[par]), float(res.bse[par]), float(res.pvalues[par])

rows = []
# FULL_log_clusterUF
n,b,se,p = fit_take(DeltaV, V2016_c, DNEC_log_c, groups=UF_series); rows.append({'modelo':'FULL_log_clusterUF','comp_col':'delta_competicao_log_c','N':n,'coef_interacao':b,'se':se,'pvalue':p})
# FULL_log_HC1
n,b,se,p = fit_take(DeltaV, V2016_c, DNEC_log_c); rows.append({'modelo':'FULL_log_HC1','comp_col':'delta_competicao_log_c','N':n,'coef_interacao':b,'se':se,'pvalue':p})

# versões diff se disponíveis (usa DNEC_diff calculado acima)
if 'DNEC_diff' in locals() and DNEC_diff is not None:
    DNEC_diff_c = DNEC_diff - DNEC_diff.mean()
    n,b,se,p = fit_take(DeltaV, V2016_c, DNEC_diff_c, groups=UF_series); rows.append({'modelo':'FULL_diff_clusterUF','comp_col':'delta_competicao_diff_c','N':n,'coef_interacao':b,'se':se,'pvalue':p})
    n,b,se,p = fit_take(DeltaV, V2016_c, DNEC_diff_c); rows.append({'modelo':'FULL_diff_HC1','comp_col':'delta_competicao_diff_c','N':n,'coef_interacao':b,'se':se,'pvalue':p})

mask_top = df.index.isin(top5_idx)
# TOP5
n,b,se,p = fit_take(DeltaV[mask_top], V2016_c[mask_top], DNEC_log_c[mask_top], groups=UF_series[mask_top]); rows.append({'modelo':'TOP5_log_clusterUF','comp_col':'delta_competicao_log_c','N':n,'coef_interacao':b,'se':se,'pvalue':p})
n,b,se,p = fit_take(DeltaV[mask_top], V2016_c[mask_top], DNEC_log_c[mask_top]); rows.append({'modelo':'TOP5_log_HC1','comp_col':'delta_competicao_log_c','N':n,'coef_interacao':b,'se':se,'pvalue':p})

if 'DNEC_diff_c' in locals():
    n,b,se,p = fit_take(DeltaV[mask_top], V2016_c[mask_top], DNEC_diff_c[mask_top], groups=UF_series[mask_top]); rows.append({'modelo':'TOP5_diff_clusterUF','comp_col':'delta_competicao_diff_c','N':n,'coef_interacao':b,'se':se,'pvalue':p})
    n,b,se,p = fit_take(DeltaV[mask_top], V2016_c[mask_top], DNEC_diff_c[mask_top]); rows.append({'modelo':'TOP5_diff_HC1','comp_col':'delta_competicao_diff_c','N':n,'coef_interacao':b,'se':se,'pvalue':p})

pd.DataFrame(rows).to_csv(outdir/'Tabela_3_4d_robustez_competicao.csv', index=False, encoding='utf-8')

# ------------------------------ FIGURAS --------------------------------

sns.set_theme(style='whitegrid')
plt.rcParams['axes.unicode_minus'] = False

# Figura 3.x — Reeleição: Gestão × Δ Abstenção (prob. predita)
fig, ax = plt.subplots(figsize=(10,5.6))
xgrid = np.linspace(float(DeltaAbst_c.min()), float(DeltaAbst_c.max()), 101)
G_levels = [Gestao_c.mean()+Gestao_Index.std(), Gestao_c.mean()-Gestao_Index.std(), Gestao_c.mean()]
labels = ['Gestão +1dp','Gestão -1dp','Gestão média']
for g, lab in zip(G_levels, labels):
    Xg = pd.DataFrame({'const':1.0,'Gestao_Index_c':g,'delta_abstencao_pp_c':xgrid,'Gestao_Index_c:delta_abstencao_pp_c':g*xgrid})
    pr = res_logit.predict(Xg)
    ax.plot(xgrid, pr, label=lab)
ax.set_xlabel('delta'); ax.set_ylabel('pr'); ax.legend(); fig.tight_layout()
fig.savefig(outdir/'fig_pred_logit_gestao_x_deltaabst.png'); plt.close(fig)

# Figura 3.y — ΔVantagem ~ V2016 × ΔCompetição (linha de predição)
fig, ax = plt.subplots(figsize=(9.6,5.2))
xgrid_c = np.linspace(float(DNEC_log_c.min()), float(DNEC_log_c.max()), 101)
V_levels = [V2016_c.mean()+V2016.std(), V2016_c.mean()-V2016.std(), V2016_c.mean()]
labels = ['Vantagem +1dp','Vantagem -1dp','Vantagem média']
for v, lab in zip(V_levels, labels):
    Xg = pd.DataFrame({'const':1.0,'vantagem2016_c':v,'delta_competicao_log_c':xgrid_c,'vantagem2016_c:delta_competicao_log_c':v*xgrid_c})
    yhat = res_ols.predict(Xg)
    ax.plot(xgrid_c, yhat, label=lab)
ax.set_xlabel('comp'); ax.set_ylabel('yhat'); ax.legend(); fig.tight_layout()
fig.savefig(outdir/'fig_pred_ols_v2016_x_dnec_log.png'); plt.close(fig)

print('[OK] Saídas em', outdir.resolve())
print(' - Tabela_3_4a_logit_gestao_x_deltaabst.csv')
print(' - Tabela_3_4b_marginais_logit_gestao_x_deltaabst.csv')
print(' - fig_pred_logit_gestao_x_deltaabst.png')
print(' - Tabela_3_4c_ols_v2016_x_dnec_log.csv')
print(' - fig_pred_ols_v2016_x_dnec_log.png')
print(' - Tabela_3_4d_robustez_competicao.csv')
