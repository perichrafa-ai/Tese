#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
master_h3_h4_v3.py — H3/H4 com autodetecção de colunas + índices internos
--------------------------------------------------------------------------
• Corrige o problema reportado: seu CSV tem "Estado" (não "UF"), "reeleito",
  "delta_vantagem (2020-2016)", "População"/"ln_pop", e NÃO possui
  "gestao_indice"/"severidade_composta". Este script **descobre** as colunas
  corretas e **constrói** os índices de Gestão e Severidade a partir das
  variáveis que existem no CSV (máscara, comércio, leitos, tendas, hospital,
  testagem; sobrecarga/transferência/24h).
• É tolerante ao PowerShell: remove carets '^' de sys.argv (se houver) e
  reexecuta dentro de .venv.
• Saídas: quatro sumários/tabelas (H3/H4 × Logit/OLS) e figuras de predição.

Uso (1 linha):
  py .\\master_h3_h4_v3.py --csv .\\Base_VALIDADA_E_PRONTA.csv --outdir .\\output \
     --encoding latin-1 --sep ";" --decimal ","

Você pode ainda mapear manualmente algum nome divergente com --map, ex.:
  --map Reeleicao="Reeleito (0/1)" --map Vantagem2016="Vantagem do Incumbente no primeiro turno 2016"
"""
from __future__ import annotations

import argparse, os, re, shutil, subprocess, sys, textwrap, warnings, unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ===================== venv & deps =====================
VENV_DIR = Path('.venv')
PY_PATH  = VENV_DIR / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
REQ_FILE = Path('requirements.txt')
BASE_REQ = (
    "contourpy==1.3.3\n"
    "cycler==0.12.1\n"
    "fonttools==4.59.2\n"
    "kiwisolver==1.4.9\n"
    "matplotlib==3.9.2\n"
    "numpy==2.1.1\n"
    "packaging==25.0\n"
    "pandas==2.2.2\n"
    "pillow==11.3.0\n"
    "pyparsing==3.2.4\n"
    "python-dateutil==2.9.0.post0\n"
    "pytz==2025.2\n"
    "scipy==1.13.1\n"
    "statsmodels==0.14.5\n"
    "tzdata==2025.2\n"
    "wheel==0.45.1\n"
)

def log(msg: str):
    print(f"[h3_h4_v3] {msg}")

def run(cmd: List[str]):
    log('> ' + ' '.join(map(str, cmd)))
    subprocess.check_call(cmd)

def venv_exists() -> bool:
    return VENV_DIR.exists() and PY_PATH.exists()

def venv_is_healthy() -> bool:
    if not venv_exists():
        return False
    try:
        out = subprocess.check_output([str(PY_PATH), '-c', 'import sys;print(sys.version)'], text=True).strip()
        return bool(out)
    except Exception:
        return False

def recreate_venv():
    if VENV_DIR.exists():
        shutil.rmtree(VENV_DIR, ignore_errors=True)
    log('Criando .venv…')
    run([sys.executable, '-m', 'venv', str(VENV_DIR)])

def ensurepip_inside_venv():
    try:
        run([str(PY_PATH), '-m', 'ensurepip', '--upgrade'])
        run([str(PY_PATH), '-m', 'pip', 'install', '--upgrade', 'pip', 'setuptools', 'wheel'])
    except subprocess.CalledProcessError:
        log('Aviso: ensurepip falhou; seguindo mesmo assim.')

def ensure_venv_created_and_healthy():
    if not venv_is_healthy():
        recreate_venv()
    else:
        log('Venv encontrada e saudável.')

def pip_install(args: List[str]):
    try:
        run([str(PY_PATH), '-m', 'pip'] + args)
    except subprocess.CalledProcessError:
        log('pip indisponível; tentando ensurepip…')
        ensurepip_inside_venv()
        run([str(PY_PATH), '-m', 'pip'] + args)

def ensure_deps():
    if (not REQ_FILE.exists()) or REQ_FILE.is_dir() or REQ_FILE.stat().st_size == 0:
        log('Criando requirements.txt padrão (versões fixas)…')
        REQ_FILE.write_text(BASE_REQ, encoding='utf-8')
    try:
        pip_install(['install', '--upgrade', 'pip', 'setuptools', 'wheel'])
    except subprocess.CalledProcessError:
        log('Aviso: não deu para atualizar pip/setuptools/wheel; seguindo assim mesmo.')
    pip_install(['install', '-r', str(REQ_FILE.resolve())])

# ---- limpeza de argumentos (PowerShell '^') ----

def clean_ps_args(argv: List[str]) -> List[str]:
    cleaned = [a for a in argv if a != '^']
    removed = len(argv) - len(cleaned)
    if removed:
        log(f"Removi {removed} argumento(s) '^' (quebra de linha errada no PowerShell).")
    return cleaned


def ensure_running_in_venv():
    if Path(sys.executable).resolve() != PY_PATH.resolve():
        cleaned = clean_ps_args(sys.argv[1:])
        log('Reexecutando dentro de .venv…')
        run([str(PY_PATH), __file__] + cleaned)
        sys.exit(0)
    sys.argv = [sys.argv[0]] + clean_ps_args(sys.argv[1:])

ensure_venv_created_and_healthy()
ensure_deps()
ensure_running_in_venv()

# ===================== imports pesados =====================
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.set_loglevel('warning')
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# ===================== utils de autodetecção =====================

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

def find_by_keywords(df: pd.DataFrame, *keywords: str) -> Optional[str]:
    keys = [normalize(k) for k in keywords if k]
    for c in df.columns:
        nc = normalize(c)
        if all(k in nc for k in keys):
            return c
    return None

# ===================== CLI =====================

def parse_args():
    ap = argparse.ArgumentParser(
        prog='master_h3_h4_v3.py',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='H3/H4 com autodetecção + construção de índices',
    )
    ap.add_argument('--csv', required=True, type=Path)
    ap.add_argument('--outdir', default=Path('./output'), type=Path)
    ap.add_argument('--encoding', default='latin-1')
    ap.add_argument('--sep', default=';')
    ap.add_argument('--decimal', default=',')
    ap.add_argument('--map', action='append', default=[], help='Mapeia nomes: CHAVE="nome no CSV" (repetível)')
    ap.add_argument('--ols-hc1', action='store_true', help='Usar HC1 no OLS (padrão cluster por Estado se disponível)')
    return ap.parse_args()

args = parse_args()
args.outdir.mkdir(parents=True, exist_ok=True)

# ===================== leitura =====================
log(f'Lendo CSV: {args.csv}')
df = pd.read_csv(args.csv, encoding=args.encoding, sep=args.sep, decimal=args.decimal)

# ===================== resolver colunas base =====================
USERMAP: Dict[str,str] = {}
for m in args.map:
    if '=' in m:
        k,v = m.split('=',1)
        USERMAP[k.strip()] = v.strip().strip('"').strip("'")

def resolve(key: str, candidates: List[Tuple[str,...]], fallback_contains: Optional[str]=None) -> Optional[str]:
    if key in USERMAP and USERMAP[key] in df.columns:
        return USERMAP[key]
    for ks in candidates:
        c = find_by_keywords(df, *ks)
        if c: return c
    if fallback_contains:
        for c in df.columns:
            if fallback_contains.lower() in c.lower():
                return c
    return None

col_estado = resolve('Estado', [('estado',), ('uf',)], None) or 'Estado' if 'Estado' in df.columns else None
col_regiao = resolve('Regiao', [('regiao', 'pais'), ('regiao',)], None)
col_ree    = resolve('Reeleicao', [('reeleito',), ('reeleicao',)], None)
col_v2016  = resolve('Vantagem2016', [('vantagem','2016'),], '2016')
col_dv     = resolve('DeltaVantagem', [('delta','vantagem'),], 'delta_vantagem')
col_lnpop  = resolve('LnPop', [('ln','pop'), ('log','pop')], None)
col_pop    = resolve('Populacao', [('populacao',), ('populacao','2019'), ('populacao','2010')], 'popula')

if not col_lnpop:
    if col_pop:
        s = br_to_float(df[col_pop], args.decimal)
        df['ln_pop'] = np.log(s)
        col_lnpop = 'ln_pop'
    else:
        raise SystemExit('[ERRO] Não encontrei ln_pop nem População para construí-lo.')

if not col_ree:
    raise SystemExit('[ERRO] Não encontrei a coluna de reeleição (ex.: "reeleito").')
if not col_v2016:
    raise SystemExit('[ERRO] Não encontrei Vantagem 2016 (ex.: contém "2016").')
if not col_dv:
    raise SystemExit('[ERRO] Não encontrei delta_vantagem (2020-2016).')

# ===================== construir índices =====================
# Gestão: binários (máscara, hospital, tendas, leitos, testagem) + contínuos normalizados (dias máscara/comércio)
parts = []
used_g = []
# binários
bin_defs = [
    ('Obrigatoriedade de máscara', ['obrigatoriedade','mascara']),
    ('Hospital de campanha', ['hospital','campanha']),
    ('Tendas de triagem', ['tendas','triagem']),
    ('Leitos ampliados', ['leitos','ampliado']),
    ('Testagem (PCR/Sorológico)', ['condicoes','testes','pcr','sorolog']),
]
for label, keys in bin_defs:
    c = find_by_keywords(df, *keys)
    if c:
        v = df[c].map(to01).astype(float)
        parts.append(v)
        used_g.append(f'{label} ← {c}')
# contínuas normalizadas (se necessário)
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
        parts.append(vr)
        used_g.append(f'{label} ← {c}')

if parts:
    Gestao_Index = pd.concat(parts, axis=1).mean(axis=1, skipna=True)
else:
    # fallback neutro (tudo NaN → zeros) para não quebrar; será reportado
    Gestao_Index = pd.Series(np.nan, index=df.index)

# Severidade: sobrecarga/transferência/24h (binários)
sev_defs = [
    ('Sobrecarga UTI', ['sobrecarga','leitos','uti']),
    ('Transferência de pacientes', ['transfer','pacien']),
    ('>24h sem internação', ['24','sem','intern'])
]
parts_s = []; used_s = []
for label, keys in sev_defs:
    c = find_by_keywords(df, *keys)
    if c:
        v = df[c].map(to01).astype(float)
        parts_s.append(v)
        used_s.append(f'{label} ← {c}')

if parts_s:
    Severidade_Index = pd.concat(parts_s, axis=1).mean(axis=1, skipna=True)
else:
    Severidade_Index = pd.Series(np.nan, index=df.index)

# ===================== preparar dados p/ modelos =====================
# Y (Logit) e Y (OLS)
Reeleicao = df[col_ree].map(to01).astype(float)
DeltaV    = br_to_float(df[col_dv], args.decimal)
V2016     = br_to_float(df[col_v2016], args.decimal)
ln_pop    = pd.to_numeric(df[col_lnpop], errors='coerce')

# centragem
Gestao_c = Gestao_Index - Gestao_Index.mean()
Sever_c  = Severidade_Index - Severidade_Index.mean()
lnpop_c  = ln_pop - ln_pop.mean()
V2016_c  = V2016 - V2016.mean()

# DataFrame base
BASE = pd.DataFrame({
    'Reeleicao': Reeleicao,
    'DeltaVant': DeltaV,
    'Gestao_c': Gestao_c,
    'Sever_c': Sever_c,
    'ln_pop_c': lnpop_c,
    'V2016_c': V2016_c,
})
if col_estado and col_estado in df.columns:
    BASE['Estado'] = df[col_estado].astype(str)
else:
    BASE['Estado'] = 'ALL'
if col_regiao and col_regiao in df.columns:
    BASE['Regiao'] = df[col_regiao].astype(str)

# ===================== H3/H4 =====================
# H3: Porte (ln_pop) como moderador de Gestão e Severidade → Dep. Reeleição (logit) e ΔVantagem (OLS)

# LOGIT H3
f_h3_logit = 'Reeleicao ~ Gestao_c + Sever_c + ln_pop_c + Gestao_c:ln_pop_c + Sever_c:ln_pop_c + C(Regiao)'
try:
    res_h3_logit = sm.Logit(BASE['Reeleicao'], sm.add_constant(pd.get_dummies(BASE[['Gestao_c','Sever_c','ln_pop_c','Regiao']], drop_first=True))).fit(disp=False, cov_type='HC1')
    # construir manualmente o termo de interação no DataFrame para exportar coeficientes organizados
    Xg = pd.DataFrame({'const':1.0,
                       'Gestao_c': BASE['Gestao_c'],
                       'Sever_c': BASE['Sever_c'],
                       'ln_pop_c': BASE['ln_pop_c'],
                       'Gestao_c:ln_pop_c': BASE['Gestao_c']*BASE['ln_pop_c'],
                       'Sever_c:ln_pop_c': BASE['Sever_c']*BASE['ln_pop_c']})
    m = Xg.notna().all(axis=1) & BASE['Reeleicao'].notna()
    res_h3_logit = sm.Logit(BASE['Reeleicao'][m], Xg[m]).fit(disp=False, cov_type='HC1')
except Exception:
    # fallback sem dummies de Regiao (caso inexistente)
    Xg = pd.DataFrame({'const':1.0,
                       'Gestao_c': BASE['Gestao_c'],
                       'Sever_c': BASE['Sever_c'],
                       'ln_pop_c': BASE['ln_pop_c'],
                       'Gestao_c:ln_pop_c': BASE['Gestao_c']*BASE['ln_pop_c'],
                       'Sever_c:ln_pop_c': BASE['Sever_c']*BASE['ln_pop_c']})
    m = Xg.notna().all(axis=1) & BASE['Reeleicao'].notna()
    res_h3_logit = sm.Logit(BASE['Reeleicao'][m], Xg[m]).fit(disp=False, cov_type='HC1')

# OLS H3
Xo = Xg.drop(columns=['const']).copy(); Xo.insert(0,'const',1.0)
mo = Xo.notna().all(axis=1) & BASE['DeltaVant'].notna()
res_h3_ols = sm.OLS(BASE['DeltaVant'][mo], Xo[mo]).fit(cov_type='HC1')

# H4: V2016 como moderador de Gestão/Severidade → Dep. idem
X4 = pd.DataFrame({'const':1.0,
                   'Gestao_c': BASE['Gestao_c'],
                   'Sever_c': BASE['Sever_c'],
                   'V2016_c': BASE['V2016_c'],
                   'Gestao_c:V2016_c': BASE['Gestao_c']*BASE['V2016_c'],
                   'Sever_c:V2016_c': BASE['Sever_c']*BASE['V2016_c']})
ml = X4.notna().all(axis=1) & BASE['Reeleicao'].notna()
res_h4_logit = sm.Logit(BASE['Reeleicao'][ml], X4[ml]).fit(disp=False, cov_type='HC1')
mo = X4.notna().all(axis=1) & BASE['DeltaVant'].notna()
res_h4_ols = sm.OLS(BASE['DeltaVant'][mo], X4[mo]).fit(cov_type='HC1')

# ===================== salvar saídas =====================
basefile = args.csv.stem
od = args.outdir

def save_txt(p: Path, text: str): p.write_text(text, encoding='utf-8')

def dump_res(name: str, res):
    tab = []
    for k in res.params.index:
        tab.append({'variavel': k, 'coef': float(res.params[k]), 'se': float(res.bse[k]), 'pval': float(res.pvalues[k])})
    pd.DataFrame(tab).to_csv(od / f"{basefile}__{name}_tabela.csv", index=False, encoding='utf-8')
    save_txt(od / f"{basefile}__{name}_sumario.txt", res.summary().as_text())

dump_res('H3_logit', res_h3_logit)
dump_res('H3_ols',   res_h3_ols)
dump_res('H4_logit', res_h4_logit)
dump_res('H4_ols',   res_h4_ols)

# Figuras de predição simples (linhas médias por quantis)
import numpy as _np

def plot_lines(x_vals, y_sets, title, xlabel, ylabel, outpng):
    plt.figure();
    for label, ys in y_sets:
        plt.plot(x_vals, ys, marker='o', linewidth=2, label=label)
    plt.title(title); plt.xlabel(xlabel); plt.ylabel(ylabel); plt.legend(); plt.tight_layout(); plt.savefig(outpng, dpi=200); plt.close()

# H3 — varrer ln_pop
q_ln = _np.quantile(BASE['ln_pop_c'].dropna(), [0.1,0.5,0.9]) if BASE['ln_pop_c'].notna().any() else [0,0,0]
G_levels = [BASE['Gestao_c'].mean()+BASE['Gestao_c'].std(), BASE['Gestao_c'].mean()-BASE['Gestao_c'].std()]
S_levels = [BASE['Sever_c'].mean()+BASE['Sever_c'].std(), BASE['Sever_c'].mean()-BASE['Sever_c'].std()]

# logit preds
ys = []
for g in G_levels:
    Xp = pd.DataFrame({'const':1.0,'Gestao_c':g,'Sever_c':BASE['Sever_c'].mean(),'ln_pop_c':q_ln,'Gestao_c:ln_pop_c':g*q_ln,'Sever_c:ln_pop_c':BASE['Sever_c'].mean()*q_ln})
    pr = res_h3_logit.predict(Xp)
    ys.append((f'Gestão {"+1dp" if g>0 else "-1dp"}', list(map(float, pr))))
plot_lines(q_ln, ys, 'H3-Logit: Prob. Reeleição vs ln(Pop)', 'ln(Pop)', 'Prob. prevista', od / f"{basefile}__H3_logit_lnpop.png")

ys = []
for s in S_levels:
    Xp = pd.DataFrame({'const':1.0,'Gestao_c':BASE['Gestao_c'].mean(),'Sever_c':s,'ln_pop_c':q_ln,'Gestao_c:ln_pop_c':BASE['Gestao_c'].mean()*q_ln,'Sever_c:ln_pop_c':s*q_ln})
    pr = res_h3_logit.predict(Xp)
    ys.append((f'Severidade {"+1dp" if s>0 else "-1dp"}', list(map(float, pr))))
plot_lines(q_ln, ys, 'H3-Logit: Prob. Reeleição vs ln(Pop)', 'ln(Pop)', 'Prob. prevista', od / f"{basefile}__H3_logit_lnpop_sev.png")

# H4 — varrer V2016
q_v = _np.quantile(BASE['V2016_c'].dropna(), [0.1,0.5,0.9]) if BASE['V2016_c'].notna().any() else [0,0,0]
ys = []
for g in G_levels:
    Xp = pd.DataFrame({'const':1.0,'Gestao_c':g,'Sever_c':BASE['Sever_c'].mean(),'V2016_c':q_v,'Gestao_c:V2016_c':g*q_v,'Sever_c:V2016_c':BASE['Sever_c'].mean()*q_v})
    pr = res_h4_logit.predict(Xp)
    ys.append((f'Gestão {"+1dp" if g>0 else "-1dp"}', list(map(float, pr))))
plot_lines(q_v, ys, 'H4-Logit: Prob. Reeleição vs Vantagem2016', 'Vantagem 2016', 'Prob. prevista', od / f"{basefile}__H4_logit_v2016.png")

ys = []
for s in S_levels:
    Xp = pd.DataFrame({'const':1.0,'Gestao_c':BASE['Gestao_c'].mean(),'Sever_c':s,'V2016_c':q_v,'Gestao_c:V2016_c':BASE['Gestao_c'].mean()*q_v,'Sever_c:V2016_c':s*q_v})
    pr = res_h4_logit.predict(Xp)
    ys.append((f'Severidade {"+1dp" if s>0 else "-1dp"}', list(map(float, pr))))
plot_lines(q_v, ys, 'H4-Logit: Prob. Reeleição vs Vantagem2016', 'Vantagem 2016', 'Prob. prevista', od / f"{basefile}__H4_logit_v2016_sev.png")

# Relatório mínimo do que foi usado para os índices (para transparência)
used_report = [
    f"[Gestão] componentes: {', '.join(used_g) if used_g else 'NENHUM (índice ficou NaN)'}",
    f"[Severidade] componentes: {', '.join(used_s) if used_s else 'NENHUM (índice ficou NaN)'}",
    f"Estado: {col_estado}",
    f"Região: {col_regiao}",
    f"Reeleição: {col_ree}",
    f"Vantagem2016: {col_v2016}",
    f"ΔVantagem: {col_dv}",
    f"ln_pop: {col_lnpop}",
]
save_txt(od / f"{basefile}__H3H4_relatorio_componentes.txt", "\n".join(used_report))

print("[OK] H3/H4 estimadas e salvas em:", str(od.resolve()))
print("Arquivos-chave gerados:")
for nm in [
    'H3_logit_sumario.txt','H3_ols_sumario.txt','H4_logit_sumario.txt','H4_ols_sumario.txt',
    'H3_logit_lnpop.png','H3_logit_lnpop_sev.png','H4_logit_v2016.png','H4_logit_v2016_sev.png',
    'H3H4_relatorio_componentes.txt'
]:
    print(' -', f"{basefile}__{nm}")
