#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
master_v18.py — pipeline principal COMPLETO
-------------------------------------------
Correções principais versus v17:
• Some com os logs do Matplotlib do tipo “Using categorical units…”
  (usa mpl.set_loglevel('warning')).
• Some com SettingWithCopyWarning e usa .copy() nos slices.
• Converte explicitamente todos os eixos/arrays numéricos para float.
• Mantém Tabela 3.0 (VIF), Tabela 3.1 (Logit), Tabela 3.2 (OLS) e
  Figura 3.7 (coefplot do Modelo 6). Nada de títulos embutidos em plots.

Como rodar (PowerShell, 1 linha):
  py .\master_v18.py --csv ".\Base_VALIDADA_E_PRONTA.csv" --outdir ".\output" --encoding latin-1 --sep ";" --decimal ","

Mapeamentos opcionais (se algum nome divergir do seu CSV):
  --map Reeleicao="Reeleito (0/1)" --map DeltaIFDM_Emprego="Δ IFDM Emprego & Renda (2020-2016)" \
  --map DeltaIFDM_Saude="IFDM Saúde – variação (2016-2020)"
"""
from __future__ import annotations

import argparse, logging, os, re, shutil, subprocess, sys, warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ----------------------- venv / pip -----------------------
VENV_DIR = Path('.venv')
PY_PATH  = VENV_DIR / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
REQ_FILE = Path('requirements.txt')

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Silenciar avisos/logs ruidosos
warnings.filterwarnings("ignore", message="Glyph .* missing from", category=UserWarning)
warnings.filterwarnings("ignore", message="Tight layout not applied", category=UserWarning)
try:
    import pandas as _pd
    import pandas.errors as _pderr
    warnings.simplefilter(action='ignore', category=_pderr.SettingWithCopyWarning)
    _pd.options.mode.chained_assignment = None  # type: ignore[attr-defined]
except Exception:
    pass

BASE_REQ = (
    "contourpy==1.3.3\n"
    "cycler==0.12.1\n"
    "et_xmlfile==2.0.0\n"
    "fonttools==4.59.2\n"
    "kiwisolver==1.4.9\n"
    "matplotlib==3.9.2\n"
    "numpy==2.1.1\n"
    "openpyxl==3.1.5\n"
    "packaging==25.0\n"
    "pandas==2.2.2\n"
    "pillow==11.3.0\n"
    "pyparsing==3.2.4\n"
    "python-dateutil==2.9.0.post0\n"
    "pytz==2025.2\n"
    "scipy==1.13.1\n"
    "seaborn==0.13.2\n"
    "six==1.17.0\n"
    "statsmodels==0.14.2\n"
    "tzdata==2025.2\n"
    "wheel==0.45.1\n"
)

def run(cmd: List[str]):
    logger.info('> ' + ' '.join(map(str, cmd)))
    subprocess.check_call(cmd)

def venv_exists() -> bool:
    return VENV_DIR.exists() and PY_PATH.exists()

def venv_is_healthy() -> bool:
    if not venv_exists(): return False
    try:
        out = subprocess.check_output([str(PY_PATH), '-c', 'import sys;print(sys.version)'], text=True).strip()
        return bool(out)
    except Exception:
        return False

def recreate_venv():
    if VENV_DIR.exists():
        logger.warning("Venv existente parece quebrada. Removendo .venv para recriar…")
        shutil.rmtree(VENV_DIR, ignore_errors=True)
    logger.info('Criando venv…')
    run([sys.executable, '-m', 'venv', str(VENV_DIR)])

def ensurepip_inside_venv():
    try:
        run([str(PY_PATH), '-m', 'ensurepip', '--upgrade'])
        run([str(PY_PATH), '-m', 'pip', 'install', '--upgrade', 'pip', 'setuptools', 'wheel'])
    except subprocess.CalledProcessError:
        logger.warning("Falha ao rodar ensurepip; tentando seguir mesmo assim.")

def ensure_venv_created_and_healthy():
    if not venv_is_healthy():
        recreate_venv()
    else:
        logger.info('Venv encontrada e saudável.')

def pip_install(args: List[str]):
    try:
        run([str(PY_PATH), '-m', 'pip'] + args)
    except subprocess.CalledProcessError:
        logger.warning("pip dentro da venv parece indisponível; tentando ensurepip…")
        ensurepip_inside_venv()
        run([str(PY_PATH), '-m', 'pip'] + args)

def ensure_deps():
    if (not REQ_FILE.exists()) or REQ_FILE.is_dir() or REQ_FILE.stat().st_size == 0:
        logger.info('Criando requirements.txt padrão (wheels estáveis)…')
        REQ_FILE.write_text(BASE_REQ, encoding='utf-8')
    try:
        pip_install(['install', '--upgrade', 'pip', 'setuptools', 'wheel'])
    except subprocess.CalledProcessError:
        logger.warning('Falha ao atualizar pip/setuptools/wheel; seguindo assim mesmo.')
    pip_install(['install', '-r', str(REQ_FILE.resolve())])

# ----------------------- CLI -----------------------
def parse_args():
    p = argparse.ArgumentParser(description='Pipeline completo: descritivas + VIF/Logit + OLS/coefplot com diagnósticos')
    p.add_argument('--csv', required=True, type=Path)
    p.add_argument('--outdir', default=Path('./output'), type=Path)
    p.add_argument('--encoding', default='latin-1')
    p.add_argument('--sep', default=';')
    p.add_argument('--decimal', default=',')
    p.add_argument('--no-xlsx', action='store_true')
    p.add_argument('--max-num-graphs-per-cat', type=int, default=3)
    p.add_argument('--top-k-categories', type=int, default=30)
    p.add_argument('--check-delta-bins', action='store_true')
    # IFDM KDE
    p.add_argument('--ifdm_xmin', type=float, default=-0.2)
    p.add_argument('--ifdm_xmax', type=float, default=0.3)
    p.add_argument('--ifdm_bw_adjust', type=float, default=0.8)
    # mapping manual
    p.add_argument('--map', action='append', default=[], help='Mapeia nomes: VAR="Nome no CSV" (pode repetir)')
    return p.parse_args()

args = parse_args()
ensure_venv_created_and_healthy()
ensure_deps()

# ----------------------- imports pesados -----------------------
import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.set_loglevel("warning")  # corta logs INFO do Matplotlib (categorical units)
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.max_open_warning'] = 0
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
sns.set_theme(style='whitegrid')

HAS_XLSX = False if args.no_xlsx else True
if HAS_XLSX:
    try:
        import openpyxl  # noqa: F401
    except Exception as e:
        logger.warning(f'openpyxl indisponível ({e}); XLSX será pulado.')
        HAS_XLSX = False

# ----------------------- utilidades -----------------------
CATEG_FIXED = {'Município','Estado','Nome do Incumbente','Partido','Região do País','Gênero','Faixa de População'}
NUMERIC_PATTERNS = re.compile(r'(?i)(População|Número efetivo|delta|Votos|Abstenção|Eleitores|Vantagem|%|PIB|IFDM|Dias da eleição|Normalizado)')

class Numberer:
    def __init__(self):
        self.tab = 1; self.fig = 1
    @staticmethod
    def slug(text: str) -> str:
        text = re.sub(r'[\s]+','_', str(text).strip())
        text = re.sub(r'[^0-9A-Za-z_\-À-ÿ]', '', text)
        return text[:120]
    def table(self, logical_name: str, ext: str) -> str:
        name = f"{self.tab:02d}_{self.slug(logical_name)}.{ext}"; self.tab += 1; return name
    def figure(self, logical_name: str) -> str:
        name = f"{self.fig:02d}_{self.slug(logical_name)}.png"; self.fig += 1; return name

NUM = Numberer()

def br_to_float_series(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = (s.str.replace('%','', regex=False)
           .str.replace('‰','', regex=False)
           .str.replace('\u00A0','', regex=False)
           .str.replace(' ', '', regex=False))
    s = s.str.replace(r"\.(?=\d{3}(\D|$))", '', regex=True)
    s = s.str.replace(',', '.', regex=False)
    return pd.to_numeric(s, errors='coerce')

def to_binary(series: pd.Series) -> pd.Series:
    low = series.astype(str).str.strip().str.lower()
    yes = {'sim','true','yes','1','y','t','masculino','m'}
    no  = {'não','nao','false','no','0','n','f','feminino','f'}
    return pd.Series(np.where(low.isin(yes), 1, np.where(low.isin(no), 0, np.nan)), index=series.index, dtype='float')

def clean_labels(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.replace(r"^\s*\d+\s*[-–—]\s*", '', regex=True).str.strip()
    return df

def fixed_type_for(name: str) -> str:
    if name in CATEG_FIXED: return 'categorical'
    if name.startswith('Início '): return 'datetime'
    if NUMERIC_PATTERNS.search(name): return 'numeric'
    return 'categorical'

def build_type_map(cols: List[str]) -> Dict[str, str]:
    return {c: fixed_type_for(c) for c in cols}

def write_table(df: pd.DataFrame, outdir: Path, nome: str):
    csv_name = NUM.table(nome, 'csv'); df.to_csv(outdir / csv_name, index=False, encoding='utf-8')
    if HAS_XLSX:
        try:
            xlsx_name = NUM.table(nome, 'xlsx'); df.to_excel(outdir / xlsx_name, index=False)
        except Exception as e:
            logger.warning(f'Falha ao salvar XLSX ({nome}): {e} — seguindo com CSV.')

def save_figure(fig, outdir: Path, nome: str):
    fname = NUM.figure(nome); fig.tight_layout(); fig.savefig(outdir / fname); plt.close(fig)

def frequency_table(series: pd.Series, dropna=False) -> pd.DataFrame:
    vc = series.value_counts(dropna=dropna); total = series.shape[0]
    return pd.DataFrame({'valor': vc.index.astype(str), 'contagem': vc.values, 'percentual': (vc.values/total)*100})

# ----------------------- carregamento -----------------------

def load_csv(path: Path, encoding='latin-1', sep=';', decimal=',') -> pd.DataFrame:
    logger.info(f'Lendo CSV: {path}')
    df = pd.read_csv(path, encoding=encoding, sep=sep, decimal=decimal)
    logger.info(f'Dimensão: {df.shape[0]} linhas × {df.shape[1]} colunas')
    return df

# ----------------------- blocos descritivos -----------------------

def aggregate_by_category(df: pd.DataFrame, cat_col: str, numeric_cols: List[str]) -> pd.DataFrame:
    agg_dict = {c: ['count','sum','mean','std','min','max'] for c in numeric_cols}
    g = df.groupby(cat_col, dropna=False)
    out = g[numeric_cols].agg(agg_dict)
    out.columns = ['__'.join(c).strip() for c in out.columns.to_flat_index()]
    out = out.reset_index().rename(columns={cat_col: 'categoria'})
    return out

def choose_top_numeric_for_plots(df_num: pd.DataFrame, numeric_cols: List[str], k: int, cat_col: str) -> List[str]:
    if k <= 0 or not numeric_cols: return []
    variances = {}; g = df_num.groupby(cat_col, dropna=False)
    for c in numeric_cols:
        try: variances[c] = g[c].mean().var()
        except Exception: variances[c] = 0.0
    top = sorted(variances.items(), key=lambda x: (x[1] if x[1] is not None else -1), reverse=True)
    return [t[0] for t in top[:k]]

def plot_cat_count(tab_freq: pd.DataFrame, titulo: str, outdir: Path, topk: int):
    top = tab_freq.head(topk).copy()
    top['contagem'] = pd.to_numeric(top['contagem'], errors='coerce').astype(float)
    fig, ax = plt.subplots(figsize=(12, max(4, 0.33*len(top))))
    sns.barplot(y='valor', x='contagem', data=top, ax=ax)
    xlim = ax.get_xlim(); span = xlim[1] - xlim[0]
    for i, (v, c) in enumerate(zip(top['valor'], top['contagem'])):
        ax.text(float(c) + span*0.01, i, f"{int(c)}", va='center')
    ax.set_xlabel('Contagem'); ax.set_ylabel('Categoria')
    save_figure(fig, outdir, f"freq_{titulo}")

def plot_cat_numeric_bar(df: pd.DataFrame, cat_col: str, num_col: str, how: str, titulo: str, outdir: Path, topk: int):
    s = df.groupby(cat_col, dropna=False)[num_col].agg(how)
    tab = s.reset_index().rename(columns={cat_col: 'categoria', 0: how, num_col: how}).sort_values(by=how, ascending=False)
    tab[how] = pd.to_numeric(tab[how], errors='coerce').astype(float)
    top = tab.head(topk).copy()
    fig, ax = plt.subplots(figsize=(12, max(4, 0.33*len(top))))
    sns.barplot(y='categoria', x=how, data=top, ax=ax)
    xlim = ax.get_xlim(); span = xlim[1] - xlim[0]
    for i, v in enumerate(top[how].values):
        ax.text(float(v) + span*0.01, i, f"{float(v):.2f}", va='center')
    ax.set_xlabel(how.upper()); ax.set_ylabel('Categoria')
    save_figure(fig, outdir, f"{how}_{num_col}_por_{cat_col}")

def dictionary_vars(df: pd.DataFrame, type_map: Dict[str,str]) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        t = type_map.get(col, 'categorical'); miss = df[col].isna().mean()*100; sample = ''
        if t == 'numeric':
            s = br_to_float_series(df[col])
            sample = f'[{s.min():.3g}, {s.max():.3g}]' if s.notna().any() else ''
        elif t == 'datetime':
            s = pd.to_datetime(df[col], errors='coerce', dayfirst=True)
            sample = f'{s.min().date()} → {s.max().date()}' if s.notna().any() else ''
        else:
            vals = df[col].dropna().astype(str).unique().tolist()
            sample = ', '.join(vals[:5]) + (f' … (+{len(vals)-5})' if len(vals)>5 else '')
        rows.append({'variavel': col, 'tipo': t, '%faltante': round(miss,2), 'amostra/intervalo': sample})
    return pd.DataFrame(rows)

def save_categorical_block(df: pd.DataFrame, cat_col: str, outdir: Path, topk: int, max_num: int, type_map: Dict[str,str]):
    title = f"Frequência de {cat_col}"
    tab = frequency_table(df[cat_col], dropna=False)
    tab['contagem'] = pd.to_numeric(tab['contagem'], errors='coerce').astype(float)
    write_table(tab, outdir, f"freq_{cat_col}")
    plot_cat_count(tab, title, outdir, topk)
    numeric_cols = [c for c, t in type_map.items() if t == 'numeric']
    if numeric_cols:
        casted = df.copy()
        for c in numeric_cols: casted[c] = br_to_float_series(df[c])
        agg = aggregate_by_category(casted, cat_col, numeric_cols)
        write_table(agg, outdir, f"aggs_{cat_col}_sobre_numericas")
        chosen = choose_top_numeric_for_plots(casted, numeric_cols, max_num, cat_col)
        for num_col in chosen:
            plot_cat_numeric_bar(casted, cat_col, num_col, 'mean', f"MÉDIA de {num_col} por {cat_col}", outdir, topk)
            plot_cat_numeric_bar(casted, cat_col, num_col, 'sum',  f"SOMA de {num_col} por {cat_col}",  outdir, topk)

# ----------------------- blocos específicos do seu estudo -----------------------

def build_reelection_outputs(df: pd.DataFrame, outdir: Path):
    col_r = next((c for c in df.columns if any(k in c.lower() for k in ['reelei','reeleito','reeleg'])), None)
    col_g = next((c for c in df.columns if 'região do país' in c.lower() or 'regiao do pais' in c.lower()), None)
    if not col_r or not col_g:
        logger.warning('Colunas para reeleição/região não identificadas; pulando etapa de reeleição.')
        return float('nan'), pd.Series(dtype=float)
    mapping = {'sim':1,'não':0,'nao':0,'yes':1,'no':0,'true':1,'false':0,'1':1,'0':0}
    re_bin = pd.to_numeric(df[col_r].astype(str).str.strip().str.lower().map(mapping), errors='coerce')
    overall = float(re_bin.mean()*100) if re_bin.notna().any() else float('nan')
    by_reg = df.assign(_re=re_bin).groupby(col_g, dropna=False)['_re'].mean().mul(100).sort_values(ascending=False)
    write_table(pd.DataFrame({'Indicador':['Taxa de reeleição (amostra)'],'N (municípios)':[len(df)],'Reeleição (%)':[round(overall,2)]}), outdir, 'resumo_reeleicao_amostra')
    tmp = by_reg.reset_index().rename(columns={col_g:'Região','_re':'Reeleição (%)'})
    tmp['Reeleição (%)'] = pd.to_numeric(tmp['Reeleição (%)'], errors='coerce').astype(float)
    write_table(tmp, outdir, 'reeleicao_por_regiao')
    fig, ax = plt.subplots(figsize=(8,4.8))
    sns.barplot(x=by_reg.index.astype(str), y=pd.to_numeric(by_reg.values), ax=ax)
    for i, v in enumerate(by_reg.values): ax.text(i, float(v), f"{float(v):.1f}%", ha='center', va='bottom')
    ax.set_ylabel('Reeleição (%)'); ax.set_xlabel('Região')
    ax.set_ylim(0, max(float(by_reg.max())*1.1, 100))
    save_figure(fig, outdir, 'fig_3_1_reeleicao_por_regiao')
    return overall, by_reg

def find_delta_vant_col(df: pd.DataFrame) -> Optional[str]:
    pref = ['delta_vantagem_2020_2016','delta_vantagem (2020-2016)']
    for p in pref:
        if p in df.columns: return p
    for c in df.columns:
        if 'delta_vantagem' in c.lower(): return c
    return None

def build_delta_vantagem_outputs(df: pd.DataFrame, outdir: Path):
    col = find_delta_vant_col(df)
    if not col:
        logger.warning('Coluna de delta_vantagem não encontrada; pulando etapa.')
        return float('nan'), float('nan'), float('nan')
    s = br_to_float_series(df[col]).dropna()
    stats_df = pd.DataFrame({'métrica':['média','desvio_padrão','mediana','n'], 'valor':[s.mean(), s.std(ddof=1), s.median(), len(s)]})
    write_table(stats_df, outdir, 'delta_vantagem_stats')
    if not s.empty:
        fig, ax = plt.subplots(figsize=(10,6))
        sns.histplot(pd.to_numeric(s), bins=30, kde=False, ax=ax, edgecolor='black', alpha=0.6)
        ax.axvline(0, color='red', linestyle='--', linewidth=1.5)
        ax.set_xlabel('Variação na vantagem (p.p.)'); ax.set_ylabel('Frequência')
        save_figure(fig, outdir, 'fig_3_2_distribuicao_delta_vantagem')
    return float(s.mean()), float(s.median()), float(s.std(ddof=1))

# --- NPIs e Infraestrutura ---

def npi_cols(df: pd.DataFrame):
    c_com, c_mas = None, None
    for c in df.columns:
        lc = c.lower()
        if ('restri' in lc and 'com' in lc and 'normal' in lc) and c_com is None:
            c_com = c
        if (('másc' in lc or 'masc' in lc) and 'normal' in lc) and c_mas is None:
            c_mas = c
    return c_com, c_mas

def infra_cols(df: pd.DataFrame) -> Dict[str,str]:
    mapping = {}
    exact = {
        'Capacidade de Testagem (PCR/Sorológico)': 'Local (público ou particular) com condições de realizar testes RT-PCR ou sorológicos - existência',
        'Ampliação de Leitos': 'O número de leitos foi ampliado no município durante a pandemia',
        'Tendas de Triagem': 'Houve a instalação de tendas de triagem no município',
        'Hospital de Campanha': 'Foi instalado hospital de campanha no município',
    }
    for k,v in exact.items():
        if v in df.columns: mapping[k] = v
    if not mapping:
        for c in df.columns:
            lc = c.lower()
            if 'pcr' in lc or 'sorol' in lc or 'teste' in lc or 'testes' in lc:
                mapping['Capacidade de Testagem (PCR/Sorológico)'] = c
            elif 'leito' in lc and ('ampl' in lc or 'aument' in lc):
                mapping['Ampliação de Leitos'] = c
            elif 'tenda' in lc:
                mapping['Tendas de Triagem'] = c
            elif 'hospital de campanha' in lc:
                mapping['Hospital de Campanha'] = c
    return mapping

def pct_yes(series: pd.Series) -> float:
    s = series.astype(str).str.strip().str.lower()
    yes = s.isin(['sim','1','true','yes'])
    return float(yes.mean()*100)

def build_npi_infra_outputs(df: pd.DataFrame, outdir: Path):
    c_com, c_mas = npi_cols(df)
    npi_stats = {}
    if c_com and c_mas:
        s_com = br_to_float_series(df[c_com]); s_mas = br_to_float_series(df[c_mas])
        npi_stats = {
            'Média - Restrição ao Comércio': float(s_com.mean()),
            'DP - Restrição ao Comércio': float(s_com.std(ddof=1)),
            'Média - Obrigatoriedade de Máscara': float(s_mas.mean()),
            'DP - Obrigatoriedade de Máscara': float(s_mas.std(ddof=1)),
        }
        fig, ax = plt.subplots(figsize=(10,6))
        sns.kdeplot(s_com.dropna(), fill=True, label='Restrição ao comércio', bw_adjust=0.9, cut=0, ax=ax)
        sns.kdeplot(s_mas.dropna(), fill=True, label='Obrigatoriedade de máscara', bw_adjust=0.9, cut=0, ax=ax)
        ax.set_xlabel('Precocidade normalizada (0 = eleição, 1 = início da pandemia)'); ax.set_ylabel('Densidade')
        ax.legend()
        save_figure(fig, outdir, 'fig_3_3_precocidade_npis')
    mapping = infra_cols(df)
    rows = []
    for label, col in mapping.items():
        p = pct_yes(df[col]); rows.append({'Ação': label, 'Percentual (%)': round(p,2)})
    tab = pd.DataFrame(rows).sort_values('Percentual (%)', ascending=True)
    if not tab.empty:
        write_table(tab, outdir, 'infraestrutura_percentuais')
        fig, ax = plt.subplots(figsize=(8,6))
        sns.barplot(data=tab, y='Ação', x='Percentual (%)', ax=ax)
        for i,v in enumerate(tab['Percentual (%)']): ax.text(float(v)+1, i, f"{float(v):.1f}%", va='center')
        ax.set_xlim(0,100); ax.set_ylabel('')
        save_figure(fig, outdir, 'fig_3_4_infraestrutura')
    return npi_stats

# --- Estresse no sistema ---

def stress_cols(df: pd.DataFrame) -> Dict[str,str]:
    mapping = {}
    exact = {
        'Necessidade de transferência de pacientes': 'Houve necessidade de transferir pacientes para outros municípios',
        'Pacientes em unidades sem internação (>24h)': 'Houve manutenção de pacientes por mais de 24 horas em unidades sem capacidade de internação',
        'Sobrecarga de leitos/UTI': 'Houve sobrecarga na capacidade de leitos ou UTI',
    }
    for k,v in exact.items():
        if v in df.columns: mapping[k] = v
    if len(mapping) < 3:
        for c in df.columns:
            lc = c.lower()
            if ('transfer' in lc or 'transferê' in lc) and 'pacien' in lc:
                mapping.setdefault('Necessidade de transferência de pacientes', c)
            if ('24' in lc or '>24' in lc) and ('sem interna' in lc or 'sem intern' in lc or 'capacidade de interna' in lc):
                mapping.setdefault('Pacientes em unidades sem internação (>24h)', c)
            if 'sobrecarga' in lc or ('leitos' in lc and 'uti' in lc):
                mapping.setdefault('Sobrecarga de leitos/UTI', c)
    return mapping

def build_stress_outputs(df: pd.DataFrame, outdir: Path) -> Dict[str, float]:
    mapping = stress_cols(df)
    rows = []
    for label, col in mapping.items():
        p = pct_yes(df[col])
        rows.append({'Indicador de Estresse': label, 'Percentual (%)': round(p,2)})
    tab = pd.DataFrame(rows)
    if not tab.empty:
        ordem = ['Sobrecarga de leitos/UTI','Pacientes em unidades sem internação (>24h)','Necessidade de transferência de pacientes']
        tab['ord'] = tab['Indicador de Estresse'].apply(lambda x: ordem.index(x) if x in ordem else 99)
        tab = tab.sort_values('ord').drop(columns='ord')
        write_table(tab, outdir, 'estresse_saude_percentuais')
        fig, ax = plt.subplots(figsize=(10,6))
        sns.barplot(data=tab, y='Indicador de Estresse', x='Percentual (%)', ax=ax, orient='h')
        for i,v in enumerate(tab['Percentual (%)']): ax.text(float(v)+0.5, i, f"{float(v):.1f}%", va='center')
        ax.set_xlabel('Percentual (%)'); ax.set_ylabel('')
        ax.set_xlim(0,100)
        save_figure(fig, outdir, 'fig_3_5_estresse_saude')
    return {r['Indicador de Estresse']: r['Percentual (%)'] for r in rows}

# --- IFDM ---

def find_ifdm_delta_cols(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    emp = None; sau = None
    for c in df.columns:
        lc = c.lower()
        if emp is None and ('ifdm' in lc and ('emprego' in lc or 'renda' in lc) and ('varia' in lc or 'delta' in lc)):
            emp = c
        if sau is None and ('ifdm' in lc and ('saúde' in lc or 'saude' in lc) and ('varia' in lc or 'delta' in lc)):
            sau = c
    return emp, sau

def build_ifdm_outputs(df: pd.DataFrame, outdir: Path) -> Dict[str, float]:
    c_emp, c_sau = find_ifdm_delta_cols(df)
    rows = []
    s_emp = pd.Series(dtype=float); s_sau = pd.Series(dtype=float)
    if c_emp is not None:
        s_emp = br_to_float_series(df[c_emp]).dropna()
        rows.append({'Indicador':'IFDM Emprego & Renda','Média': float(s_emp.mean()),'DP': float(s_emp.std(ddof=1)),'% Piora': float((s_emp<0).mean()*100),'N': int(len(s_emp))})
    if c_sau is not None:
        s_sau = br_to_float_series(df[c_sau]).dropna()
        rows.append({'Indicador':'IFDM Saúde','Média': float(s_sau.mean()),'DP': float(s_sau.std(ddof=1)),'% Piora': float((s_sau<0).mean()*100),'N': int(len(s_sau))})
    if rows:
        tab = pd.DataFrame(rows)
        write_table(tab, outdir, 'ifdm_variacoes_stats')
        if not s_emp.empty or not s_sau.empty:
            fig, ax = plt.subplots(figsize=(12,6))
            if not s_emp.empty:
                sns.kdeplot(s_emp, fill=True, label='IFDM Emprego & Renda', bw_adjust=args.ifdm_bw_adjust, cut=0, ax=ax)
            if not s_sau.empty:
                sns.kdeplot(s_sau, fill=True, label='IFDM Saúde', bw_adjust=args.ifdm_bw_adjust, cut=0, ax=ax)
            ax.axvline(0, ls='--', color='crimson', linewidth=1.5)
            data_min = min([v.min() for v in [s_emp, s_sau] if not v.empty])
            data_max = max([v.max() for v in [s_emp, s_sau] if not v.empty])
            xmin = min(args.ifdm_xmin, float(data_min) - 0.02)
            xmax = max(args.ifdm_xmax, float(data_max) + 0.02)
            ax.set_xlim(xmin, xmax)
            ax.set_xlabel('Variação no IFDM (2020–2016)')
            ax.set_ylabel('Densidade')
            ax.legend()
            save_figure(fig, outdir, 'fig_3_6_ifdm_variacoes')
    return {r['Indicador']: r['Média'] for r in rows}

# ----------------------- Tabelas 3.0/3.1 (VIF e Logit) -----------------------
CANONICAL = {
    'Reeleicao': ['Reeleição','Reeleicao','Reeleito (0/1)','Reelection','reeleito','reeleicao'],
    'NEC_2020': ['NEC_2020','NEC 2020','Número efetivo de candidatos 2020','Numero efetivo 2020'],
    'Vantagem2016': ['Vantagem2016','Vantagem 2016 (p.p.)','Delta vantagem 2016'],
    'LnPop': ['LnPop','Log Pop','log_populacao','ln_pop'],
    'PIB_PC': ['PIB_PC','PIB per capita','PIB pc','PIBpc'],
    'Genero': ['Genero','Gênero','Sexo incumbente','Genero_incumbente'],
    'Regiao': ['Região do País','Regiao do Pais','Região','Regiao'],
    'NPI_Mascara': ['NPI_Mascara','Dias da eleição - Máscara (Normalizado)','precocidade_mascara'],
    'NPI_Comercio': ['NPI_Comercio','Dias da eleição - Restrição ao comércio (Normalizado)','precocidade_comercio'],
    'Infra_HospCampanha': ['Infra_HospCampanha','Foi instalado hospital de campanha no município','HospCampanha'],
    'Infra_Tendas': ['Infra_Tendas','Houve a instalação de tendas de triagem no município','TendasTriagem'],
    'Infra_Leitos': ['Infra_Leitos','O número de leitos foi ampliado no município durante a pandemia','AmpliacaoLeitos'],
    'Stress_SobrecargaUTI': ['Stress_SobrecargaUTI','Houve sobrecarga na capacidade de leitos ou UTI'],
    'Stress_Transferencia': ['Stress_Transferencia','Houve necessidade de transferir pacientes para outros municípios'],
    'Stress_Espera24h': ['Stress_Espera24h','Houve manutenção de pacientes por mais de 24 horas em unidades sem capacidade de internação'],
    'DeltaIFDM_Saude': ['DeltaIFDM_Saude','Δ IFDM Saúde (2020-2016)','IFDM Saúde – variação (2016-2020)','delta_firjan_saude'],
    'DeltaIFDM_Emprego': ['DeltaIFDM_Emprego','Δ IFDM Emprego & Renda (2020-2016)','IFDM Emprego & Renda – variação (2016-2020)','delta_firjan_emprego_renda'],
}

ORDER_VIF = [
    'LnPop','Infra_Leitos','Stress_Transferencia','PIB_PC','DeltaIFDM_Emprego',
    'Stress_SobrecargaUTI','Infra_HospCampanha','NPI_Mascara','NEC_2020','Stress_Espera24h',
    'DeltaIFDM_Saude','Infra_Tendas','Vantagem2016','Genero','NPI_Comercio'
]

GRUPO_H1 = ['NPI_Mascara','NPI_Comercio','Infra_HospCampanha','Infra_Tendas','Infra_Leitos']
GRUPO_H2 = ['Stress_SobrecargaUTI','Stress_Transferencia','Stress_Espera24h','DeltaIFDM_Saude','DeltaIFDM_Emprego']
CONTROLES = ['Vantagem2016','NEC_2020','LnPop','PIB_PC','Genero']


def parse_user_map(pairs: List[str]) -> Dict[str, str]:
    m: Dict[str,str] = {}
    for p in pairs:
        if '=' not in p: continue
        k, v = p.split('=', 1)
        k = k.strip(); v = v.strip().strip('"').strip("'")
        m[k] = v
    return m

USERMAP = parse_user_map(args.map)

def resolve_col(df: pd.DataFrame, key: str) -> Optional[str]:
    if key in USERMAP:
        return USERMAP[key] if USERMAP[key] in df.columns else None
    for cand in CANONICAL.get(key, []):
        if cand in df.columns: return cand
    key_lc = key.lower()
    for c in df.columns:
        if key_lc in c.lower(): return c
    return None


def stars(p):
    return '***' if p is not None and p < 0.01 else ('**' if p is not None and p < 0.05 else ('*' if p is not None and p < 0.1 else ''))


def format_coef(coef, se, p):
    if coef is None: return ''
    try:
        return f"{coef:.4g}{stars(p)}\n({se:.3g})"
    except Exception:
        return f"{coef}{stars(p)}\n({se})"


def build_vif_and_logit_tables(df: pd.DataFrame, outdir: Path):
    cols: Dict[str,str] = {}
    for k in set(list(CANONICAL.keys()) + ['Reeleicao']):
        c = resolve_col(df, k)
        if c is not None:
            cols[k] = c
    if 'Reeleicao' not in cols:
        raise RuntimeError('Coluna da variável dependente de reeleição não localizada. Use --map Reeleicao="nome exato".')

    D = pd.DataFrame(index=df.index)
    Y = to_binary(df[cols['Reeleicao']]).astype(float)

    for k in CONTROLES:
        if k not in cols: continue
        if k == 'Genero':
            D[k] = to_binary(df[cols[k]])
        else:
            D[k] = br_to_float_series(df[cols[k]])

    for k in GRUPO_H1:
        if k in cols:
            D[k] = br_to_float_series(df[cols[k]]) if k.startswith('NPI_') else to_binary(df[cols[k]])
    for k in GRUPO_H2:
        if k in cols:
            D[k] = br_to_float_series(df[cols[k]]) if k.startswith('DeltaIFDM') else to_binary(df[cols[k]])

    # D para float
    D = D.apply(pd.to_numeric, errors='coerce').astype(float)

    REG_DUMMIES = pd.DataFrame(index=df.index)
    if 'Regiao' in cols:
        reg = df[cols['Regiao']].astype(str)
        REG_DUMMIES = pd.get_dummies(reg, prefix='Reg', drop_first=True)
        REG_DUMMIES = REG_DUMMIES.apply(pd.to_numeric, errors='coerce').astype(float)

    mask = Y.notna()
    Y = Y[mask]; D = D.loc[mask]; REG_DUMMIES = REG_DUMMIES.loc[mask]

    # VIF
    vif_vars = [v for v in ORDER_VIF if v in D.columns]
    X_vif = D[vif_vars].dropna()
    if not X_vif.empty:
        X_vif = X_vif.apply(pd.to_numeric, errors='coerce').astype(float)
        X_np = X_vif.to_numpy(dtype=float)
        vifs = []
        for i, name in enumerate(X_vif.columns):
            try: val = variance_inflation_factor(X_np, i)
            except Exception: val = np.nan
            vifs.append({'Variável': name, 'VIF': float(val) if val is not None else np.nan})
        T30 = pd.DataFrame(vifs)
        T30['ord'] = T30['Variável'].apply(lambda x: ORDER_VIF.index(x) if x in ORDER_VIF else 999)
        T30 = T30.sort_values('ord').drop(columns='ord')
        write_table(T30, outdir, 'Tabela_3_0_Diagnostico_Multicolinearidade_VIF')
    else:
        logger.warning('Sem dados válidos para VIF.')

    # LOGIT (3.1)
    def make_X(cols_list: List[str]):
        X = pd.concat([D[[c for c in cols_list if c in D.columns]], REG_DUMMIES], axis=1)
        X = X.apply(pd.to_numeric, errors='coerce').astype(float)
        X = sm.add_constant(X, has_constant='add')
        m = X.notna().all(axis=1) & Y.notna()
        return X.loc[m], Y.loc[m]

    X1, y1 = make_X(CONTROLES)
    X2, y2 = make_X(CONTROLES + GRUPO_H1)
    X3, y3 = make_X(CONTROLES + GRUPO_H1 + GRUPO_H2)

    def fit_logit(X, y):
        mod = sm.Logit(y.astype(float), X.astype(float))
        res = mod.fit(disp=False, cov_type='HC1')
        return res

    res1 = fit_logit(X1, y1)
    res2 = fit_logit(X2, y2)
    res3 = fit_logit(X3, y3)

    def ext(res, name):
        if name in res.params.index:
            return res.params[name], res.bse[name], res.pvalues[name]
        return None, None, None

    linhas = []
    linhas.append({'Variável':'Gestão da Crise (H1)', '(1) Base':'', '(2) + Gestão':'', '(3) Completo':''})
    for var in GRUPO_H1:
        b,se,p = ext(res2, var); c2 = format_coef(b,se,p) if b is not None else ''
        b,se,p = ext(res3, var); c3 = format_coef(b,se,p) if b is not None else ''
        linhas.append({'Variável': var, '(1) Base': '', '(2) + Gestão': c2, '(3) Completo': c3})

    linhas.append({'Variável':'Severidade da Crise (H2)', '(1) Base':'', '(2) + Gestão':'', '(3) Completo':''})
    for var in GRUPO_H2:
        b,se,p = ext(res3, var); c3 = format_coef(b,se,p) if b is not None else ''
        linhas.append({'Variável': var, '(1) Base':'', '(2) + Gestão':'', '(3) Completo': c3})

    linhas.append({'Variável':'Controles Políticos e Estruturais', '(1) Base':'', '(2) + Gestão':'', '(3) Completo':''})
    for var in CONTROLES:
        b1,se1,p1 = ext(res1, var); b2,se2,p2 = ext(res2, var); b3,se3,p3 = ext(res3, var)
        linhas.append({'Variável': var,
                       '(1) Base': format_coef(b1,se1,p1) if b1 is not None else '',
                       '(2) + Gestão': format_coef(b2,se2,p2) if b2 is not None else '',
                       '(3) Completo': format_coef(b3,se3,p3) if b3 is not None else ''})

    b1,se1,p1 = ext(res1, 'const'); b2,se2,p2 = ext(res2, 'const'); b3,se3,p3 = ext(res3, 'const')
    linhas.append({'Variável':'Intercept',
                   '(1) Base': format_coef(b1,se1,p1) if b1 is not None else '',
                   '(2) + Gestão': format_coef(b2,se2,p2) if b2 is not None else '',
                   '(3) Completo': format_coef(b3,se3,p3) if b3 is not None else ''})

    T31 = pd.DataFrame(linhas + [
        {'Variável':'N','(1) Base': int(res1.nobs),'(2) + Gestão': int(res2.nobs),'(3) Completo': int(res3.nobs)},
        {'Variável':'Pseudo R2 (McFadden)','(1) Base': f"{res1.prsquared:.4f}",'(2) + Gestão': f"{res2.prsquared:.4f}",'(3) Completo': f"{res3.prsquared:.4f}"}
    ])

    write_table(T31, outdir, 'Tabela_3_1_Determinantes_Reeleicao_Logit')

    return D, REG_DUMMIES, T31

# ----------------------- Tabela 3.2 (OLS) + Figura 3.7 -----------------------
FIG7_ORDER = [
    'DeltaIFDM_Saude','Infra_Tendas','NPI_Mascara','Stress_SobrecargaUTI','Infra_HospCampanha',
    'Stress_Espera24h','NPI_Comercio','Stress_Transferencia','Infra_Leitos','DeltaIFDM_Emprego'
]
LABEL_ALIASES = {v: v for v in FIG7_ORDER}

def build_ols_tables_and_coefplot(df: pd.DataFrame, outdir: Path, D: pd.DataFrame, REG_DUMMIES: pd.DataFrame):
    col_dv = find_delta_vant_col(df)
    if not col_dv:
        raise RuntimeError('Coluna da VD (delta_vantagem) não encontrada.')
    Y = br_to_float_series(df[col_dv])

    def make_X(cols_list: List[str]):
        X = pd.concat([D[[c for c in cols_list if c in D.columns]], REG_DUMMIES], axis=1)
        X = X.apply(pd.to_numeric, errors='coerce').astype(float)
        X = sm.add_constant(X, has_constant='add')
        m = X.notna().all(axis=1) & Y.notna()
        return X.loc[m], Y.loc[m]

    X4, y4 = make_X(CONTROLES)
    X5, y5 = make_X(CONTROLES + GRUPO_H1)
    X6, y6 = make_X(CONTROLES + GRUPO_H1 + GRUPO_H2)

    def fit_ols(X, y):
        mod = sm.OLS(y.astype(float), X.astype(float))
        res = mod.fit(cov_type='HC1')
        return res

    r4 = fit_ols(X4, y4); r5 = fit_ols(X5, y5); r6 = fit_ols(X6, y6)

    def ext(res, name):
        if name in res.params.index:
            return res.params[name], res.bse[name], res.pvalues[name]
        return None, None, None

    linhas = []
    # H1
    linhas.append({'Variável':'Gestão da Crise (H1)','(4) Base':'','(5) + Gestão':'','(6) Completo':''})
    for var in GRUPO_H1:
        b5,se5,p5 = ext(r5, var); b6,se6,p6 = ext(r6, var)
        linhas.append({'Variável':var,'(4) Base':'','(5) + Gestão': (format_coef(b5,se5,p5) if b5 is not None else ''),
                       '(6) Completo': (format_coef(b6,se6,p6) if b6 is not None else '')})
    # H2
    linhas.append({'Variável':'Severidade da Crise (H2)','(4) Base':'','(5) + Gestão':'','(6) Completo':''})
    for var in GRUPO_H2:
        b6,se6,p6 = ext(r6, var)
        linhas.append({'Variável':var,'(4) Base':'','(5) + Gestão':'','(6) Completo': (format_coef(b6,se6,p6) if b6 is not None else '')})
    # Controles
    linhas.append({'Variável':'Controles Políticos e Estruturais','(4) Base':'','(5) + Gestão':'','(6) Completo':''})
    for var in CONTROLES:
        b4,se4,p4 = ext(r4, var); b5,se5,p5 = ext(r5, var); b6,se6,p6 = ext(r6, var)
        linhas.append({'Variável':var,
                       '(4) Base': (format_coef(b4,se4,p4) if b4 is not None else ''),
                       '(5) + Gestão': (format_coef(b5,se5,p5) if b5 is not None else ''),
                       '(6) Completo': (format_coef(b6,se6,p6) if b6 is not None else '')})
    # Intercept
    b4,se4,p4 = ext(r4,'const'); b5,se5,p5 = ext(r5,'const'); b6,se6,p6 = ext(r6,'const')
    linhas.append({'Variável':'Intercept', '(4) Base': (format_coef(b4,se4,p4) if b4 is not None else ''),
                   '(5) + Gestão': (format_coef(b5,se5,p5) if b5 is not None else ''),
                   '(6) Completo': (format_coef(b6,se6,p6) if b6 is not None else '')})

    T32 = pd.DataFrame(linhas + [
        {'Variável':'R-squared','(4) Base': f"{r4.rsquared:.4f}",'(5) + Gestão': f"{r5.rsquared:.4f}",'(6) Completo': f"{r6.rsquared:.4f}"},
        {'Variável':'R-squared Adj.','(4) Base': f"{r4.rsquared_adj:.4f}",'(5) + Gestão': f"{r5.rsquared_adj:.4f}",'(6) Completo': f"{r6.rsquared_adj:.4f}"},
        {'Variável':'N','(4) Base': int(r4.nobs),'(5) + Gestão': int(r5.nobs),'(6) Completo': int(r6.nobs)}
    ])

    write_table(T32, outdir, 'Tabela_3_2_Determinantes_DeltaVantagem_OLS')
    (outdir / 'Tabela_3_2_OLS.tex').write_text(
        T32.to_latex(index=False, escape=True, caption='Tabela 3.2: Determinantes da Variação na Vantagem (OLS). Erros-padrão robustos (HC1). * p<.1, ** p<.05, *** p<.01.', label='tab:ols'),
        encoding='utf-8'
    )

    # -------- Figura 3.7: coefplot (modelo 6, H1+H2) --------
    present = []
    for var in GRUPO_H1 + GRUPO_H2:
        present.append({'variavel': var,
                        'no_modelo6': var in r6.params.index,
                        'coef': float(r6.params[var]) if var in r6.params.index else None,
                        'se': float(r6.bse[var]) if var in r6.params.index else None})
    diag = pd.DataFrame(present)
    write_table(diag, outdir, 'diagnostico_modelo6_variaveis')

    rows = [
        {'var': v, 'b': float(r6.params[v]), 'se': float(r6.bse[v])}
        for v in (GRUPO_H1 + GRUPO_H2) if v in r6.params.index
    ]

    fig_saved = False
    if rows:
        dfc = pd.DataFrame(rows)
        dfc['lo'] = dfc['b'] - 1.96*dfc['se']
        dfc['hi'] = dfc['b'] + 1.96*dfc['se']
        order = [v for v in FIG7_ORDER if v in dfc['var'].tolist()]
        rest = [v for v in dfc['var'].tolist() if v not in order]
        order = order + sorted(rest)
        dfc['ord'] = dfc['var'].apply(lambda x: order.index(x))
        dfc = dfc.sort_values('ord', ascending=False)

        fig, ax = plt.subplots(figsize=(9, 7))
        y_pos = np.arange(len(dfc))
        ax.errorbar(pd.to_numeric(dfc['b']), y_pos, xerr=1.96*pd.to_numeric(dfc['se']), fmt='o', capsize=4)
        ax.axvline(0, color='red', linestyle='--', linewidth=1.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([LABEL_ALIASES.get(v, v) for v in dfc['var']])
        ax.set_xlabel('Efeito Estimado na Variação da Vantagem (p.p.)')
        fig.tight_layout(); fig.savefig(outdir / 'coeficientes_ols_completo.png', dpi=300)
        save_figure(fig, outdir, 'fig_3_7_coeficientes_ols_completo')
        fig_saved = True
    else:
        # Placeholder útil com aviso visual
        fig, ax = plt.subplots(figsize=(9, 7))
        ax.axvline(0, color='red', linestyle='--', linewidth=1.5)
        ax.text(0.5, 0.5, 'Sem variáveis de Gestão/Severidade no Modelo 6\n(verifique nomes e --map) ',
                transform=ax.transAxes, ha='center', va='center')
        ax.set_xlabel('Efeito Estimado na Variação da Vantagem (p.p.)')
        fig.tight_layout(); fig.savefig(outdir / 'coeficientes_ols_completo.png', dpi=300)
        save_figure(fig, outdir, 'fig_3_7_coeficientes_ols_completo')
        fig_saved = True

    if fig_saved:
        logger.info('Figura 3.7 salva em: ' + str((outdir / 'coeficientes_ols_completo.png').resolve()))

    return T32

# ----------------------- extras utilitários -----------------------

def save_freq_and_agg_blocks(df: pd.DataFrame, type_map: Dict[str, str], outdir: Path, topk: int, max_num: int):
    cat_cols = [c for c, t in type_map.items() if t == 'categorical']
    for col in cat_cols:
        save_categorical_block(df, col, outdir, topk, max_num, type_map)

def check_delta_bins(df: pd.DataFrame, outdir: Path, bins_list=(20,25,30)):
    col = find_delta_vant_col(df)
    if not col:
        logger.warning('delta_vantagem não encontrada; pulando --check-delta-bins')
        return
    s = br_to_float_series(df[col]).dropna()
    for bins in bins_list:
        fig, ax = plt.subplots(figsize=(12,6))
        sns.histplot(pd.to_numeric(s), bins=bins, kde=False, ax=ax)
        ax.axvline(0, color='red', linestyle='--'); ax.set_xlabel('Variação na vantagem (p.p.)'); ax.set_ylabel('Frequência')
        save_figure(fig, outdir, f'comp_bins_delta_vantagem_{bins}')

# ----------------------- cola final / main -----------------------

def main():
    df = load_csv(args.csv, encoding=args.encoding, sep=args.sep, decimal=args.decimal)
    df = clean_labels(df)
    TYPE_MAP = build_type_map(list(df.columns))

    outdir = args.outdir; outdir.mkdir(parents=True, exist_ok=True)

    write_table(df, outdir, 'dataset_original')
    build_reelection_outputs(df, outdir)
    build_delta_vantagem_outputs(df, outdir)
    save_freq_and_agg_blocks(df, TYPE_MAP, outdir, args.top_k_categories, args.max_num_graphs_per_cat)
    dict_df = dictionary_vars(df, TYPE_MAP); write_table(dict_df, outdir, 'dicionario_variaveis')
    build_npi_infra_outputs(df, outdir)
    build_stress_outputs(df, outdir)
    build_ifdm_outputs(df, outdir)

    try:
        D, REG_DUMMIES, _ = build_vif_and_logit_tables(df, outdir)
    except Exception as e:
        logger.warning(f'Falha em VIF/Logit: {e}')
        D, REG_DUMMIES = pd.DataFrame(index=df.index), pd.DataFrame(index=df.index)

    try:
        build_ols_tables_and_coefplot(df, outdir, D, REG_DUMMIES)
    except Exception as e:
        logger.warning(f'Falha ao gerar Tabela 3.2 / Figura 3.7: {e}')

    if args.check_delta_bins:
        check_delta_bins(df, outdir)

    logger.info('Concluído: pipeline completo e silencioso.')

if __name__ == '__main__':
    main()
