from __future__ import annotations

import argparse, logging, os, re, shutil, subprocess, sys, warnings, unicodedata
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
    "statsmodels==0.14.5\n"
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

def clean_ps_args(argv: List[str]) -> List[str]:
    cleaned = [a for a in argv if a != '^']
    removed = len(argv) - len(cleaned)
    if removed:
        logger.info(f"Removi {removed} argumento(s) '^' (quebra de linha errada no PowerShell).")
    return cleaned

def ensure_running_in_venv():
    if Path(sys.executable).resolve() != PY_PATH.resolve():
        cleaned = clean_ps_args(sys.argv[1:])
        logger.info('Reexecutando dentro de .venv…')
        run([str(PY_PATH), __file__] + cleaned)
        sys.exit(0)
    sys.argv = [sys.argv[0]] + clean_ps_args(sys.argv[1:])

# ----------------------- CLI -----------------------
def parse_args():
    p = argparse.ArgumentParser(description='Pipeline completo: descritivas + VIF/Logit + OLS/coefplot + interações')
    p.add_argument('--csv', required=True, type=Path)
    p.add_argument('--outdir', default=Path('./output'), type=Path)
    p.add_argument('--encoding', default='latin-1')
    p.add_argument('--sep', default=';')
    p.add_argument('--decimal', default=',')
    p.add_argument('--no-xlsx', action='store_true')
    p.add_argument('--max-num-graphs-per-cat', type=int, default=3)
    p.add_argument('--top-k-categories', type=int, default=30)
    # IFDM KDE
    p.add_argument('--ifdm_xmin', type=float, default=-0.2)
    p.add_argument('--ifdm_xmax', type=float, default=0.3)
    p.add_argument('--ifdm_bw_adjust', type=float, default=0.8)
    # mapping manual
    p.add_argument('--map', action='append', default=[], help='Mapeia nomes: VAR="Nome no CSV" (pode repetir)')
    # NOVAS ANALISES
    p.add_argument('--novos', choices=['on', 'off'], default='off', help='Ativar novas análises (default: off)')
    # Novos argumentos para gerar figuras
    p.add_argument('--run-27B-figures', action='store_true', help='Gera as 3 figuras do Logit (Modelo 27B)')
    p.add_argument('--run-27C-figures', action='store_true', help='Gera as 2 figuras do OLS (Modelo 27C)')
    p.add_argument('--run-figures-all', action='store_true', help='Executa a geração de todas as figuras (27B e 27C)')
    p.add_argument('--base', type=Path, help='Caminho para o arquivo de base de dados a ser usado nas figuras')
    return p.parse_args()

args = parse_args()
# ensure_venv_created_and_healthy()
# ensure_deps()
# ensure_running_in_venv()
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

# >>> NOVO: whitelist das TABELAS QUE ENTRAM NA TESE <<<
WRITE_TABLES_WHITELIST = {
    'Tabela_3_0_Diagnostico_Multicolinearidade_VIF',
    'Tabela_3_1_Determinantes_Reeleicao_Logit',
    'Tabela_3_2_Determinantes_DeltaVantagem_OLS',
}

class Numberer:
    def __init__(self):
        self.tab = 1; self.fig = 1
    @staticmethod
    def slug(text: str) -> str:
        text = re.sub(r'[\s]+','_', str(text).strip())
        text = re.sub(r'[^0-9A-Za-z_\-À-ÿ]', '', text)
        return text[:120]
    def table(self, logical_name: str, ext: str) -> str:
        name = f"{self.slug(logical_name)}.{ext}"; return name
    def figure(self, logical_name: str) -> str:
        name = f"{self.slug(logical_name)}.png"; return name

NUM = Numberer()

def normalize(text: str) -> str:
    if text is None: return ''
    t = unicodedata.normalize('NFKD', str(text))
    t = ''.join(ch for ch in t if not unicodedata.combining(ch))
    t = re.sub(r'[^0-9a-zA-Z]+', '', t)
    return t.lower()

def br_to_float_series(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = (s.str.replace('%','', regex=False)
           .str.replace('‰','', regex=False)
           .str.replace('\u00A0','', regex=False)
           .str.replace(' ', '', regex=False))
    s = s.str.replace(r"\.(?=\d{3}(\D|$))", '', regex=True)
    s = s.str.replace(',', '.', regex=False)
    return pd.to_numeric(s, errors='coerce')

def br_to_float(s: pd.Series, decimal: str=',') -> pd.Series:
    x = s.astype(str).str.strip()
    x = (x.str.replace('%','', regex=False)
           .str.replace('\u00A0','', regex=False)
           .str.replace(' ', '', regex=False))
    x = x.str.replace(r"\.(?=\d{3}(\D|$))", '', regex=True)
    if decimal != '.':
        x = x.str.replace(decimal, '.', regex=False)
    return pd.to_numeric(x, errors='coerce')

def to_binary(series: pd.Series) -> pd.Series:
    low = series.astype(str).str.strip().str.lower()
    yes = {'sim','true','yes','1','y','t','masculino','m'}
    no  = {'não','nao','false','no','0','n','f','feminino','f'}
    return pd.Series(np.where(low.isin(yes), 1, np.where(low.isin(no), 0, np.nan)), index=series.index, dtype='float')

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
    if nome not in WRITE_TABLES_WHITELIST:
        logger.debug(f"[SKIP] Tabela fora da tese: {nome}")
        return
    csv_name = NUM.table(nome, 'csv')
    df.to_csv(outdir / csv_name, index=False, encoding='utf-8')
    if HAS_XLSX:
        try:
            xlsx_name = NUM.table(nome, 'xlsx')
            df.to_excel(outdir / xlsx_name, index=False)
        except Exception as e:
            logger.warning(f'Falha ao salvar XLSX ({nome}): {e} — seguindo com CSV.')

def save_figure(fig, outdir: Path, nome: str):
    if nome not in expected_suffixes:
        logger.debug(f"[SKIP] Figura fora da tese: {nome}")
        return
    fname = NUM.figure(nome); fig.tight_layout(); fig.savefig(outdir / fname); plt.close(fig)

def frequency_table(series: pd.Series, dropna=False) -> pd.DataFrame:
    vc = series.value_counts(dropna=dropna); total = series.shape[0]
    return pd.DataFrame({'valor': vc.index.astype(str), 'contagem': vc.values, 'percentual': (vc.values/total)*100})

def find_by_keywords(df: pd.DataFrame, *keywords: str) -> Optional[str]:
    keys = [normalize(k) for k in keywords if k]
    for c in df.columns:
        nc = normalize(c)
        if all(k in nc for k in keys):
            return c
    return None

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
    # [SKIP] Bloco removido na limpeza — fora do escopo da tese
    logger.debug(f"[SKIP] plot_cat_count({titulo}) — figura não está na tese")
    pass

def plot_cat_numeric_bar(df: pd.DataFrame, cat_col: str, num_col: str, how: str, titulo: str, outdir: Path, topk: int):
    # [SKIP] Bloco removido na limpeza — fora do escopo da tese
    logger.debug(f"[SKIP] plot_cat_numeric_bar({cat_col}, {num_col}) — figura não está na tese")
    pass

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
    # Todas as TABELAS genéricas deste bloco estão FORA da tese → não salvar nada
    logger.debug(f"[SKIP] save_categorical_block({cat_col}) — tabelas não entram na tese")
    return

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
    logger.info('[FIG] Gerando Figura 3.1 — Reeleição (%) por Região')
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
        logger.info('[FIG] Gerando Figura 3.2 — Distribuição Δ Vantagem (2020-2016)')
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
        logger.info('[FIG] Gerando Figura 3.3 — Precocidade das NPIs (Comércio/Máscara)')
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
        # fora da tese → não grava tabela
        logger.debug("[SKIP] infraestrutura_percentuais — tabela fora da tese")
        logger.info('[FIG] Gerando Figura 3.4 — Infraestrutura de Saúde (% por ação)')
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
        # fora da tese → não grava tabela
        logger.debug("[SKIP] estresse_saude_percentuais — tabela fora da tese")
        logger.info('[FIG] Gerando Figura 3.5 — Estresse no Sistema de Saúde')
        fig, ax = plt.subplots(figsize=(10,6))
        sns.barplot(data=tab, y='Indicador de Estresse', x='Percentual (%)', ax=ax, orient='h')
        for i,v in enumerate(tab['Percentual (%)']): ax.text(float(v)+0.5, i, f"{float(v):.1f}%", va='center')
        ax.set_xlabel('Percentual (%)'); ax.set_ylabel('')
        ax.set_xlim(0,100)
        save_figure(fig, outdir, 'fig_3_5_estresse_saude')
    return {r['Indicador de Estresse']: r['Percentual (%)'] for r in rows}

# --- IFDM ---
def find_ifdm_delta_cols(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    import unicodedata

    def norm(s: str) -> str:
        s = unicodedata.normalize('NFKD', str(s))
        s = ''.join(ch for ch in s if not unicodedata.combining(ch))
        s = s.replace('Δ', 'delta')     # chave: tratar símbolo delta
        s = s.replace('–', '-').replace('—', '-')  # normalizar travessões
        return s.lower()

    emp, sau = None, None
    for c in df.columns:
        lc = norm(c)
        has_ifdm = ('ifdm' in lc) or ('firjan' in lc)
        has_emp  = ('emprego' in lc) or ('renda' in lc)
        has_sau  = ('saude' in lc) or ('saúde' in lc)
        has_var  = ('varia' in lc) or ('delta' in lc) or ('2016-2020' in lc) or ('2020-2016' in lc) or ('16-20' in lc) or ('20-16' in lc)
        if has_ifdm and has_emp and has_var and emp is None:
            emp = c
        if has_ifdm and has_sau and has_var and sau is None:
            sau = c
    return emp, sau

def build_ifdm_outputs(df: pd.DataFrame, outdir: Path) -> Dict[str, float]:
    c_emp, c_sau = find_ifdm_delta_cols(df)
    logger.info(f"[IFDM] Emprego&Renda ← {c_emp if c_emp else 'NÃO ENCONTRADO'}")
    logger.info(f"[IFDM] Saúde        ← {c_sau if c_sau else 'NÃO ENCONTRADO'}")
    rows = []
    s_emp = pd.Series(dtype=float); s_sau = pd.Series(dtype=float)
    if c_emp is not None:
        s_emp = br_to_float_series(df[c_emp]).dropna()
        rows.append({'Indicador':'IFDM Emprego & Renda','Média': float(s_emp.mean()),'DP': float(s_emp.std(ddof=1)),'% Piora': float((s_emp<0).mean()*100),'N': int(len(s_emp))})
    if c_sau is not None:
        s_sau = br_to_float_series(df[c_sau]).dropna()
        rows.append({'Indicador':'IFDM Saúde','Média': float(s_sau.mean()),'DP': float(s_sau.std(ddof=1)),'% Piora': float((s_sau<0).mean()*100),'N': int(len(s_sau))})
    if s_emp.empty and s_sau.empty:
        logger.warning("[IFDM] Nenhuma série encontrada — verifique nomes/mapeamentos.")
        return {}
    if rows:
        tab = pd.DataFrame(rows)
        # fora da tese → não grava tabela
        logger.debug("[SKIP] ifdm_variacoes_stats — tabela fora da tese")
        if not s_emp.empty or not s_sau.empty:
            logger.info('[FIG] Gerando Figura 3.6 — Variações no IFDM (Emprego/Saúde)')
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
    'Gestao': ['Gestao', 'indice_gestao'],
    'DeltaAbstencao_c': ['DeltaAbstencao_c', 'delta_abstencao_c'],
    'DeltaCompeticao_c': ['DeltaCompeticao_c', 'delta_competicao_c'],
    'Vantagem2016_c': ['Vantagem2016_c', 'vantagem_2016_c'],
    'Gestao': ['Gestao', 'indice_gestao'],
    'DeltaAbstencao_c': ['DeltaAbstencao_c', 'delta_abstencao_c'],
    'DeltaCompeticao_c': ['DeltaCompeticao_c', 'delta_competicao_c'],
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
        {'Variável':'Pseudo R2 (McFadden)','(1) Base': f"{res1.prsquared:.4f}",' (2) + Gestão': f"{res2.prsquared:.4f}",'(3) Completo': f"{res3.prsquared:.4f}"}
    ])

    write_table(T31, outdir, 'Tabela_3_1_Determinantes_Reeleicao_Logit')

    return D, REG_DUMMIES, T31, res1, res2, res3, X3, y3

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
    # fora da tese → não grava diagnóstico auxiliar do coefplot
    logger.debug("[SKIP] diagnostico_modelo6_variaveis — tabela fora da tese")

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

        logger.info('[FIG] Gerando Figura 3.7 — Coeficientes OLS Modelo 6 (Coefplot)')
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

    return T32, r4, r5, r6, X6, y6

# ----------------------- PARTE 2: Interações (H5a e V2016×ΔNEC) -----------------------
def build_parte2_outputs(df: pd.DataFrame, outdir: Path):
    logger.info('Iniciando Parte 2: Interações finais + 2 gráficos')

    # Reeleição
    col_ree = find_by_keywords(df, 'reeleito') or find_by_keywords(df, 'reeleicao')
    if col_ree is None:
        logger.warning('Não encontrei a coluna de reeleição para Parte 2')
        return
    Reeleicao = df[col_ree].map(to01).astype(float)

    # Vantagem 2016
    col_v2016 = find_by_keywords(df, 'vantagem', '2016')
    if col_v2016 is None:
        logger.warning('Não encontrei Vantagem 2016 para Parte 2')
        return
    V2016 = br_to_float(df[col_v2016], args.decimal)

    # Δ Vantagem (2020-2016)
    col_dv = find_delta_vant_col(df)
    if col_dv is None:
        logger.warning('Não encontrei delta_vantagem para Parte 2')
        return
    DeltaV    = br_to_float(df[col_dv], args.decimal)

    # Abstenção 2016/2020 → Δ Abstenção (p.p.)
    col_abs16 = find_by_keywords(df, 'abstencao','2016')
    col_abs20 = find_by_keywords(df, 'abstencao','2020')
    if col_abs16 is None or col_abs20 is None:
        logger.warning('Não encontrei Abstenção 2016/2020 para Parte 2')
        return
    DeltaAbst = br_to_float(df[col_abs20], args.decimal) - br_to_float(df[col_abs16], args.decimal)

    # Δ Competição (usa NEC 2016/2020)
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
        DNEC_diff = dd
        DNEC_log  = dd - dd.mean()
    else:
        logger.warning('Não encontrei NEC para Parte 2')
        return

    # Índice de Gestão
    parts = []
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
            parts.append(v)
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

    if not parts:
        logger.warning('Nenhum componente para Índice de Gestão (Parte 2)')
        return
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

    # LOGIT: Reeleição ~ Gestão_c * ΔAbst_c
    Xlog = pd.DataFrame({
        'const': 1.0,
        'Gestao_c': Gestao_c,
        'delta_abstencao_pp_c': DeltaAbst_c,
        'Gestao_c:delta_abstencao_pp_c': Gestao_c*DeltaAbst_c,
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

    # Tabelas
    T34a = pd.DataFrame({
        'variavel': res_logit.params.index,
        'coef': res_logit.params.values,
        'se': res_logit.bse.values,
        't_z': res_logit.tvalues,
        'pvalue': res_logit.pvalues,
    })
    logger.debug("[SKIP] Tabela_3_4a_logit_gestao_x_deltaabst — fora da tese (não salva)")

    # AMEs
    try:
        mfx = res_logit.get_margeff(at='overall', method='dydx')
        sf = mfx.summary_frame()
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
        keep = ['Unnamed: 0','dy/dx','Std, Err,','z','Pr(>|z|)','Conf, Int, Low','Cont, Int, Hi,']
        for k in keep:
            if k not in T34b.columns:
                T34b[k] = np.nan
        T34b = T34b[keep]
        logger.debug("[SKIP] Tabela_3_4b_marginais_logit_gestao_x_deltaabst — fora da tese (não salva)")
    except Exception as e:
        logger.warning(f'AMEs não exportadas (Parte 2): {e}')

    # OLS (coeficientes)
    T34c = pd.DataFrame({
        'variavel': res_ols.params.index,
        'coef': res_ols.params.values,
        'se': res_ols.bse.values,
        't_z': res_ols.tvalues,
        'pvalue': res_ols.pvalues,
    })
    logger.debug("[SKIP] Tabela_3_4c_ols_v2016_x_dnec_log — fora da tese (não salva)")

    # Robustez
    UF_series = UF
    if POP.isna().all():
        top5_idx = df.index
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
    n,b,se,p = fit_take(DeltaV, V2016_c, DNEC_log_c, groups=UF_series)
    rows.append({'modelo':'FULL_log_clusterUF','comp_col':'delta_competicao_log_c','N':n,'coef_interacao':b,'se':se,'pvalue':p})
    n,b,se,p = fit_take(DeltaV, V2016_c, DNEC_log_c)
    rows.append({'modelo':'FULL_log_HC1','comp_col':'delta_competicao_log_c','N':n,'coef_interacao':b,'se':se,'pvalue':p})

    if 'DNEC_diff' in locals() and DNEC_diff is not None:
        DNEC_diff_c = DNEC_diff - DNEC_diff.mean()
        n,b,se,p = fit_take(DeltaV, V2016_c, DNEC_diff_c, groups=UF_series)
        rows.append({'modelo':'FULL_diff_clusterUF','comp_col':'delta_competicao_diff_c','N':n,'coef_interacao':b,'se':se,'pvalue':p})
        n,b,se,p = fit_take(DeltaV, V2016_c, DNEC_diff_c)
        rows.append({'modelo':'FULL_diff_HC1','comp_col':'delta_competicao_diff_c','N':n,'coef_interacao':b,'se':se,'pvalue':p})

    mask_top = df.index.isin(top5_idx)
    n,b,se,p = fit_take(DeltaV[mask_top], V2016_c[mask_top], DNEC_log_c[mask_top], groups=UF_series[mask_top])
    rows.append({'modelo':'TOP5_log_clusterUF','comp_col':'delta_competicao_log_c','N':n,'coef_interacao':b,'se':se,'pvalue':p})
    n,b,se,p = fit_take(DeltaV[mask_top], V2016_c[mask_top], DNEC_log_c[mask_top])
    rows.append({'modelo':'TOP5_log_HC1','comp_col':'delta_competicao_log_c','N':n,'coef_interacao':b,'se':se,'pvalue':p})

    if 'DNEC_diff_c' in locals():
        n,b,se,p = fit_take(DeltaV[mask_top], V2016_c[mask_top], DNEC_diff_c[mask_top], groups=UF_series[mask_top])
        rows.append({'modelo':'TOP5_diff_clusterUF','comp_col':'delta_competicao_diff_c','N':n,'coef_interacao':b,'se':se,'pvalue':p})
        n,b,se,p = fit_take(DeltaV[mask_top], V2016_c[mask_top], DNEC_diff_c[mask_top])
        rows.append({'modelo':'TOP5_diff_HC1','comp_col':'delta_competicao_diff_c','N':n,'coef_interacao':b,'se':se,'pvalue':p})

    logger.debug("[SKIP] Tabela_3_4d_robustez_competicao — fora da tese (não salva)")

    # [SKIP] Figuras removidas na limpeza — fora do escopo da tese
    logger.info("[SKIP] Figuras Parte 2 (fig_pred_logit_gestao_x_deltaabst, fig_pred_ols_v2016_x_dnec_log) — não estão na tese")

    logger.info('Parte 2 concluída')

# ----------------------- PARTE 3: H3/H4 -----------------------
def build_parte3_outputs(df: pd.DataFrame, outdir: Path):
    logger.info('Iniciando Parte 3: H3/H4 com autodetecção')

    # Resolver colunas
    col_estado = find_by_keywords(df, 'estado') or find_by_keywords(df, 'uf')
    col_regiao = find_by_keywords(df, 'regiao', 'pais') or find_by_keywords(df, 'regiao')
    col_ree    = find_by_keywords(df, 'reeleito') or find_by_keywords(df, 'reeleicao')
    col_v2016  = find_by_keywords(df, 'vantagem','2016')
    col_dv     = find_delta_vant_col(df)
    col_lnpop  = find_by_keywords(df, 'ln','pop') or find_by_keywords(df, 'log','pop')
    col_pop    = find_by_keywords(df, 'populacao')

    if not col_lnpop:
        if col_pop:
            s = br_to_float(df[col_pop], args.decimal)
            df['ln_pop'] = np.log(s)
            col_lnpop = 'ln_pop'
        else:
            logger.warning('Não encontrei ln_pop nem População para Parte 3')
            return

    if not col_ree or not col_v2016 or not col_dv:
        logger.warning('Colunas essenciais faltando para Parte 3')
        return

    # Construir índices Gestão e Severidade
    parts = []
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

    if parts:
        Gestao_Index = pd.concat(parts, axis=1).mean(axis=1, skipna=True)
    else:
        Gestao_Index = pd.Series(np.nan, index=df.index)

    # Severidade
    sev_defs = [
        ('Sobrecarga UTI', ['sobrecarga','leitos','uti']),
        ('Transferência de pacientes', ['transfer','pacien']),
        ('>24h sem internação', ['24','sem','intern'])
    ]
    parts_s = []
    for label, keys in sev_defs:
        c = find_by_keywords(df, *keys)
        if c:
            v = df[c].map(to01).astype(float)
            parts_s.append(v)

    if parts_s:
        Severidade_Index = pd.concat(parts_s, axis=1).mean(axis=1, skipna=True)
    else:
        Severidade_Index = pd.Series(np.nan, index=df.index)

    # Preparar dados
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

    # H3: ln_pop como moderador
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

    # H4: V2016 como moderador
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

    # Salvar tabelas
    basefile = args.csv.stem

    def dump_res(name: str, res):
        tab = []
        for k in res.params.index:
            tab.append({'variavel': k, 'coef': float(res.params[k]), 'se': float(res.bse[k]), 'pval': float(res.pvalues[k])})
        pd.DataFrame(tab).to_csv(outdir / f"{basefile}__{name}_tabela.csv", index=False, encoding='utf-8')
        (outdir / f"{basefile}__{name}_sumario.txt").write_text(res.summary().as_text(), encoding='utf-8')

    dump_res('H3_logit', res_h3_logit)
    dump_res('H3_ols',   res_h3_ols)
    dump_res('H4_logit', res_h4_logit)
    dump_res('H4_ols',   res_h4_ols)

    # Figuras
    q_ln = np.quantile(BASE['ln_pop_c'].dropna(), [0.1,0.5,0.9]) if BASE['ln_pop_c'].notna().any() else [0,0,0]
    G_levels = [BASE['Gestao_c'].mean()+BASE['Gestao_c'].std(), BASE['Gestao_c'].mean()-BASE['Gestao_c'].std()]
    S_levels = [BASE['Sever_c'].mean()+BASE['Sever_c'].std(), BASE['Sever_c'].mean()-BASE['Sever_c'].std()]

    # H3 logit - Gestão
    logger.info('[FIG] Gerando H3 Logit — Prob. prevista vs ln(Pop), Gestão ±1dp')
    fig = plt.figure()
    for g in G_levels:
        Xp = pd.DataFrame({'const':1.0,'Gestao_c':g,'Sever_c':BASE['Sever_c'].mean(),'ln_pop_c':q_ln,'Gestao_c:ln_pop_c':g*q_ln,'Sever_c:ln_pop_c':BASE['Sever_c'].mean()*q_ln})
        pr = res_h3_logit.predict(Xp)
        plt.plot(q_ln, pr, marker='o', linewidth=2, label=f'Gestão {"+1dp" if g>0 else "-1dp"}')
    plt.xlabel('ln(Pop)'); plt.ylabel('Prob. prevista'); plt.legend(); plt.tight_layout()
    plt.savefig(outdir / f"{basefile}__H3_logit_lnpop.png", dpi=200); plt.close()

    # H3 logit - Severidade
    logger.info('[FIG] Gerando H3 Logit — Prob. prevista vs ln(Pop), Severidade ±1dp')
    fig = plt.figure()
    for s in S_levels:
        Xp = pd.DataFrame({'const':1.0,'Gestao_c':BASE['Gestao_c'].mean(),'Sever_c':s,'ln_pop_c':q_ln,'Gestao_c:ln_pop_c':BASE['Gestao_c'].mean()*q_ln,'Sever_c:ln_pop_c':s*q_ln})
        pr = res_h3_logit.predict(Xp)
        plt.plot(q_ln, pr, marker='o', linewidth=2, label=f'Severidade {"+1dp" if s>0 else "-1dp"}')
    plt.xlabel('ln(Pop)'); plt.ylabel('Prob. prevista'); plt.legend(); plt.tight_layout()
    plt.savefig(outdir / f"{basefile}__H3_logit_lnpop_sev.png", dpi=200); plt.close()

    # [SKIP] Figuras H4 removidas na limpeza — fora do escopo da tese
    logger.info("[SKIP] Figuras H4 (__H4_logit_v2016.png, __H4_logit_v2016_sev.png) — não estão na tese")

    logger.info('Parte 3 concluída')

# ----------------------- extras utilitários -----------------------
def save_freq_and_agg_blocks(df: pd.DataFrame, type_map: Dict[str, str], outdir: Path, topk: int, max_num: int):
    # Todas as figuras/tabelas genéricas categóricas estão FORA da tese → não gerar nada
    logger.debug("[SKIP] save_freq_and_agg_blocks — todas as figuras/tabelas genéricas fora da tese")
    return

def check_delta_bins(df: pd.DataFrame, outdir: Path, bins_list=(20,25,30)):
    # [SKIP] Bloco removido na limpeza — fora do escopo da tese
    logger.info("[SKIP] check_delta_bins — figuras de diagnóstico não estão na tese")
    pass

# ----------------------- validação de figuras (whitelist) -----------------------
def validate_figures_whitelist(outdir: Path):
    logger.info('=' * 60)
    logger.info('VALIDAÇÃO: Verificando figuras geradas (whitelist)')
    logger.info('=' * 60)

    # Lista branca: padrões esperados (sufixos dos nomes lógicos)
    # Obs: Numberer adiciona prefixo numérico, então validamos por sufixo
    expected_suffixes = [
        'fig_3_1_reeleicao_por_regiao.png',
        'fig_3_2_distribuicao_delta_vantagem.png',
        'fig_3_3_precocidade_npis.png',
        'fig_3_4_infraestrutura.png',
        'fig_3_5_estresse_saude.png',
        'fig_3_6_ifdm_variacoes.png',
        'fig_3_7_coeficientes_ols_completo.png',
        'coeficientes_ols_completo.png',  # duplicata da 3.7 (sem numeração)
        'Base_VALIDADA_E_PRONTA__27A_prepared__fig-3-2-2a-forest-logit.png',
        'Base_VALIDADA_E_PRONTA__27A_prepared__fig-3-2-2b-ames-logit.png',
        'Base_VALIDADA_E_PRONTA__27A_prepared__fig-3-2-2c-pred-prob_gestao_x_dabst.png',
        'Base_VALIDADA_E_PRONTA__27A_prepared__fig-3-2-3a-forest-ols.png',
        'Base_VALIDADA_E_PRONTA__27A_prepared__fig-3-2-3b-pred-val_dcomp_x_v2016.png',
    ]
    # H3 logit: nomes completos (variam por basefile, mas terminam assim)
    h3_patterns = ['__H3_logit_lnpop.png', '__H3_logit_lnpop_sev.png']

    # Listar todos os .png gerados
    all_pngs = sorted(outdir.glob('*.png'))

    # Validar cada PNG
    found_figs = {
        'fig_3_1': False, 'fig_3_2': False, 'fig_3_3': False, 'fig_3_4': False,
        'fig_3_5': False, 'fig_3_6': False, 'fig_3_7': False, 'fig_3_7_dup': False,
        'h3_lnpop': False, 'h3_lnpop_sev': False
    }
    unexpected = []

    for png in all_pngs:
        name = png.name
        matched = False

        # Verificar sufixos esperados (figuras 3.1-3.7)
        for suff in expected_suffixes:
            if name.endswith(suff):
                matched = True
                # Mapear para chaves de rastreamento
                if 'fig_3_1' in suff: found_figs['fig_3_1'] = True
                elif 'fig_3_2' in suff: found_figs['fig_3_2'] = True
                elif 'fig_3_3' in suff: found_figs['fig_3_3'] = True
                elif 'fig_3_4' in suff: found_figs['fig_3_4'] = True
                elif 'fig_3_5' in suff: found_figs['fig_3_5'] = True
                elif 'fig_3_6' in suff: found_figs['fig_3_6'] = True
                elif 'fig_3_7' in suff: found_figs['fig_3_7'] = True
                elif name == 'coeficientes_ols_completo.png': found_figs['fig_3_7_dup'] = True
                break

        # Verificar padrões H3
        if not matched:
            for pat in h3_patterns:
                if name.endswith(pat):
                    matched = True
                    if '__H3_logit_lnpop_sev.png' in name: found_figs['h3_lnpop_sev'] = True
                    elif '__H3_logit_lnpop.png' in name: found_figs['h3_lnpop'] = True
                    break

        if not matched:
            unexpected.append(name)

    # Log de status
    logger.info('Figuras esperadas encontradas:')
    for key, status in found_figs.items():
        logger.info(f"  {key}: {'✓' if status else '✗'}")

    if unexpected:
        logger.error('ERRO: Figuras FORA DA LISTA BRANCA detectadas:')
        for u in unexpected:
            logger.error(f"  - {u}")
        raise RuntimeError(
            f"Validação FALHOU: {len(unexpected)} figura(s) fora da tese detectada(s). "
            f"Revise o código para garantir que apenas as 9 figuras-alvo sejam geradas."
        )

    # Verificar se todas as esperadas foram encontradas (warning, não erro)
    missing = [k for k, v in found_figs.items() if not v]
    if missing:
        logger.warning(f"Atenção: Algumas figuras esperadas não foram encontradas: {missing}")
    else:
        logger.info('✓ Validação APROVADA: Apenas as 9 figuras-alvo foram geradas.')

    logger.info('=' * 60)

# ----------------------- Novas Funções de Figuras -----------------------
def save_csv(df, outdir, name, base_name):
    file_name = f"{base_name}__{name}.csv"
    df.to_csv(outdir / file_name, sep=';', decimal=',', index=False)
    logger.info(f"CSV salvo em: {outdir / file_name}")

def generate_logit_forest_plot(res, outdir, base_name):
    params = res.params.drop('const')
    conf = res.conf_int().drop('const')
    odds_ratios = np.exp(params)
    conf_int_or = np.exp(conf)
    
    df_plot = pd.DataFrame({
        'variable': params.index,
        'odds_ratio': odds_ratios,
        'ci_lower': conf_int_or.iloc[:, 0],
        'ci_upper': conf_int_or.iloc[:, 1]
    })
    
    save_csv(df_plot, outdir, 'fig-3-2-2a-forest-logit-data', base_name)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.errorbar(df_plot['odds_ratio'], df_plot['variable'], xerr=[df_plot['odds_ratio'] - df_plot['ci_lower'], df_plot['ci_upper'] - df_plot['odds_ratio']], fmt='o', capsize=5)
    ax.axvline(1, color='red', linestyle='--')
    ax.set_xlabel('Odds Ratio')
    ax.set_title('Modelo 27B: Forest Plot (Odds Ratios)')
    save_figure(fig, outdir, f'{base_name}__fig-3-2-2a-forest-logit')

def generate_logit_ames_plot(res, X, outdir, base_name):
    ames = res.get_margeff(at='mean', method='dydx')
    ames_summary = ames.summary_frame()
    
    save_csv(ames_summary, outdir, 'fig-3-2-2b-ames-logit-data', base_name)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.errorbar(ames_summary['dy/dx'], ames_summary.index, xerr=1.96 * ames_summary['Std. Err.'], fmt='o', capsize=5)
    ax.axvline(0, color='red', linestyle='--')
    ax.set_xlabel('Efeito Marginal Médio')
    ax.set_title('Modelo 27B: Efeitos Marginais Médios (AMEs)')
    save_figure(fig, outdir, f'{base_name}__fig-3-2-2b-ames-logit')

def generate_logit_interaction_plot(res, df, outdir, base_name):
    # Encontrar as colunas corretas
    gestao_col = find_by_keywords(df, 'Gestao')
    delta_abst_col = find_by_keywords(df, 'DeltaAbstencao_c')

    if not gestao_col or not delta_abst_col:
        logger.warning("Colunas 'Gestao' ou 'DeltaAbstencao_c' não encontradas para o gráfico de interação.")
        return

    # Criar um grid de valores para a interação
    interact_df = pd.DataFrame({
        gestao_col: np.repeat(np.linspace(df[gestao_col].min(), df[gestao_col].max(), 20), 20),
        delta_abst_col: np.tile(np.linspace(df[delta_abst_col].min(), df[delta_abst_col].max(), 20), 20)
    })

    # Manter outras variáveis em suas médias
    other_vars = [col for col in res.model.exog_names if col not in ['const', gestao_col, delta_abst_col]]
    for var in other_vars:
        interact_df[var] = df[var].mean()
        
    interact_df = sm.add_constant(interact_df, has_constant='add')
    
    # Prever as probabilidades
    interact_df['predicted_prob'] = res.predict(interact_df[res.model.exog_names])
    
    save_csv(interact_df, outdir, 'fig-3-2-2c-pred-prob_gestao_x_dabst-data', base_name)

    fig, ax = plt.subplots(figsize=(10, 8))
    contour = ax.contourf(
        np.linspace(df[gestao_col].min(), df[gestao_col].max(), 20),
        np.linspace(df[delta_abst_col].min(), df[delta_abst_col].max(), 20),
        interact_df['predicted_prob'].values.reshape(20, 20),
        cmap='viridis'
    )
    fig.colorbar(contour, ax=ax, label='Probabilidade Prevista de Reeleição')
    ax.set_xlabel(gestao_col)
    ax.set_ylabel(delta_abst_col)
    ax.set_title('Modelo 27B: Probabilidades Previstas (Gestao x ΔAbstenção_c)')
    save_figure(fig, outdir, f'{base_name}__fig-3-2-2c-pred-prob_gestao_x_dabst')

def generate_ols_forest_plot(res, outdir, base_name):
    params = res.params.drop('const')
    conf = res.conf_int().drop('const')
    
    df_plot = pd.DataFrame({
        'variable': params.index,
        'coefficient': params,
        'ci_lower': conf.iloc[:, 0],
        'ci_upper': conf.iloc[:, 1]
    })
    
    save_csv(df_plot, outdir, 'fig-3-2-3a-forest-ols-data', base_name)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.errorbar(df_plot['coefficient'], df_plot['variable'], xerr=[df_plot['coefficient'] - df_plot['ci_lower'], df_plot['ci_upper'] - df_plot['coefficient']], fmt='o', capsize=5)
    ax.axvline(0, color='red', linestyle='--')
    ax.set_xlabel('Coeficiente')
    ax.set_title('Modelo 27C: Forest Plot (Coeficientes)')
    save_figure(fig, outdir, f'{base_name}__fig-3-2-3a-forest-ols')

def generate_ols_interaction_plot(res, df, outdir, base_name):
    # Encontrar as colunas corretas
    vantagem_col = find_by_keywords(df, 'Vantagem2016_c')
    delta_comp_col = find_by_keywords(df, 'DeltaCompeticao_c')

    if not vantagem_col or not delta_comp_col:
        logger.warning("Colunas 'Vantagem2016_c' ou 'DeltaCompeticao_c' não encontradas para o gráfico de interação.")
        return

    # Criar um grid de valores para a interação
    interact_df = pd.DataFrame({
        vantagem_col: np.repeat(np.linspace(df[vantagem_col].min(), df[vantagem_col].max(), 20), 20),
        delta_comp_col: np.tile(np.linspace(df[delta_comp_col].min(), df[delta_comp_col].max(), 20), 20)
    })

    # Manter outras variáveis em suas médias
    other_vars = [col for col in res.model.exog_names if col not in ['const', vantagem_col, delta_comp_col]]
    for var in other_vars:
        interact_df[var] = df[var].mean()
        
    interact_df = sm.add_constant(interact_df, has_constant='add')
    
    # Prever os valores
    interact_df['predicted_value'] = res.predict(interact_df[res.model.exog_names])
    
    save_csv(interact_df, outdir, 'fig-3-2-3b-pred-val_dcomp_x_v2016-data', base_name)

    fig, ax = plt.subplots(figsize=(10, 8))
    contour = ax.contourf(
        np.linspace(df[vantagem_col].min(), df[vantagem_col].max(), 20),
        np.linspace(df[delta_comp_col].min(), df[delta_comp_col].max(), 20),
        interact_df['predicted_value'].values.reshape(20, 20),
        cmap='viridis'
    )
    fig.colorbar(contour, ax=ax, label='Valor Previsto de ΔVantagem')
    ax.set_xlabel(vantagem_col)
    ax.set_ylabel(delta_comp_col)
    ax.set_title('Modelo 27C: Valores Previstos (Vantagem2016_c x ΔCompeticao_c)')
    save_figure(fig, outdir, f'{base_name}__fig-3-2-3b-pred-val_dcomp_x_v2016')

# ----------------------- cola final / main -----------------------
def main():
    logger.info('=' * 60)
    logger.info('MASTER.PY — Pipeline Completo Integrado')
    logger.info('=' * 60)

    if args.base:
        df = load_csv(args.base, encoding=args.encoding, sep=args.sep, decimal=args.decimal)
    else:
        df = load_csv(args.csv, encoding=args.encoding, sep=args.sep, decimal=args.decimal)
        
    df = clean_labels(df)
    TYPE_MAP = build_type_map(list(df.columns))

    outdir = args.outdir; outdir.mkdir(parents=True, exist_ok=True)

    logger.info('Iniciando Parte 1: Análises descritivas + VIF/Logit + OLS')
    write_table(df, outdir, 'dataset_original')
    build_reelection_outputs(df, outdir)
    build_delta_vantagem_outputs(df, outdir)
    save_freq_and_agg_blocks(df, TYPE_MAP, outdir, args.top_k_categories, args.max_num_graphs_per_cat)
    dict_df = dictionary_vars(df, TYPE_MAP); write_table(dict_df, outdir, 'dicionario_variaveis')
    build_npi_infra_outputs(df, outdir)
    build_stress_outputs(df, outdir)
    build_ifdm_outputs(df, outdir)

    try:
        D, REG_DUMMIES, _, res1, res2, res3, X3, y3 = build_vif_and_logit_tables(df, outdir)
    except Exception as e:
        logger.warning(f'Falha em VIF/Logit: {e}')
        D, REG_DUMMIES, res1, res2, res3, X3, y3 = pd.DataFrame(index=df.index), pd.DataFrame(index=df.index), None, None, None, None, None

    try:
        _, r4, r5, r6, X6, y6 = build_ols_tables_and_coefplot(df, outdir, D, REG_DUMMIES)
    except Exception as e:
        logger.warning(f'Falha ao gerar Tabela 3.2 / Figura 3.7: {e}')
        r4, r5, r6, X6, y6 = None, None, None, None, None

    logger.info('Parte 1 concluída')

    # Parte 2
    try:
        build_parte2_outputs(df, outdir)
    except Exception as e:
        logger.warning(f'Falha na Parte 2: {e}')

    # Parte 3
    try:
        build_parte3_outputs(df, outdir)
    except Exception as e:
        logger.warning(f'Falha na Parte 3: {e}')

    # Geração de figuras
    if args.run_27B_figures or args.run_figures_all:
        if res3 is not None and X3 is not None:
            logger.info("Gerando figuras para o Modelo 27B (Logit)")
            base_name = args.base.stem if args.base else args.csv.stem
            generate_logit_forest_plot(res3, outdir, base_name)
            generate_logit_ames_plot(res3, X3, outdir, base_name)
            generate_logit_interaction_plot(res3, df, outdir, base_name)
            # Salvar IDs
            pd.DataFrame({'ID': X3.index}).to_csv(outdir / f'{base_name}_sample_ids__27B.csv', index=False, sep=';')


    if args.run_27C_figures or args.run_figures_all:
        if r6 is not None and X6 is not None:
            logger.info("Gerando figuras para o Modelo 27C (OLS)")
            base_name = args.base.stem if args.base else args.csv.stem
            generate_ols_forest_plot(r6, outdir, base_name)
            generate_ols_interaction_plot(r6, df, outdir, base_name)
            # Salvar IDs
            pd.DataFrame({'ID': X6.index}).to_csv(outdir / f'{base_name}_sample_ids__27C.csv', index=False, sep=';')

    # Validação de figuras (whitelist)
    try:
        validate_figures_whitelist(outdir)
    except RuntimeError as e:
        logger.error(f'VALIDAÇÃO FALHOU: {e}')
        raise

    # === NOVAS ANÁLISES ===
    if args.novos == 'on':
        logger.info('=' * 60)
        logger.info('INICIANDO NOVAS ANÁLISES')
        logger.info('=' * 60)
        try:
            run_novas_analises(df, args.csv, outdir)
        except Exception as e:
            logger.warning(f'Erro nas novas análises: {e}')
            import traceback
            logger.error(traceback.format_exc())

    logger.info('=' * 60)
    logger.info('PIPELINE COMPLETO FINALIZADO COM SUCESSO')
    logger.info('=' * 60)

# === NOVAS ANALISES (NAO ALTERAR ACIMA) ===
def run_novas_analises(df: pd.DataFrame, csv_path: Path, outdir: Path):
    import json
    import datetime
    from pathlib import Path

    # Criar pasta novos/
    novos_dir = outdir / 'novos'
    novos_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging para run.log
    log_file = novos_dir / 'run.log'
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
    logger.addHandler(file_handler)

    logger.info(f"=== NOVAS ANÁLISES INICIADAS ===")
    logger.info(f"Data/Hora: {datetime.datetime.now()}")
    logger.info(f"CSV: {csv_path.name}")

    # Prefixo para arquivos (nome do CSV sem extensão)
    prefix = csv_path.stem

    # === MAPEAMENTO DE NOMES DE COLUNAS ===
    # Mapear nomes reais do CSV para nomes esperados pelas análises
    col_map = {
        'reeleito': 'reeleito',  # já está correto
        'delta_vantagem (2020-2016)': 'delta_vantagem',
        'Estado': 'UF',
        'Número efetivo de candidatos de 2020': 'NEC',
        'ln_pop': 'ln_pop',  # já está correto
        'IFDM 2016': 'ifdm_geral',
        'Abstenção 2016': 'abstencao_2016',
        'Abstenção 2020': 'abstencao_2020',
        'Dias da eleição - Máscara (Normalizado)': 'precocidade_mascara',
        'Dias da eleição - Restrição ao comércio (Normalizado)': 'precocidade_comercio',
        'delta_nec (2020-2016)': 'delta_nec'
    }

    # Criar DataFrame com nomes mapeados
    df_work = df.copy()
    rename_dict = {}
    for old_name, new_name in col_map.items():
        if old_name in df_work.columns:
            rename_dict[old_name] = new_name

    df_work.rename(columns=rename_dict, inplace=True)
    logger.info(f"Colunas mapeadas: {list(rename_dict.keys())}")

    # Calcular variáveis derivadas se necessário
    if 'abstencao_2016' in df_work.columns and 'abstencao_2020' in df_work.columns:
        df_work['abstencao_var'] = df_work['abstencao_2020'] - df_work['abstencao_2016']

    # Usar média das precocidades como índice de gestão (simplificado)
    if 'precocidade_mascara' in df_work.columns and 'precocidade_comercio' in df_work.columns:
        df_work['gestao_index'] = (df_work['precocidade_mascara'] + df_work['precocidade_comercio']) / 2
    elif 'precocidade_mascara' in df_work.columns:
        df_work['gestao_index'] = df_work['precocidade_mascara']

    # Usar precocidade_mascara como proxy de precocidade geral
    if 'precocidade_mascara' in df_work.columns:
        df_work['precocidade'] = df_work['precocidade_mascara']

    # Manifesto
    manifest = {
        'timestamp': datetime.datetime.now().isoformat(),
        'csv': str(csv_path),
        'n_amostra': len(df),
        'arquivos_gerados': [],
        'dependencias': {}
    }

    # Registrar versões
    try:
        manifest['dependencias']['pandas'] = pd.__version__
        manifest['dependencias']['numpy'] = np.__version__
        manifest['dependencias']['statsmodels'] = sm.__version__
        import matplotlib
        manifest['dependencias']['matplotlib'] = matplotlib.__version__
        try:
            import sklearn
            manifest['dependencias']['scikit-learn'] = sklearn.__version__
        except:
            manifest['dependencias']['scikit-learn'] = 'não instalado'
    except Exception as e:
        logger.warning(f"Erro ao capturar versões: {e}")

    def save_artifact(name: str, content: str = None, df_: pd.DataFrame = None, fig_=None, json_obj=None):
        filepath = novos_dir / f"{prefix}__{name}"
        try:
            if df_ is not None:
                df_.to_csv(filepath, index=False, encoding='utf-8')
            elif fig_ is not None:
                fig_.tight_layout()
                fig_.savefig(filepath, dpi=300)
                plt.close(fig_)
            elif json_obj is not None:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(json_obj, f, indent=2, ensure_ascii=False)
            elif content is not None:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
            manifest['arquivos_gerados'].append(f"{prefix}__{name}")
            logger.info(f"✓ Salvo: {prefix}__{name}")
        except Exception as e:
            logger.error(f"✗ Erro ao salvar {name}: {e}")

    # ===== 4.1 SMD/IPW/Cobertura removido (foge da base) =====

    # ===== 4.2 ICC =====
    logger.info("=== 4.2: ICC ===")
    try:
        from statsmodels.regression.mixed_linear_model import MixedLM

        icc_results = {}

        # ICC linear para delta_vantagem
        if 'delta_vantagem' in df_work.columns and 'UF' in df_work.columns:
            try:
                df_clean = df_work[['delta_vantagem', 'UF']].dropna()
                if len(df_clean) > 10:
                    md_null = MixedLM(df_clean['delta_vantagem'],
                                     np.ones(len(df_clean)),
                                     groups=df_clean['UF'])
                    mdf_null = md_null.fit(reml=False)

                    var_between = float(mdf_null.cov_re.iloc[0, 0]) if hasattr(mdf_null.cov_re, 'iloc') else 0
                    var_within = float(mdf_null.scale)
                    icc_linear = var_between / (var_between + var_within) if (var_between + var_within) > 0 else 0

                    icc_results['delta_vantagem'] = {
                        'var_between': var_between,
                        'var_within': var_within,
                        'ICC': icc_linear,
                        'converged': mdf_null.converged
                    }
                    logger.info(f"ICC (delta_vantagem): {icc_linear:.4f}")
            except Exception as e:
                logger.warning(f"Erro ao calcular ICC linear: {e}")

        # ICC binário aproximado para reeleito
        if 'reeleito' in df_work.columns and 'UF' in df_work.columns:
            try:
                df_clean = df_work[['reeleito', 'UF']].dropna()
                if len(df_clean) > 10:
                    md_lpm = MixedLM(df_clean['reeleito'],
                                    np.ones(len(df_clean)),
                                    groups=df_clean['UF'])
                    mdf_lpm = md_lpm.fit(reml=False)

                    var_between = float(mdf_lpm.cov_re.iloc[0, 0]) if hasattr(mdf_lpm.cov_re, 'iloc') else 0
                    var_within_lpm = float(mdf_lpm.scale)
                    icc_lpm = var_between / (var_between + var_within_lpm) if (var_between + var_within_lpm) > 0 else 0

                    icc_results['reeleito_LPM_aprox'] = {
                        'ICC_LPM_aprox': icc_lpm,
                        'nota': 'Aproximação via LPM; ICC logístico ≈ ICC_LPM / (ICC_LPM + π²/3)'
                    }
                    logger.info(f"ICC LPM aprox (reeleito): {icc_lpm:.4f}")
            except Exception as e:
                logger.warning(f"Erro ao calcular ICC binário: {e}")

        if icc_results:
            save_artifact('ICC.json', json_obj=icc_results)
        else:
            logger.warning("Nenhum ICC calculado; variáveis ausentes")

    except ImportError:
        logger.warning("MixedLM não disponível; pulando ICC")
    except Exception as e:
        logger.error(f"Erro em ICC: {e}")

    # ===== 4.3 ΔVantagem: OLS vs Mixed =====
    logger.info("=== 4.3: ΔVantagem OLS vs Mixed ===")
    try:
        from statsmodels.regression.mixed_linear_model import MixedLM

        # Covariáveis para o modelo
        y_var = 'delta_vantagem'
        x_vars = []
        for v in ['gestao_index', 'NEC', 'ln_pop', 'ifdm_geral', 'abstencao_var', 'precocidade']:
            if v in df_work.columns:
                x_vars.append(v)

        if y_var in df_work.columns and x_vars and 'UF' in df_work.columns:
            df_model = df_work[[y_var, 'UF'] + x_vars].dropna()

            if len(df_model) > len(x_vars) + 5:
                y = df_model[y_var]
                X = sm.add_constant(df_model[x_vars])
                groups = df_model['UF']

                results_dv = {}

                # OLS com HC1
                try:
                    ols_mod = sm.OLS(y, X).fit(cov_type='HC1')
                    results_dv['OLS'] = {
                        'AIC': float(ols_mod.aic),
                        'BIC': float(ols_mod.bic),
                        'coef': {k: float(v) for k, v in ols_mod.params.items()},
                        'pvalues': {k: float(v) for k, v in ols_mod.pvalues.items()}
                    }
                    logger.info(f"OLS AIC={ols_mod.aic:.2f}, BIC={ols_mod.bic:.2f}")
                except Exception as e:
                    logger.warning(f"Erro OLS: {e}")

                # Mixed intercepto aleatório
                try:
                    mixed_int = MixedLM(y, X, groups=groups).fit(reml=False)
                    var_re = float(mixed_int.cov_re.iloc[0,0]) if hasattr(mixed_int.cov_re, 'iloc') else 0
                    results_dv['Mixed_intercept'] = {
                        'AIC': float(mixed_int.aic),
                        'BIC': float(mixed_int.bic),
                        'var_RE': var_re,
                        'converged': mixed_int.converged,
                        'coef': {k: float(v) for k, v in mixed_int.fe_params.items()},
                        'pvalues': {k: float(v) for k, v in mixed_int.pvalues.items()}
                    }
                    logger.info(f"Mixed(int) AIC={mixed_int.aic:.2f}, BIC={mixed_int.bic:.2f}, var_RE={var_re:.4f}")
                except Exception as e:
                    logger.warning(f"Erro Mixed intercept: {e}")

                # Mixed slope aleatório para NEC (se NEC existir)
                if 'NEC' in x_vars:
                    try:
                        mixed_slope = MixedLM(y, X, groups=groups,
                                             exog_re=df_model[['NEC']]).fit(reml=False)
                        var_re_slope = float(mixed_slope.cov_re.iloc[0,0]) if hasattr(mixed_slope.cov_re, 'iloc') else 0
                        results_dv['Mixed_slope_NEC'] = {
                            'AIC': float(mixed_slope.aic),
                            'BIC': float(mixed_slope.bic),
                            'var_RE_slope': var_re_slope,
                            'converged': mixed_slope.converged,
                            'coef': {k: float(v) for k, v in mixed_slope.fe_params.items()}
                        }
                        logger.info(f"Mixed(slope NEC) AIC={mixed_slope.aic:.2f}")
                    except Exception as e:
                        logger.warning(f"Erro Mixed slope: {e}")

                # Decisão automática
                if 'OLS' in results_dv and 'Mixed_intercept' in results_dv:
                    aic_ols = results_dv['OLS']['AIC']
                    aic_mixed = results_dv['Mixed_intercept']['AIC']
                    var_re = results_dv['Mixed_intercept'].get('var_RE', 0)

                    if aic_mixed < aic_ols and var_re > 0:
                        results_dv['decisao'] = 'Preferir Mixed_intercept (AIC menor e var_RE > 0)'
                    else:
                        results_dv['decisao'] = 'Manter OLS (AIC melhor ou var_RE ~ 0)'

                save_artifact('DeltaVantagem_modelos.json', json_obj=results_dv)

                # Forest plot
                try:
                    coefs_plot = results_dv.get('Mixed_intercept', results_dv.get('OLS', {})).get('coef', {})
                    if coefs_plot:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        vars_plot = [k for k in coefs_plot.keys() if k != 'const']
                        vals = [coefs_plot[k] for k in vars_plot]
                        ax.barh(vars_plot, vals, color='steelblue')
                        ax.axvline(0, color='red', linestyle='--', linewidth=1)
                        ax.set_xlabel('Coeficiente')
                        ax.set_title(f'ΔVantagem (N={len(df_model)})')
                        save_artifact('DeltaVantagem_forest.png', fig_=fig)
                except Exception as e:
                    logger.warning(f"Erro ao gerar forest plot ΔVantagem: {e}")
            else:
                logger.warning("Dados insuficientes para modelo ΔVantagem")
        else:
            logger.warning(f"Variáveis ausentes para ΔVantagem: y={y_var in df_work.columns}, X={x_vars}, UF={'UF' in df_work.columns}")
    except Exception as e:
        logger.error(f"Erro em ΔVantagem: {e}")

    # ===== 4.4 Reeleição GLM Binomial =====
    logger.info("=== 4.4: Reeleição GLM Binomial ===")
    try:
        y_var = 'reeleito'
        x_vars = []
        for v in ['gestao_index', 'NEC', 'ln_pop', 'ifdm_geral', 'abstencao_var', 'precocidade']:
            if v in df_work.columns:
                x_vars.append(v)

        if y_var in df_work.columns and x_vars and 'UF' in df_work.columns:
            df_model = df_work[[y_var, 'UF'] + x_vars].dropna()

            if len(df_model) > len(x_vars) + 5:
                y = df_model[y_var]
                X = sm.add_constant(df_model[x_vars])

                try:
                    from statsmodels.genmod.families import Binomial
                    from statsmodels.genmod.cov_struct import Exchangeable
                    from statsmodels.genmod.generalized_estimating_equations import GEE

                    # GLM com cluster por UF (via GEE)
                    gee_mod = GEE(y, X, groups=df_model['UF'],
                                 family=Binomial(), cov_struct=Exchangeable())
                    gee_fit = gee_mod.fit()

                    reeleicao_res = {
                        'N': len(df_model),
                        'coef': {k: float(v) for k, v in gee_fit.params.items()},
                        'pvalues': {k: float(v) for k, v in gee_fit.pvalues.items()},
                        'conf_int': {k: [float(gee_fit.conf_int().loc[k, 0]),
                                        float(gee_fit.conf_int().loc[k, 1])]
                                    for k in gee_fit.params.index}
                    }

                    save_artifact('Reeleicao_resumo.json', json_obj=reeleicao_res)
                    logger.info(f"GLM Binomial estimado (N={len(df_model)})")

                    # Forest plot
                    fig, ax = plt.subplots(figsize=(8, 6))
                    vars_plot = [k for k in gee_fit.params.index if k != 'const']
                    coefs = [gee_fit.params[k] for k in vars_plot]
                    cis_lower = [gee_fit.conf_int().loc[k, 0] for k in vars_plot]
                    cis_upper = [gee_fit.conf_int().loc[k, 1] for k in vars_plot]
                    errors = [[coefs[i] - cis_lower[i], cis_upper[i] - coefs[i]]
                             for i in range(len(vars_plot))]

                    ax.errorbar(coefs, range(len(vars_plot)), xerr=np.array(errors).T,
                               fmt='o', capsize=5, color='steelblue')
                    ax.set_yticks(range(len(vars_plot)))
                    ax.set_yticklabels(vars_plot)
                    ax.axvline(0, color='red', linestyle='--', linewidth=1)
                    ax.set_xlabel('Coeficiente (logit)')
                    ax.set_title(f'Reeleição (N={len(df_model)}, clusters UF)')
                    save_artifact('Reeleicao_forest.png', fig_=fig)

                    # Predicted probabilities vs NEC
                    if 'NEC' in x_vars:
                        try:
                            nec_quantiles = df_model['NEC'].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).values
                            X_pred_data = []
                            for nec_val in nec_quantiles:
                                x_pred = X.median().copy()
                                x_pred['const'] = 1
                                x_pred['NEC'] = nec_val
                                X_pred_data.append(x_pred.values)

                            X_pred = pd.DataFrame(X_pred_data, columns=X.columns)
                            probs = gee_fit.predict(X_pred)

                            df_pred = pd.DataFrame({
                                'NEC': nec_quantiles,
                                'prob_reeleicao': probs
                            })
                            save_artifact('Reeleicao_PredProbs_vs_NEC.csv', df_=df_pred)

                            # Curve plot
                            fig, ax = plt.subplots(figsize=(8, 5))
                            ax.plot(nec_quantiles, probs, 'o-', color='steelblue', linewidth=2)
                            ax.set_xlabel('NEC (percentis)')
                            ax.set_ylabel('Prob. Reeleição')
                            ax.set_title(f'Predição vs NEC (N={len(df_model)})')
                            ax.grid(True, alpha=0.3)
                            save_artifact('Reeleicao_predcurve_NEC.png', fig_=fig)

                        except Exception as e:
                            logger.warning(f"Erro ao gerar predições vs NEC: {e}")

                    # AMEs (simplificado, sem statsmodels.discrete.discrete_model.get_margeff)
                    logger.info("AMEs: não implementado (requer statsmodels.discrete ou margins manualmente)")

                except Exception as e:
                    logger.error(f"Erro ao estimar GLM Binomial: {e}")
            else:
                logger.warning("Dados insuficientes para modelo Reeleição")
        else:
            logger.warning(f"Variáveis ausentes para Reeleição")
    except Exception as e:
        logger.error(f"Erro em Reeleição: {e}")

    # ===== 4.5 Índice Névoa =====
    logger.info("=== 4.5: Índice Névoa ===")
    try:
        # Buscar variáveis de temporalidade/medidas no dataframe
        nevoa_vars = []

        # Variáveis derivadas que já temos
        if 'precocidade' in df_work.columns:
            nevoa_vars.append('precocidade')
        if 'gestao_index' in df_work.columns:
            nevoa_vars.append('gestao_index')

        # Buscar colunas normalizadas (dias da eleição)
        for col in df_work.columns:
            col_lower = col.lower()
            # Pegar colunas "Dias da eleição - X (Normalizado)"
            if 'normalizado' in col_lower and 'dias' in col_lower:
                if col not in nevoa_vars:
                    nevoa_vars.append(col)

        # Buscar outras colunas de temporalidade
        for col in df_work.columns:
            col_lower = col.lower()
            if any(x in col_lower for x in ['duracao', 'duração', 'intensidade']) and 'medida' in col_lower:
                if col not in nevoa_vars:
                    nevoa_vars.append(col)

        if len(nevoa_vars) >= 2:
            df_nevoa = df_work[nevoa_vars].dropna()

            if len(df_nevoa) > 10:
                try:
                    from sklearn.preprocessing import StandardScaler
                    from sklearn.decomposition import PCA

                    scaler = StandardScaler()
                    nevoa_z = scaler.fit_transform(df_nevoa)

                    # Média z como índice simples
                    nevoa_index_z = nevoa_z.mean(axis=1)

                    df_nevoa_out = pd.DataFrame({
                        'id': df_nevoa.index,
                        'nevoa_index_z': nevoa_index_z
                    })

                    # Adicionar componentes individuais
                    for i, v in enumerate(nevoa_vars):
                        df_nevoa_out[f'{v}_z'] = nevoa_z[:, i]

                    save_artifact('NevoaIndex.csv', df_=df_nevoa_out)
                    logger.info(f"✓ Índice Névoa criado com {len(nevoa_vars)} variáveis: {nevoa_vars}")

                    # PCA
                    try:
                        pca = PCA(n_components=min(len(nevoa_vars), 3))
                        pca.fit(nevoa_z)
                        explained = pca.explained_variance_ratio_

                        pca_txt = f"PCA Névoa:\n"
                        pca_txt += f"  Variáveis ({len(nevoa_vars)}): {', '.join(nevoa_vars)}\n"
                        for i, exp in enumerate(explained):
                            pca_txt += f"  PC{i+1}: {exp:.3f} ({exp*100:.1f}%)\n"
                        pca_txt += f"  Total explicado (PC1-{len(explained)}): {sum(explained):.3f} ({sum(explained)*100:.1f}%)\n"
                        save_artifact('Nevoa_PCA_explained.txt', content=pca_txt)
                        logger.info(f"✓ PCA Névoa: PC1 explica {explained[0]*100:.1f}%")
                    except Exception as e:
                        logger.warning(f"Erro PCA: {e}")
                except ImportError as e:
                    logger.warning(f"sklearn não disponível - Névoa não criado: {e}")
                except Exception as e:
                    logger.error(f"Erro ao criar Névoa: {e}")
            else:
                logger.warning("Dados insuficientes para Índice Névoa após dropna")
        else:
            logger.warning(f"Variáveis insuficientes para Névoa (encontradas {len(nevoa_vars)}): {nevoa_vars}")

    except Exception as e:
        logger.error(f"Erro em Névoa: {e}")

    # ===== 4.5b Janelas 30/60/90 =====
    logger.info("=== 4.5b: Janelas 30/60/90 ===")
    try:
        # Buscar colunas de data
        date_cols = []
        for col in df_work.columns:
            col_lower = col.lower()
            if any(x in col_lower for x in ['data', 'início', 'inicio', 'obrigatoriedade']):
                if 'máscara' in col_lower or 'mascara' in col_lower or 'comércio' in col_lower or 'comercio' in col_lower:
                    date_cols.append(col)

        # Precisamos de data da eleição (15/11/2020) e datas das medidas
        if len(date_cols) >= 1:
            eleicao_date = pd.Timestamp('2020-11-15')

            janelas = []
            for col in date_cols:
                if col in df_work.columns:
                    df_temp = df_work[[col]].copy()

                    # Tentar converter para datetime
                    try:
                        df_temp[col] = pd.to_datetime(df_temp[col], errors='coerce')

                        # Calcular dias antes da eleição
                        df_temp['dias_ate_eleicao'] = (eleicao_date - df_temp[col]).dt.days

                        # Janelas: <30, 30-60, 60-90, >90
                        df_temp['janela_30'] = (df_temp['dias_ate_eleicao'] < 30).astype(int)
                        df_temp['janela_60'] = ((df_temp['dias_ate_eleicao'] >= 30) & (df_temp['dias_ate_eleicao'] < 60)).astype(int)
                        df_temp['janela_90'] = ((df_temp['dias_ate_eleicao'] >= 60) & (df_temp['dias_ate_eleicao'] < 90)).astype(int)
                        df_temp['janela_90plus'] = (df_temp['dias_ate_eleicao'] >= 90).astype(int)

                        janelas.append({
                            'medida': col,
                            'n_30': int(df_temp['janela_30'].sum()),
                            'n_60': int(df_temp['janela_60'].sum()),
                            'n_90': int(df_temp['janela_90'].sum()),
                            'n_90plus': int(df_temp['janela_90plus'].sum())
                        })
                    except:
                        pass

            if janelas:
                janelas_df = pd.DataFrame(janelas)
                save_artifact('Janela_30_60_90.csv', df_=janelas_df)
                logger.info(f"✓ Janelas 30/60/90: {len(janelas)} medidas analisadas")
            else:
                logger.info("Janelas 30/60/90: colunas de data não conversíveis")
        else:
            logger.info("Janelas 30/60/90: sem colunas de data identificadas")

    except Exception as e:
        logger.error(f"Erro em Janelas: {e}")

# === FIM NOVAS ANALISES ===

if __name__ == '__main__':
    main()
