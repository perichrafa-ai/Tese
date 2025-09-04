# -*- coding: utf-8 -*-
"""
tese_master.py — Master consolidado (nomes fixos e robustez compatível)

Blocos:
  27A — Preparo: ΔAbstenção, ΔNEC_diff, ΔNEC_log, Gestao_Index (+ centragem)
  27B — Logit: Reeleição ~ Gestao_Index_c × ΔAbstenção_c (+ AMEs + figura)
  27C — OLS: ΔVantagem ~ Vantagem2016_c × ΔNEC_log_c (cluster por UF, se existir)
  27D — Sensibilidade: (log vs diff; Top-5 por UF), com robustez

Requisitos: numpy, pandas, statsmodels, matplotlib
CSV padrão: encoding='latin-1', sep=';', decimais com vírgula
Compatível Windows (pausa ao final).
"""
import os, re, sys, argparse, warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning)

# =====================================
# CONFIGURAÇÃO DE NOMES (sem detecção)
# =====================================
CONFIG_COLS = {
    # Insumos 27A (na base original)
    "ABST_2016": None,     # se quiser forçar, informe o nome exato aqui; caso None, usaremos padrões leves só no 27A
    "ABST_2020": None,
    "ELEIT_2016": None,
    "ELEIT_2020": None,
    "VANT_2016":  "Vantagem do Incumbente no primeiro turno 2016",
    "NEC_2016":   "Número efetivo de candidatos de 2016",
    "NEC_2020":   "Número efetivo de candidatos de 2020",
    "UF":         "UF",          # se sua base tiver "Estado" ao invés de "UF", 27A copia para UF
    "POP":        "População",   # se não existir, 27A tenta achar "População" e normaliza para 'Populacao'

    # Saídas 27A (na base preparada)
    "REELEITO":   "VD1_Reeleito",          # mantenha este nome após 27A (ou 'Reeleito' se for o seu)
    "DELTA_VANT": "VD2_DeltaVantagem",
    "DELTA_ABST_C": "delta_abstencao_pp_c",
    "GESTAO_C":     "Gestao_Index_c",
    "COMP_LOG_C":   "delta_competicao_log_c",
    "COMP_DIFF_C":  "delta_competicao_diff_c",
}

# =================================================================================
# Utilidades
# =================================================================================
def _safe_fname(s):
    return re.sub(r'[^a-zA-Z0-9_.-]+', '_', str(s))

def _outbase(base_csv_path):
    outdir = os.path.dirname(os.path.abspath(base_csv_path))
    base = os.path.splitext(os.path.basename(base_csv_path))[0]
    return outdir, base

def _print_paths(kind, csv_path, extra=None):
    if extra is None: extra = {}
    print(f"[{kind}] CSV: {csv_path}")
    for k, v in extra.items():
        print(f"[{kind}] {k}: {v}")

def _to_num(x):
    """Converte série/valor com vírgula decimal em float; remove separador de milhar '.'."""
    if isinstance(x, pd.Series):
        s = x.astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
        return pd.to_numeric(s, errors='coerce')
    if isinstance(x, str):
        x = x.replace('.', '').replace(',', '.')
    return pd.to_numeric(x, errors='coerce')

def _center(x):
    x = pd.to_numeric(x, errors='coerce')
    mu = float(np.nanmean(x))
    return x - mu, mu

def _bin_map(series):
    s = series.astype(str).str.strip().str.lower()
    out = pd.Series(index=series.index, dtype='float64')
    out.loc[s.isin(['sim','sí','si'])] = 1.0
    out.loc[s.isin(['não','nao','no'])] = 0.0
    rest = out.isna()
    out.loc[rest] = pd.to_numeric(series[rest], errors='coerce')
    return out

# =================================================================================
# 27A — Preparação (mantém padrões leves só aqui para achar colunas básicas se None)
# =================================================================================
def _find_col(df, patterns):
    for pat in patterns:
        for c in df.columns:
            if re.search(pat, str(c), flags=re.IGNORECASE):
                return c
    return None

def _find_nec_cols(df):
    pat16 = r'(NEC|efetiv[oa]|n[úu]mero efetivo).*2016|2016.*(NEC|efetiv[oa])|candidat[oa]s?.*efetiv[oa].*2016'
    pat20 = r'(NEC|efetiv[oa]|n[úu]mero efetivo).*2020|2020.*(NEC|efetiv[oa])|candidat[oa]s?.*efetiv[oa].*2020'
    c16 = CONFIG_COLS["NEC_2016"] if CONFIG_COLS["NEC_2016"] in df.columns else _find_col(df, [pat16])
    c20 = CONFIG_COLS["NEC_2020"] if CONFIG_COLS["NEC_2020"] in df.columns else _find_col(df, [pat20])
    return c16, c20

def _compute_gestao_index(df):
    """Índice de Gestão (0-1) baseado em componentes disponíveis, média simples."""
    pats_bin = {
        "hosp_camp": r'hospital.*campanh',
        "tenda": r'tenda.*triag',
        "testes": r'teste[s]?.*exist[êe]n',
        "leitos_amp": r'leitos.*ampl',
        "samu_pub": r'servi[cç]o.*atendimento.*emerg',
        "vig_epi": r'vigilan[cç]a.*epidemiol',
    }
    pats_norm = {
        "mask_norm": r'm[áa]scara.*\(normalizad',
        "rest_norm": r'restri[cç][aã]o.*com[ée]rcio.*\(normalizad',
    }
    comp_list = []
    for key, pat in pats_bin.items():
        col = _find_col(df, [pat])
        if col is not None:
            comp_list.append(_bin_map(df[col]).rename(key))
    for key, pat in pats_norm.items():
        col = _find_col(df, [pat])
        if col is not None:
            comp = _to_num(df[col]).clip(0.0, 1.0)
            comp_list.append(comp.rename(key))
    if not comp_list:
        return pd.Series(np.nan, index=df.index, name='Gestao_Index')
    comp_df = pd.concat(comp_list, axis=1)
    gest = comp_df.mean(axis=1, skipna=True)
    gest.name = 'Gestao_Index'
    return gest

def run_27A_preparacao(base_csv_path="Base_VALIDADA_E_PRONTA.csv", encoding="latin-1", sep=";"):
    df = pd.read_csv(base_csv_path, encoding=encoding, sep=sep)
    df = df.loc[:, ~df.columns.str.startswith('Unnamed:')].copy()
    df.columns = df.columns.str.strip()

    # Colunas base (usar CONFIG quando definido; senão, padrões leves)
    col_abs16 = CONFIG_COLS["ABST_2016"] if CONFIG_COLS["ABST_2016"] in (df.columns) else _find_col(df, [r'\bAbsten(ç|c)[aã]o\s*2016\b', r'\babsten.*2016'])
    col_abs20 = CONFIG_COLS["ABST_2020"] if CONFIG_COLS["ABST_2020"] in (df.columns) else _find_col(df, [r'\bAbsten(ç|c)[aã]o\s*2020\b', r'\babsten.*2020'])
    col_ele16 = CONFIG_COLS["ELEIT_2016"] if CONFIG_COLS["ELEIT_2016"] in (df.columns) else _find_col(df, [r'\bEleitor(es)?\s*2016\b', r'\beleitor.*2016'])
    col_ele20 = CONFIG_COLS["ELEIT_2020"] if CONFIG_COLS["ELEIT_2020"] in (df.columns) else _find_col(df, [r'\bEleitor(es)?\s*2020\b', r'\beleitor.*2020'])
    col_vant2016 = CONFIG_COLS["VANT_2016"] if CONFIG_COLS["VANT_2016"] in (df.columns) else _find_col(df, [r'\bVantagem.*2016\b', r'\bVantagem2016\b'])

    if any(c is None for c in [col_abs16,col_abs20,col_ele16,col_ele20,col_vant2016]):
        raise RuntimeError("27A: coluna obrigatória não encontrada (abstenção/eleitores/vantagem2016).")

    for c in [col_abs16,col_abs20,col_ele16,col_ele20,col_vant2016]:
        df[c] = _to_num(df[c])

    df['tx_abstencao_2016'] = df[col_abs16] / df[col_ele16]
    df['tx_abstencao_2020'] = df[col_abs20] / df[col_ele20]
    df['delta_abstencao_pp'] = (df['tx_abstencao_2020'] - df['tx_abstencao_2016']) * 100.0

    # NEC
    col_nec16, col_nec20 = _find_nec_cols(df)
    if (col_nec16 is None) or (col_nec20 is None):
        raise RuntimeError("27A: NEC_2016/NEC_2020 não encontradas.")
    nec16 = _to_num(df[col_nec16]).mask(lambda x: x<=0, np.nan)
    nec20 = _to_num(df[col_nec20]).mask(lambda x: x<=0, np.nan)
    df['NEC_2016_calc'] = nec16
    df['NEC_2020_calc'] = nec20
    df['delta_competicao_diff'] = nec20 - nec16
    with np.errstate(divide='ignore', invalid='ignore'):
        df['delta_competicao_log'] = np.log(nec20 / nec16)

    # Índice de Gestão
    df['Gestao_Index'] = _compute_gestao_index(df)
    df['Gestao_Index_c'], _ = _center(df['Gestao_Index'])

    # Centragem moderadoras e vantagem
    df['delta_abstencao_pp_c'], _ = _center(df['delta_abstencao_pp'])
    df['delta_competicao_diff_c'], _ = _center(df['delta_competicao_diff'])
    df['delta_competicao_log_c'], _ = _center(df['delta_competicao_log'])
    df['vantagem2016_c'], _ = _center(df[col_vant2016])

    # UF e Populacao
    if CONFIG_COLS["UF"] in df.columns:
        df['UF'] = df[CONFIG_COLS["UF"]]
    elif 'Estado' in df.columns:
        df['UF'] = df['Estado']
    if CONFIG_COLS["POP"] in df.columns:
        df['Populacao'] = _to_num(df[CONFIG_COLS["POP"]])
    else:
        col_pop = _find_col(df, [r'Popula(ç|c)[aã]o'])
        if col_pop: df['Populacao'] = _to_num(df[col_pop])

    # Reeleito/DeltaVantagem (garante nomes “oficiais” se existirem)
    if 'VD1_Reeleito' not in df.columns:
        col_r = _find_col(df, [r'\bVD1_Reeleito\b', r'\bReeleito\b'])
        if col_r: df['VD1_Reeleito'] = df[col_r]
    if 'VD2_DeltaVantagem' not in df.columns:
        col_dv = _find_col(df, [r'\bVD2_DeltaVantagem\b', r'delta.*vantagem', r'varia(ç|c)[aã]o.*vantagem'])
        if col_dv: df['VD2_DeltaVantagem'] = df[col_dv]

    # Máscara mínima
    need = ['delta_abstencao_pp', 'Gestao_Index', 'vantagem2016_c']
    mask = np.ones(len(df), dtype=bool)
    for c in need:
        mask &= df[c].notna().values
    df['mask_27A'] = mask

    outdir, base = _outbase(base_csv_path)
    out_csv = os.path.join(outdir, f"{base}__27A_prepared.csv")
    df.to_csv(out_csv, index=False, encoding=encoding, sep=sep)

    def _desc(name):
        s = pd.to_numeric(df[name], errors='coerce')
        return (name, int(s.notna().sum()),
                float(np.nanmean(s)), float(np.nanstd(s, ddof=0)),
                float(np.nanmin(s)), float(np.nanmax(s)))
    resumo_vars = ['tx_abstencao_2016','tx_abstencao_2020','delta_abstencao_pp',
                   'delta_competicao_diff','delta_competicao_log',
                   'Gestao_Index','vantagem2016_c']
    resumo_vars = [v for v in resumo_vars if v in df.columns]
    resumo = [_desc(v) for v in resumo_vars]
    log_path = os.path.join(outdir, f"relatorio_27A_{base}.txt")
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("27A — Proxies (ΔAbstenção; ΔNEC_diff; ΔNEC_log) + Gestao_Index (centrada)\n")
        f.write(f"Arquivo de entrada: {base_csv_path}\nLinhas: {len(df)}\n\n")
        f.write("Resumo (N, média, dp, min, máx):\n")
        for (name,N,mean,std,minv,maxv) in resumo:
            f.write(f" - {name}: N={N}, mean={mean:.6f}, std={std:.6f}, min={minv:.6f}, max={maxv:.6f}\n")
        f.write(f"\nmask_27A TRUE: {int(df['mask_27A'].sum())} de {len(df)}\n")
    _print_paths("27A", out_csv, {"Relatório": log_path})
    return out_csv, log_path

# =================================================================================
# Camada de robustez compatível + salvamento de tabelas
# =================================================================================
def _robust_like(res, cov_type="HC1", groups=None):
    """
    Tenta aplicar robustez. Se a versão do statsmodels não tiver o método,
    ou falhar, retorna o próprio 'res'. Sempre retorna algo “compatível”.
    """
    try:
        if cov_type == "cluster" and groups is not None:
            return res._results.get_robustcov_results(cov_type='cluster', groups=groups)
        elif cov_type.upper().startswith("HC"):
            return res._results.get_robustcov_results(cov_type=cov_type)
        else:
            return res
    except Exception:
        try:
            if cov_type == "cluster" and groups is not None:
                return res.get_robustcov_results(cov_type='cluster', groups=groups)
            elif cov_type.upper().startswith("HC"):
                return res.get_robustcov_results(cov_type=cov_type)
        except Exception:
            return res

def _extract_series(res_like, res_ref):
    # Garante pandas.Series com o mesmo index dos params "normais"
    idx = res_ref.params.index if hasattr(res_ref.params, 'index') else pd.Index(range(len(res_ref.params)))
    def to_series(x):
        x = np.asarray(x).reshape(-1)
        return pd.Series(x, index=idx)
    out = {
        "params":  to_series(getattr(res_like, 'params',  res_ref.params)),
        "bse":     to_series(getattr(res_like, 'bse',     res_ref.bse)),
        "tvalues": to_series(getattr(res_like, 'tvalues', res_ref.tvalues)),
        "pvalues": to_series(getattr(res_like, 'pvalues', res_ref.pvalues)),
    }
    return out

def _save_coef_table(res_like, res_ref, path_csv):
    ext = _extract_series(res_like, res_ref)
    rows = []
    for name in ext["params"].index:
        rows.append({
            "variavel": name,
            "coef": float(ext["params"].loc[name]),
            "se": float(ext["bse"].loc[name]),
            "t_z": float(ext["tvalues"].loc[name]),
            "pvalue": float(ext["pvalues"].loc[name]),
        })
    pd.DataFrame(rows).to_csv(path_csv, index=False, encoding="utf-8")

# =================================================================================
# 27B — Logit (Gestão × Δ Abstenção) — NOMES FIXOS
# =================================================================================
def run_27B_interacaoA(base_csv_path="Base_VALIDADA_E_PRONTA__27A_prepared.csv", encoding="latin-1", sep=";"):
    C = CONFIG_COLS
    df = pd.read_csv(base_csv_path, encoding=encoding, sep=sep)
    df = df.loc[:, ~df.columns.str.startswith('Unnamed:')].copy()
    df.columns = df.columns.str.strip()

    gestao_col = C["GESTAO_C"]
    reeleito_col = C["REELEITO"] if C["REELEITO"] in df.columns else "Reeleito"
    delta_abst_c = C["DELTA_ABST_C"]

    for need in [gestao_col, reeleito_col, delta_abst_c]:
        if need not in df.columns:
            raise RuntimeError(f"27B: coluna necessária ausente: {need}")

    dfm = df[[gestao_col, delta_abst_c, reeleito_col]].dropna().copy()
    dfm.rename(columns={reeleito_col: 'Reeleito'}, inplace=True)
    dfm['Reeleito'] = _to_num(dfm['Reeleito']).astype(int)

    formula = f"Reeleito ~ {gestao_col} * {delta_abst_c}"
    res = smf.logit(formula, data=dfm).fit(disp=0, maxiter=200)

    # Robustez compatível (se falhar, fica o próprio res)
    res_rb = _robust_like(res, cov_type="HC1")

    outdir, base = _outbase(base_csv_path)
    coef_csv = os.path.join(outdir, f"tabela_coef_logit_interacaoA_{base}.csv")
    _save_coef_table(res_rb, res, coef_csv)

    # AMEs: calcule a partir do modelo "não-robusto" (mais estável para margeff)
    me = res.get_margeff(at='overall', method='dydx')
    ame_df = me.summary_frame()
    ame_csv = os.path.join(outdir, f"tabela_AMEs_logit_interacaoA_{base}.csv")
    ame_df.to_csv(ame_csv, index=True, encoding='utf-8')

    # Figura — linhas de predição
    pred_df = _logit_predict_grid(dfm=dfm, model_res=res, gestao_col=gestao_col, delta_col=delta_abst_c, n=121)
    fig_png = os.path.join(outdir, f"fig_27B_predprob_interacaoA_{base}.png")
    _plot_lines(pred_df, x="delta", y="pr", hue="gestao_label",
                title="Reeleição: Gestão × Δ Abstenção (prob. predita)", path_png=fig_png)

    _print_paths("27B", coef_csv, {"AMEs": ame_csv, "Figura": fig_png})
    return coef_csv, ame_csv, fig_png

def _logit_predict_grid(dfm, model_res, gestao_col, delta_col, n=121):
    mu_g = float(np.nanmean(dfm[gestao_col]))
    sd_g = float(np.nanstd(dfm[gestao_col]))
    g_levels = [("Gestão -1dp", mu_g - sd_g), ("Gestão média", mu_g), ("Gestão +1dp", mu_g + sd_g)]
    x = np.linspace(float(np.nanmin(dfm[delta_col])), float(np.nanmax(dfm[delta_col])), n)

    design_cols = model_res.model.exog_names
    base_vals = {}
    for col in design_cols:
        if col == "Intercept": continue
        if col in dfm.columns:
            base_vals[col] = float(np.nanmean(dfm[col]))

    preds = []
    for label, gval in g_levels:
        for xv in x:
            row = base_vals.copy()
            row[gestao_col] = gval
            row[delta_col] = xv
            inter_name = f"{gestao_col}:{delta_col}"
            if inter_name in design_cols:
                row[inter_name] = gval * xv
            ex = [1.0]
            for col in design_cols:
                if col == "Intercept": continue
                ex.append(row.get(col, base_vals.get(col, 0.0)))
            lin = np.dot(np.array(ex), model_res.params.values)
            pr = 1.0/(1.0+np.exp(-lin))
            preds.append({"delta": xv, "gestao_label": label, "pr": pr})
    return pd.DataFrame(preds)

# =================================================================================
# 27C — OLS (Vantagem2016 × ΔNEC_log) — NOMES FIXOS
# =================================================================================
def run_27C_interacaoB(base_csv_path="Base_VALIDADA_E_PRONTA__27A_prepared.csv", encoding="latin-1", sep=";"):
    C = CONFIG_COLS
    df = pd.read_csv(base_csv_path, encoding=encoding, sep=sep)
    df = df.loc[:, ~df.columns.str.startswith('Unnamed:')].copy()
    df.columns = df.columns.str.strip()

    dv_col = C["DELTA_VANT"]
    comp_col = C["COMP_LOG_C"] if (C["COMP_LOG_C"] in df.columns and df[C["COMP_LOG_C"]].notna().any()) \
                               else C["COMP_DIFF_C"]
    vant_c = 'vantagem2016_c'
    for need in [dv_col, comp_col, vant_c]:
        if need not in df.columns:
            raise RuntimeError(f"27C: coluna necessária ausente: {need}")

    dff = df[[dv_col, vant_c, comp_col]].dropna().copy()
    dff.rename(columns={dv_col: 'DeltaVantagem'}, inplace=True)

    res = smf.ols(f"DeltaVantagem ~ {vant_c} * {comp_col}", data=dff).fit()

    # cluster por UF se existir; senão HC1; senão simples
    if CONFIG_COLS["UF"] in df.columns and df[CONFIG_COLS["UF"]].notna().any():
        res_rb = _robust_like(res, cov_type="cluster", groups=df.loc[dff.index, CONFIG_COLS["UF"]])
    else:
        res_rb = _robust_like(res, cov_type="HC1")

    outdir, base = _outbase(base_csv_path)
    coef_csv = os.path.join(outdir, f"tabela_coef_ols_interacaoB_{base}.csv")
    _save_coef_table(res_rb, res, coef_csv)

    # Figura — linhas de predição por níveis de vantagem
    mu_v = float(np.nanmean(dff[vant_c])); sd_v = float(np.nanstd(dff[vant_c]))
    v_levels = [("Vantagem -1dp", mu_v - sd_v), ("Vantagem média", mu_v), ("Vantagem +1dp", mu_v + sd_v)]
    x = np.linspace(float(np.nanmin(dff[comp_col])), float(np.nanmax(dff[comp_col])), 121)

    preds = []
    design_cols = res.model.exog_names
    base_vals = {col: float(np.nanmean(dff[col])) for col in dff.columns if col in design_cols}
    for label, vval in v_levels:
        for xv in x:
            row = base_vals.copy()
            row[vant_c] = vval
            row[comp_col] = xv
            inter_name = f"{vant_c}:{comp_col}"
            if inter_name in design_cols:
                row[inter_name] = vval * xv
            ex = [1.0]
            for col in design_cols:
                if col == "Intercept": continue
                ex.append(row.get(col, base_vals.get(col, 0.0)))
            yhat = np.dot(np.array(ex), res.params.values)
            preds.append({"comp": xv, "vant_label": label, "yhat": yhat})
    pred_df = pd.DataFrame(preds)
    fig_png = os.path.join(outdir, f"fig_27C_pred_interacaoB_{base}.png")
    _plot_lines(pred_df, x="comp", y="yhat", hue="vant_label",
                title=f"ΔVantagem ~ {vant_c} × {comp_col} (linha de predição)", path_png=fig_png)

    _print_paths("27C", coef_csv, {"Figura": fig_png})
    return coef_csv, fig_png

# =================================================================================
# 27D — Sensibilidade — NOMES FIXOS
# =================================================================================
def _top5_por_uf(df, uf_col, pop_col):
    if (uf_col not in df.columns) or (pop_col not in df.columns):
        return None
    dfx = df[[uf_col, pop_col]].copy()
    dfx[pop_col] = _to_num(dfx[pop_col])
    df['_rank_pop_uf'] = dfx.groupby(uf_col)[pop_col].rank(ascending=False, method='first')
    top5 = df[df['_rank_pop_uf'] <= 5].copy()
    top5.drop(columns=['_rank_pop_uf'], inplace=True)
    return top5

def run_27D_sensibilidade(base_csv_path="Base_VALIDADA_E_PRONTA__27A_prepared.csv", encoding="latin-1", sep=";"):
    C = CONFIG_COLS
    df = pd.read_csv(base_csv_path, encoding=encoding, sep=sep)
    df = df.loc[:, ~df.columns.str.startswith('Unnamed:')].copy()
    df.columns = df.columns.str.strip()

    dv_col = C["DELTA_VANT"]
    comp_log_c = C["COMP_LOG_C"] if C["COMP_LOG_C"] in df.columns else None
    comp_diff_c = C["COMP_DIFF_C"] if C["COMP_DIFF_C"] in df.columns else None
    if dv_col not in df.columns or ('vantagem2016_c' not in df.columns) or ((comp_log_c is None) and (comp_diff_c is None)):
        raise RuntimeError("27D: colunas necessárias ausentes; rode 27A.")

    outdir, base = _outbase(base_csv_path)
    rows = []

    def _fit_ols(dfin, label, comp_col, cov_type="HC1", groups=None):
        dfx = dfin[[dv_col, 'vantagem2016_c', comp_col]].dropna().copy()
        dfx.rename(columns={dv_col: 'DeltaVantagem'}, inplace=True)
        res = smf.ols(f"DeltaVantagem ~ vantagem2016_c * {comp_col}", data=dfx).fit()
        res_rb = _robust_like(res, cov_type=("cluster" if (cov_type=="cluster" and groups is not None) else "HC1"), groups=groups)
        # extrair interação de forma consistente
        inter = f"vantagem2016_c:{comp_col}"
        ext = _extract_series(res_rb, res)
        coef = ext["params"].reindex_like(ext["params"]).get(inter, np.nan)
        se   = ext["bse"].reindex_like(ext["bse"]).get(inter, np.nan)
        p    = ext["pvalues"].reindex_like(ext["pvalues"]).get(inter, np.nan)
        rows.append({"modelo": label, "comp_col": comp_col, "N": int(dfx.shape[0]),
                     "coef_interacao": float(coef) if pd.notna(coef) else np.nan,
                     "se": float(se) if pd.notna(se) else np.nan,
                     "pvalue": float(p) if pd.notna(p) else np.nan})

    # FULL
    if comp_log_c is not None:
        if CONFIG_COLS["UF"] in df.columns and df[CONFIG_COLS["UF"]].notna().any():
            _fit_ols(df, "FULL_log_clusterUF", comp_log_c, cov_type="cluster", groups=df[CONFIG_COLS["UF"]])
        _fit_ols(df, "FULL_log_HC1", comp_log_c, cov_type="HC1")
    if comp_diff_c is not None:
        if CONFIG_COLS["UF"] in df.columns and df[CONFIG_COLS["UF"]].notna().any():
            _fit_ols(df, "FULL_diff_clusterUF", comp_diff_c, cov_type="cluster", groups=df[CONFIG_COLS["UF"]])
        _fit_ols(df, "FULL_diff_HC1", comp_diff_c, cov_type="HC1")

    # TOP-5 por UF (se possível)
    top5 = _top5_por_uf(df.copy(), CONFIG_COLS["UF"], 'Populacao' if 'Populacao' in df.columns else CONFIG_COLS["POP"])
    if top5 is not None:
        if comp_log_c is not None:
            if CONFIG_COLS["UF"] in top5.columns and top5[CONFIG_COLS["UF"]].notna().any():
                _fit_ols(top5, "TOP5_log_clusterUF", comp_log_c, cov_type="cluster", groups=top5[CONFIG_COLS["UF"]])
            _fit_ols(top5, "TOP5_log_HC1", comp_log_c, cov_type="HC1")
        if comp_diff_c is not None:
            if CONFIG_COLS["UF"] in top5.columns and top5[CONFIG_COLS["UF"]].notna().any():
                _fit_ols(top5, "TOP5_diff_clusterUF", comp_diff_c, cov_type="cluster", groups=top5[CONFIG_COLS["UF"]])
            _fit_ols(top5, "TOP5_diff_HC1", comp_diff_c, cov_type="HC1")

    out_csv = os.path.join(outdir, f"tabela_27D_sensibilidade_{base}.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding='utf-8')
    _print_paths("27D", out_csv)
    return out_csv

# =================================================================================
# Plot utilitário
# =================================================================================
def _plot_lines(df, x, y, hue, title, path_png):
    plt.figure(figsize=(7.2, 4.2), dpi=150)
    labels = sorted(df[hue].unique())
    for lab in labels:
        sub = df[df[hue]==lab].sort_values(x)
        plt.plot(sub[x].values, sub[y].values, label=lab, linewidth=2.0)
    plt.title(title)
    plt.xlabel(x); plt.ylabel(y)
    plt.legend(loc="best", frameon=False)
    plt.tight_layout()
    plt.savefig(path_png, bbox_inches="tight")
    plt.close()

# =================================================================================
# CLI
# =================================================================================
def _pause():
    try:
        input("\n----------------------------------------------------------\nExecução finalizada. Pressione Enter para sair...")
    except Exception:
        pass

def main():
    parser = argparse.ArgumentParser(description="tese_master — Execução 27A–27D com nomes fixos")
    parser.add_argument("--base", type=str, default="Base_VALIDADA_E_PRONTA.csv", help="CSV de entrada")
    parser.add_argument("--encoding", type=str, default="latin-1", help="Encoding (padrão latin-1)")
    parser.add_argument("--sep", type=str, default=";", help="Separador do CSV (padrão ;)")

    parser.add_argument("--run-27A", action="store_true", help="Rodar 27A (preparo)")
    parser.add_argument("--run-27B", action="store_true", help="Rodar 27B (Logit interacaoA)")
    parser.add_argument("--run-27C", action="store_true", help="Rodar 27C (OLS interacaoB)")
    parser.add_argument("--run-27D", action="store_true", help="Rodar 27D (sensibilidade)")

    args = parser.parse_args()
    try:
        if args.run_27A:
            run_27A_preparacao(base_csv_path=args.base, encoding=args.encoding, sep=args.sep)
        if args.run_27B:
            run_27B_interacaoA(base_csv_path=args.base, encoding=args.encoding, sep=args.sep)
        if args.run_27C:
            run_27C_interacaoB(base_csv_path=args.base, encoding=args.encoding, sep=args.sep)
        if args.run_27D:
            run_27D_sensibilidade(base_csv_path=args.base, encoding=args.encoding, sep=args.sep)
    finally:
        _pause()

if __name__ == "__main__":
    main()

