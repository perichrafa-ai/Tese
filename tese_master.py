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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning)

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
    return x - mu

def _find_col(df, candidates_regex):
    for pat in candidates_regex:
        cols = [c for c in df.columns if re.search(pat, c, flags=re.I)]
        if cols:
            return cols[0]
    return None

def _pause():
    if os.name == "nt":
        try:
            os.system("pause")
        except Exception:
            pass

# =================================================================================
# Extração “compatível” de séries do statsmodels
# =================================================================================
def _extract_series(res_like, res_ref=None):
    """Extrai params/bse/t/pvalues de forma resiliente (HC/cluster quando disponível)."""
    try:
        params = pd.Series(res_like.params, index=res_like.params.index)
        bse = pd.Series(res_like.bse, index=res_like.bse.index)
        tvalues = pd.Series(res_like.tvalues, index=res_like.tvalues.index)
        pvalues = pd.Series(res_like.pvalues, index=res_like.pvalues.index)
        return {"params": params, "bse": bse, "tvalues": tvalues, "pvalues": pvalues}
    except Exception:
        if res_ref is None:
            raise
        return {
            "params": pd.Series(res_ref.params, index=res_ref.params.index),
            "bse": pd.Series(res_ref.bse, index=res_ref.bse.index),
            "tvalues": pd.Series(res_ref.tvalues, index=res_ref.tvalues.index),
            "pvalues": pd.Series(res_ref.pvalues, index=res_ref.pvalues.index)
        }

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
# ROBUSTEZ: cluster/HC compatível
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

# =================================================================================
# Predições para 27B/27C
# =================================================================================
def _logit_predict_grid(dfm, model_res, gestao_col, delta_col, outdir, base):
    g_levels = [("Gestão baixa", float(np.nanpercentile(dfm[gestao_col], 25))),
                ("Gestão média", float(np.nanpercentile(dfm[gestao_col], 50))),
                ("Gestão alta",  float(np.nanpercentile(dfm[gestao_col], 75)))]
    x = np.linspace(float(np.nanmin(dfm[delta_col])), float(np.nanmax(dfm[delta_col])), 121)

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
            pr = sm.Logit(dfm["Reeleito"], sm.add_constant(pd.DataFrame([row])[design_cols])).fit(disp=0, maxiter=100).predict()[0]
            preds.append({"ΔAbstenção_c": xv, "Prob.Reeleição": pr, "Gestão": label})

    pred_df = pd.DataFrame(preds)
    fig_path = os.path.join(outdir, f"fig_27B_predprob_interacaoA_{base}.png")
    _plot_lines(pred_df, x="ΔAbstenção_c", y="Prob.Reeleição", hue="Gestão",
                title="Predição de Probabilidade — Logit (Gestão×ΔAbstenção)", path_png=fig_path)
    return pred_df, fig_path

# =================================================================================
# 27A — Preparo
# =================================================================================
def run_27A_preparacao(base_csv_path="Base_VALIDADA_E_PRONTA.csv", encoding="latin-1", sep=";"):
    df = pd.read_csv(base_csv_path, encoding=encoding, sep=sep)
    df = df.loc[:, ~df.columns.str.startswith('Unnamed:')].copy()
    df.columns = df.columns.str.strip()

    # Colunas base (usar CONFIG quando definido; senão, padrões leves)
    col_abs16 = CONFIG_COLS["ABST_2016"] if CONFIG_COLS["ABST_2016"] else _find_col(df, [r'\bAbsten(ç|c)[aã]o\s*2016\b', r'\babsten.*2016'])
    col_abs20 = CONFIG_COLS["ABST_2020"] if CONFIG_COLS["ABST_2020"] else _find_col(df, [r'\bAbsten(ç|c)[aã]o\s*2020\b', r'\babsten.*2020'])
    col_ele16 = CONFIG_COLS["ELEIT_2016"] if CONFIG_COLS["ELEIT_2016"] else _find_col(df, [r'\bEleitor(es)?\s*2016\b', r'\beleitor.*2016'])
    col_ele20 = CONFIG_COLS["ELEIT_2020"] if CONFIG_COLS["ELEIT_2020"] else _find_col(df, [r'\bEleitor(es)?\s*2020\b', r'\beleitor.*2020'])
    col_vant2016 = CONFIG_COLS["VANT_2016"] if CONFIG_COLS["VANT_2016"] else _find_col(df, [r'\bVantagem.*2016\b', r'\bVantagem2016\b'])

    if any(c is None for c in [col_abs16,col_abs20,col_ele16,col_ele20,col_vant2016]):
        raise RuntimeError("27A: coluna obrigatória não encontrada (abstenção/eleitores/vantagem2016).")

    # Limpeza numérica
    for c in [col_abs16,col_abs20,col_ele16,col_ele20,col_vant2016]:
        df[c] = _to_num(df[c])

    # Δ Abstenção em p.p. e centragem
    taxa_abs16 = df[col_abs16] / df[col_ele16] * 100.0
    taxa_abs20 = df[col_abs20] / df[col_ele20] * 100.0
    df["delta_abstencao_pp"] = taxa_abs20 - taxa_abs16
    df["delta_abstencao_pp_c"] = _center(df["delta_abstencao_pp"])

    # Δ competição (NEC): log e diff, + centragem
    nec16 = _to_num(df[CONFIG_COLS["NEC_2016"]]) if CONFIG_COLS["NEC_2016"] in df.columns else None
    nec20 = _to_num(df[CONFIG_COLS["NEC_2020"]]) if CONFIG_COLS["NEC_2020"] in df.columns else None
    if nec16 is not None and nec20 is not None:
        df["delta_competicao_diff"] = nec20 - nec16
        with np.errstate(divide='ignore', invalid='ignore'):
            df["delta_competicao_log"] = np.log(nec20.replace(0, np.nan)) - np.log(nec16.replace(0, np.nan))
        df["delta_competicao_diff_c"] = _center(df["delta_competicao_diff"])
        df["delta_competicao_log_c"]  = _center(df["delta_competicao_log"])

    # Gestão index (placeholder: se já existir, usa; senão, z-score de colunas 'Gestao_*' se houver)
    gest_cols = [c for c in df.columns if re.search(r'^Gest[aã]o', c, flags=re.I)]
    if "Gestao_Index" in df.columns:
        df["Gestao_Index_c"] = _center(_to_num(df["Gestao_Index"]))
    elif gest_cols:
        z = (_to_num(df[gest_cols]).apply(pd.to_numeric, errors='coerce') - _to_num(df[gest_cols]).mean()) / _to_num(df[gest_cols]).std(ddof=0)
        df["Gestao_Index_c"] = _center(z.mean(axis=1))
    else:
        df["Gestao_Index_c"] = np.nan  # se não existir, fica vazio

    # VD1/VD2 (placeholder: se já existir, respeita)
    if CONFIG_COLS["REELEITO"] not in df.columns:
        df["VD1_Reeleito"] = np.nan
    if CONFIG_COLS["DELTA_VANT"] not in df.columns:
        df["VD2_DeltaVantagem"] = np.nan

    # UF/Pop (normalização leve)
    if CONFIG_COLS["UF"] not in df.columns:
        if "Estado" in df.columns: df["UF"] = df["Estado"]
    if "Populacao" not in df.columns:
        if CONFIG_COLS["POP"] in df.columns:
            df["Populacao"] = _to_num(df[CONFIG_COLS["POP"]])

    outdir, base = _outbase(base_csv_path)
    out_csv = os.path.join(outdir, f"{base}__27A_prepared.csv")
    df.to_csv(out_csv, index=False, sep=";", encoding="utf-8")
    _print_paths("27A", out_csv, extra={"N": df.shape[0]})
    return out_csv

# =================================================================================
# 27B — Logit (Gestão × Δ Abstenção) — NOMES FIXOS
# =================================================================================
def run_27B_interacaoA(base_csv_path="Base_VALIDADA_E_PRONTA__27A_prepared.csv", encoding="latin-1", sep=";"):
    C = CONFIG_COLS
    df = pd.read_csv(base_csv_path, encoding=encoding, sep=sep)
    df = df.loc[:, ~df.columns.str.startswith('Unnamed:')].copy()
    df.columns = df.columns.str.strip()

    # Colunas
    reeleito_col = C["REELEITO"] if C["REELEITO"] in df.columns else _find_col(df, [r'\bReeleit'])
    gestao_col = C["GESTAO_C"] if C["GESTAO_C"] in df.columns else "Gestao_Index_c"
    delta_abst_c = C["DELTA_ABST_C"] if C["DELTA_ABST_C"] in df.columns else "delta_abstencao_pp_c"
    if any(c not in df.columns for c in [reeleito_col, gestao_col, delta_abst_c]):
        raise RuntimeError("27B: colunas necessárias ausentes; rode 27A.")

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
    pred_df, fig_path = _logit_predict_grid(dfm=dfm, model_res=res, gestao_col=gestao_col,
                                            delta_col=delta_abst_c, outdir=outdir, base=base)
    _print_paths("27B", coef_csv, extra={"AMEs": ame_csv, "FIG": fig_path})
    return coef_csv

# =================================================================================
# 27C — OLS (ΔVantagem ~ Vantagem2016 × ΔCompetição) — NOMES FIXOS
# =================================================================================
def run_27C_interacaoB(base_csv_path="Base_VALIDADA_E_PRONTA__27A_prepared.csv", encoding="latin-1", sep=";"):
    C = CONFIG_COLS
    df = pd.read_csv(base_csv_path, encoding=encoding, sep=sep)
    df = df.loc[:, ~df.columns.str.startswith('Unnamed:')].copy()
    df.columns = df.columns.str.strip()

    dv_col = C["DELTA_VANT"]
    comp_log_c = C["COMP_LOG_C"] if C["COMP_LOG_C"] in df.columns else None
    comp_diff_c = C["COMP_DIFF_C"] if C["COMP_DIFF_C"] in df.columns else None
    if dv_col not in df.columns or ('vantagem2016_c' not in df.columns) or ((comp_log_c is None) and (comp_diff_c is None)):
        raise RuntimeError("27C: colunas necessárias ausentes; rode 27A.")

    # Escolha preferencial: log; se não houver, diff
    comp_col = comp_log_c if comp_log_c is not None else comp_diff_c

    dff = df[[dv_col, 'vantagem2016_c', comp_col, CONFIG_COLS["UF"]] if CONFIG_COLS["UF"] in df.columns else [dv_col, 'vantagem2016_c', comp_col]].dropna().copy()
    dff.rename(columns={dv_col: 'DeltaVantagem'}, inplace=True)

    res = smf.ols(f"DeltaVantagem ~ vantagem2016_c * {comp_col}", data=dff).fit()

    # cluster por UF se existir; senão HC1; senão simples
    if CONFIG_COLS["UF"] in df.columns and df[CONFIG_COLS["UF"]].notna().any():
        res_rb = _robust_like(res, cov_type="cluster", groups=df.loc[dff.index, CONFIG_COLS["UF"]])
    else:
        res_rb = _robust_like(res, cov_type="HC1")

    outdir, base = _outbase(base_csv_path)
    coef_csv = os.path.join(outdir, f"tabela_coef_ols_interacaoB_{base}.csv")
    _save_coef_table(res_rb, res, coef_csv)

    # Figura — linhas de predição por níveis de vantagem
    mu_v = float(np.nanmean(dff['vantagem2016_c'])); sd_v = float(np.nanstd(dff['vantagem2016_c']))
    v_levels = [("Vantagem -1dp", mu_v - sd_v), ("Vantagem média", mu_v), ("Vantagem +1dp", mu_v + sd_v)]
    x = np.linspace(float(np.nanmin(dff[comp_col])), float(np.nanmax(dff[comp_col])), 121)

    preds = []
    design_cols = res.model.exog_names
    base_vals = {col: float(np.nanmean(dff[col])) for col in dff.columns if col in design_cols}
    for label, vval in v_levels:
        for xv in x:
            row = base_vals.copy()
            row['vantagem2016_c'] = vval
            row[comp_col] = xv
            inter_name = f"vantagem2016_c:{comp_col}"
            if inter_name in design_cols:
                row[inter_name] = vval * xv
            yhat = sm.OLS(dff['DeltaVantagem'], sm.add_constant(pd.DataFrame([row])[design_cols])).fit().predict()[0]
            preds.append({"ΔCompetição_c": xv, "ΔVantagem (pred.)": yhat, "Vantagem2016": label})

    pred_df = pd.DataFrame(preds)
    fig_path = os.path.join(outdir, f"fig_27C_pred_interacaoB_{base}.png")
    _plot_lines(pred_df, x="ΔCompetição_c", y="ΔVantagem (pred.)", hue="Vantagem2016",
                title="Predição — OLS (Vantagem2016×ΔCompetição)", path_png=fig_path)

    _print_paths("27C", coef_csv, extra={"FIG": fig_path, "comp_col": comp_col})
    return coef_csv

# =================================================================================
# 27D — Sensibilidade (log/diff; Top-5; HC1/cluster) — NOMES FIXOS
# =================================================================================
def _top5_por_uf(df, uf_col, pop_col="Populacao", k=5):
    if uf_col not in df.columns:
        return df
    df2 = df.copy()
    if pop_col not in df2.columns and "População" in df2.columns:
        df2["Populacao"] = _to_num(df2["População"])
    if pop_col not in df2.columns:
        return df2
    df2["rank_pop_uf"] = df2.groupby(uf_col)[pop_col].rank(ascending=False, method="first")
    top5 = df2[df2["rank_pop_uf"] <= k].copy()
    top5.drop(columns=['rank_pop_uf'], inplace=True)
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
        dff = dfin[[dv_col, 'vantagem2016_c', comp_col] + ([C["UF"]] if C["UF"] in dfin.columns else [])].dropna().copy()
        dff.rename(columns={dv_col: 'DeltaVantagem'}, inplace=True)
        res = smf.ols(f"DeltaVantagem ~ vantagem2016_c * {comp_col}", data=dff).fit()
        res_rb = _robust_like(res, cov_type=cov_type, groups=groups)
        ext = _extract_series(res_rb, res)
        inter = f"vantagem2016_c:{comp_col}"
        coef = ext["params"].reindex_like(ext["params"]).get(inter, np.nan)
        se   = ext["bse"].reindex_like(ext["bse"]).get(inter, np.nan)
        p    = ext["pvalues"].reindex_like(ext["pvalues"]).get(inter, np.nan)
        rows.append({"modelo": label, "comp_col": comp_col, "N": int(dff.shape[0]),
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

    # TOP5 por UF
    top5 = _top5_por_uf(df, uf_col=CONFIG_COLS["UF"])
    if top5.shape[0] >= 50:
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
    plt.savefig(path_png, dpi=150)
    plt.close()

# =================================================================================
# 32 — Quadro 4.1 (síntese final 27B–27D)
# =================================================================================
def _read_csv_safe(path):
    try:
        return pd.read_csv(path, encoding="utf-8")
    except FileNotFoundError:
        try:
            return pd.read_csv(path, encoding="latin-1", sep=";")
        except Exception:
            return None
    except Exception:
        return None

def _find_var_row(df, must_include):
    """
    Encontra a linha cujo campo 'variavel' contém todos os substrings de must_include (case-insensitive).
    Retorna (coef, pvalue, name) ou (np.nan, np.nan, None) se não achar.
    """
    if df is None or "variavel" not in df.columns:
        return np.nan, np.nan, None
    var_series = df["variavel"].astype(str).str.lower()
    mask = np.ones(len(var_series), dtype=bool)
    for sub in must_include:
        mask &= var_series.str.contains(str(sub).lower(), na=False)
    idxs = np.where(mask)[0]
    if len(idxs) == 0:
        # tente com ':' normalizado (alguns SM usam 'x' em vez de ':')
        var_series2 = var_series.str.replace('*', ':', regex=False).str.replace(' x ', ':', regex=False)
        mask = np.ones(len(var_series2), dtype=bool)
        for sub in must_include:
            mask &= var_series2.str.contains(str(sub).lower(), na=False)
        idxs = np.where(mask)[0]
    if len(idxs) == 0:
        return np.nan, np.nan, None
    i = int(idxs[0])
    row = df.iloc[i]
    coef = pd.to_numeric(row.get("coef", np.nan), errors="coerce")
    pval = pd.to_numeric(row.get("pvalue", np.nan), errors="coerce")
    return coef, pval, str(row.get("variavel"))

def _classif_sig(p):
    try:
        p = float(p)
    except Exception:
        return "n.s."
    if np.isnan(p):
        return "n.s."
    if p < 0.01:
        return "p<0,01"
    if p < 0.05:
        return "p<0,05"
    if p < 0.10:
        return "p<0,10"
    return "n.s."

def _sign_symbol(x, eps=1e-12):
    if pd.isna(x):
        return "0"
    if x > eps:
        return "+"
    if x < -eps:
        return "–"
    return "0"

def _robust_summary_27D(path_27d):
    """
    Lê a tabela 27D e produz um sumário compacto de robustez para a interação B.
    Retorna dicionário com flags (HC1, clusterUF, TOP5, log, diff) baseados em p<0.10
    e consistência de sinal.
    """
    out = {"HC1": None, "clusterUF": None, "TOP5": None, "log": None, "diff": None}
    df = _read_csv_safe(path_27d)
    if df is None or df.empty:
        return out

    def _ok(subdf):
        # significante a 10% e sinal consistente dentro do subconjunto
        if subdf.empty:
            return None
        sig = subdf["pvalue"].astype(float) < 0.10
        if not sig.any():
            return False
        signs = np.sign(subdf["coef_interacao"].astype(float).values)
        # pelo menos maioria absoluta com mesmo sinal
        return True if (np.sum(signs > 0) == len(signs) or np.sum(signs < 0) == len(signs)) else False

    # Flags por cov_type
    if "modelo" in df.columns:
        df["modelo"] = df["modelo"].astype(str)
        out["HC1"] = _ok(df[df["modelo"].str.contains("_HC1", case=False, na=False)])
        out["clusterUF"] = _ok(df[df["modelo"].str.contains("cluster", case=False, na=False)])
        out["TOP5"] = _ok(df[df["modelo"].str.contains("^TOP5", case=False, na=False)])
        out["log"] = _ok(df[df["comp_col"].astype(str).str.contains("log", case=False, na=False)]) if "comp_col" in df.columns else None
        out["diff"] = _ok(df[df["comp_col"].astype(str).str.contains("diff", case=False, na=False)]) if "comp_col" in df.columns else None
    return out

def run_32_quadro41(base_csv_path="Base_VALIDADA_E_PRONTA__27A_prepared.csv", encoding="latin-1", sep=";"):
    """
    Constrói o Quadro 4.1 agregando:
      - 27B: coef da interação Gestão×ΔAbstenção + efeitos principais (opcional)
      - 27C: coef da interação Vantagem2016×ΔCompetição + efeitos principais
      - 27D: sumário de robustez (HC1; clusterUF; TOP5; log/diff)
    Saída: quadro_4_1_sintese.csv (sep=';', decimal=',')
    """
    outdir, base = _outbase(base_csv_path)

    # Caminhos esperados
    p27b_coef = os.path.join(outdir, f"tabela_coef_logit_interacaoA_{base}.csv")
    p27b_ame  = os.path.join(outdir, f"tabela_AMEs_logit_interacaoA_{base}.csv")
    p27c_coef = os.path.join(outdir, f"tabela_coef_ols_interacaoB_{base}.csv")
    p27d_rob  = os.path.join(outdir, f"tabela_27D_sensibilidade_{base}.csv")

    df27b = _read_csv_safe(p27b_coef)
    df27c = _read_csv_safe(p27c_coef)

    # 27D — robustez (apenas para Interação B)
    rb = _robust_summary_27D(p27d_rob)
    def _rb_str(d):
        bits = []
        for k in ["HC1", "clusterUF", "TOP5", "log", "diff"]:
            v = d.get(k)
            if v is True: bits.append(f"{k} ✓")
            elif v is False: bits.append(f"{k} ×")
        return "; ".join(bits) if bits else "—"

    rows = []

    # =========================
    # 27B — Logit (Gestão × Δ Abstenção)
    # =========================
    if df27b is not None:
        coef_int, p_int, name_int = _find_var_row(df27b, ["gest", "abst", ":"])
        coef_g, p_g, name_g = _find_var_row(df27b, ["gest"])   # efeito principal (melhor esforço)
        coef_a, p_a, name_a = _find_var_row(df27b, ["abst"])

        # Interação A
        rows.append({
            "Mecanismo/H": "H5 — Névoa (moderação)",
            "Variável/termo": "Gestão × Δ Abstenção",
            "Direção esperada": "–",   # ajuste se necessário no texto
            "Sinal encontrado": _sign_symbol(coef_int),
            "Significância": _classif_sig(p_int),
            "Robustez": "HC1",  # logit com HC1 (27B)
            "Leitura em 1 linha": "Efeito da gestão varia com a abstenção; sinal conforme coeficiente.",
            "Arquivo-base": os.path.basename(p27b_coef),
        })
        # Gestão (efeito principal)
        if not (pd.isna(coef_g) and pd.isna(p_g)):
            rows.append({
                "Mecanismo/H": "—",
                "Variável/termo": "Gestão (efeito principal)",
                "Direção esperada": "—",
                "Sinal encontrado": _sign_symbol(coef_g),
                "Significância": _classif_sig(p_g),
                "Robustez": "HC1",
                "Leitura em 1 linha": "Efeito direto da gestão sobre a probabilidade de reeleição.",
                "Arquivo-base": os.path.basename(p27b_coef),
            })
        # Δ Abstenção (efeito principal)
        if not (pd.isna(coef_a) and pd.isna(p_a)):
            rows.append({
                "Mecanismo/H": "—",
                "Variável/termo": "Δ Abstenção (efeito principal)",
                "Direção esperada": "—",
                "Sinal encontrado": _sign_symbol(coef_a),
                "Significância": _classif_sig(p_a),
                "Robustez": "HC1",
                "Leitura em 1 linha": "Abstenção como proxy de névoa e seu efeito direto em reeleição.",
                "Arquivo-base": os.path.basename(p27b_coef),
            })

    # =========================
    # 27C — OLS (ΔVantagem ~ Vantagem2016 × ΔCompetição)
    # =========================
    if df27c is not None:
        coef_int, p_int, name_int = _find_var_row(df27c, ["vant", "comp", ":"])
        coef_v, p_v, name_v = _find_var_row(df27c, ["vantagem2016"])
        coef_c, p_c, name_c = _find_var_row(df27c, ["comp"])

        # Interação B
        rows.append({
            "Mecanismo/H": "—",
            "Variável/termo": "Vantagem2016 × Δ Competição",
            "Direção esperada": "–",
            "Sinal encontrado": _sign_symbol(coef_int),
            "Significância": _classif_sig(p_int),
            "Robustez": _rb_str(rb),
            "Leitura em 1 linha": "Competição altera o efeito da vantagem prévia sobre ΔVantagem.",
            "Arquivo-base": os.path.basename(p27c_coef) + ("; " + os.path.basename(p27d_rob) if os.path.exists(p27d_rob) else ""),
        })
        # Vantagem2016 (efeito principal)
        if not (pd.isna(coef_v) and pd.isna(p_v)):
            rows.append({
                "Mecanismo/H": "—",
                "Variável/termo": "Vantagem2016 (efeito principal)",
                "Direção esperada": "–",  # regressão à média
                "Sinal encontrado": _sign_symbol(coef_v),
                "Significância": _classif_sig(p_v),
                "Robustez": "clusterUF/HC1",
                "Leitura em 1 linha": "Regressão à média: maior vantagem prévia tende a reduzir ΔVantagem.",
                "Arquivo-base": os.path.basename(p27c_coef),
            })
        # Δ Competição (efeito principal)
        if not (pd.isna(coef_c) and pd.isna(p_c)):
            rows.append({
                "Mecanismo/H": "—",
                "Variável/termo": "Δ Competição (efeito principal)",
                "Direção esperada": "–",
                "Sinal encontrado": _sign_symbol(coef_c),
                "Significância": _classif_sig(p_c),
                "Robustez": "clusterUF/HC1",
                "Leitura em 1 linha": "Aumento de competição tende a reduzir a vantagem do incumbente.",
                "Arquivo-base": os.path.basename(p27c_coef),
            })

    # Construção do DataFrame final
    quadro = pd.DataFrame(rows, columns=[
        "Mecanismo/H", "Variável/termo", "Direção esperada",
        "Sinal encontrado", "Significância", "Robustez",
        "Leitura em 1 linha", "Arquivo-base"
    ])

    out_csv = os.path.join(outdir, "quadro_4_1_sintese.csv")
    # Garantir separador ';' e vírgula decimal para quaisquer números (aqui, quase tudo é texto)
    quadro.to_csv(out_csv, index=False, encoding="utf-8", sep=";")
    _print_paths("32", out_csv, extra={"linhas": quadro.shape[0]})
    return out_csv

# =================================================================================
# MAIN
# =================================================================================
def main():
    parser = argparse.ArgumentParser(description="tese_master — Execução 27A–27D com nomes fixos")
    parser.add_argument("--base", type=str, default="Base_VALIDADA_E_PRONTA.csv", help="CSV de entrada")
    parser.add_argument("--encoding", type=str, default="latin-1", help="Encoding (padrão latin-1)")
    parser.add_argument("--sep", type=str, default=";", help="Separador do CSV (padrão ;)")

    parser.add_argument("--run-27A", action="store_true", help="Rodar 27A (preparo)")
    parser.add_argument("--run-27B", action="store_true", help="Rodar 27B (Logit interacaoA)")
    parser.add_argument("--run-27C", action="store_true", help="Rodar 27C (OLS interacaoB)")
    parser.add_argument("--run-27D", action="store_true", help="Rodar 27D (sensibilidade)")
    parser.add_argument("--run-32", action="store_true", help="Rodar 32 (Quadro 4.1 síntese)")

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
        if args.run_32:
            run_32_quadro41(base_csv_path=args.base, encoding=args.encoding, sep=args.sep)
    finally:
        _pause()

if __name__ == "__main__":
    main()
