#!/usr/bin/env python3
"""
ПОЛНАЯ ВАЛИДАЦИЯ БЕЙЗЛАЙНА С НУЛЯ (самодостаточная).

Читает ТОЛЬКО целые источники:
  bioemu_baselines/<folder>/batch_*.npz   (координаты сэмплов, nm -> приводим к Å)
  baselines/<regime>/<core>/<core>_path{A,B}.pdb   (бэкбоны, Å)

Пишет в bioemu_plots/bioemu_baselines_statistics/:
  Таблички:
    bioemu_eval_per_sample.csv        метрики на каждый сэмпл
    bioemu_eval_per_ensemble.csv      агрегат на каждый ансамбль (+ FigA-D источник)
    bioemu_eval_stats_by_regime.csv   сводка по 6 regime
    bioemu_multistate_pairs.csv       пары A+B + bb_rmsd_AB + multistate_success
    bioemu_multistate_stats.csv       success-rate по regime/task/noise
  Графики (подписи на английском, читабельный масштаб):
    eval_figA_rmsd.png/pdf            self-RMSD distributions (path x task)
    eval_figB_tm.png/pdf              self-TM boxplot by regime
    eval_figC_mixing.png/pdf          state-mixing partner_pref vs noise (per task)
    eval_figD_cross.png/pdf           cross-RMSD heatmaps A->B and B->A
    eval_ms_fig1_bb_hist.png/pdf      калибровка T_distinct (распределение bb_rmsd_AB)
    eval_ms_fig2_plasticity.png/pdf   tm_self vs tm_cross, цвет = bb (пластичность vs вырождение)
    eval_ms_fig3_success_vs_bb.png/pdf  HEADLINE: success-rate vs bb-бакет (+ хук модели)
    eval_ms_fig4_success_grid.png/pdf   success-rate regime x noise

Зависимости: numpy pandas matplotlib seaborn tqdm. MDAnalysis опционален (для n_states_xtc).
Быстрый тест:  MS_LIMIT=300 python run_full_validation.py
Хук модели:    MODEL_PAIRS=my_pairs.csv python run_full_validation.py
"""
import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

warnings.filterwarnings("ignore")

try:
    import MDAnalysis as mda
    HAS_MDA = True
except Exception:
    HAS_MDA = False

# ==================== ПУТИ ====================
BASE = Path("/home/domain/aristowi/la-proteina-main")
BIO_DIR = BASE / "bioemu_baselines"
BB_DIR = BASE / "baselines"
OUT = BASE / "bioemu_plots" / "bioemu_baselines_statistics"
OUT.mkdir(parents=True, exist_ok=True)

PER_SAMPLE = OUT / "bioemu_eval_per_sample.csv"
PER_ENS = OUT / "bioemu_eval_per_ensemble.csv"
STATS_REG = OUT / "bioemu_eval_stats_by_regime.csv"
PAIRS_CSV = OUT / "bioemu_multistate_pairs.csv"
MS_STATS = OUT / "bioemu_multistate_stats.csv"
MODEL_PAIRS = os.environ.get("MODEL_PAIRS", "").strip()
MS_LIMIT = int(os.environ.get("MS_LIMIT", "0"))

# ==================== ГЕОМЕТРИЧЕСКИЕ КОНСТАНТЫ ====================
NM_TO_ANG = 10.0
AUTO_UNIT = True
CONTACT_CUTOFF = 8.0
MIN_SEQ_SEP = 4
TM_COEF, TM_SHIFT = 1.24, 1.8

# ==================== ПОРОГИ MULTISTATE (калибруй по Fig1!) ====================
T_TM_SELF = 0.7
T_TM_CROSS = 0.5
T_DISTINCT = 2.0
BB_EDGES = [0, 1, 2, 3, 4, 5, 6, 8, 10, 20, np.inf]
BB_LABELS = ["0-1", "1-2", "2-3", "3-4", "4-5", "5-6", "6-8", "8-10", "10-20", "20+"]

# ==================== РЕГУЛЯРКИ (из имени папки) ====================
REGIME_RE = re.compile(r"(task\d+_(?:default|sim_eps))", re.I)
PATH_RE = re.compile(r"path[_-]?([AB])(?=[_-]|$)", re.I)
CORE_RE = re.compile(r"noise_[0-9.]+_job_[0-9]+_n_[0-9]+_id_[0-9]+")
NOISE_RE = re.compile(r"noise_([0-9.]+)")

sns.set_theme(style="whitegrid", context="paper", font="DejaVu Sans")
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.titleweight": "bold", "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "legend.frameon": False, "font.size": 10,
})
MODE_COLOR = {"default": "#2E86C1", "sim_eps": "#E67E22"}
TASKS = ["task1", "task2", "task3"]
REGIMES = ["task1_default", "task1_sim_eps", "task2_default",
           "task2_sim_eps", "task3_default", "task3_sim_eps"]


# ==================== ГЕОМЕТРИЯ ====================
def read_ca(pdb_path: Path) -> np.ndarray:
    ca = []
    with open(pdb_path) as f:
        for ln in f:
            if ln.startswith(("ATOM", "HETATM")) and ln[12:16].strip() == "CA":
                ca.append([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
    return np.asarray(ca, dtype=np.float64)


def _rg_mean(pos: np.ndarray) -> float:
    c = pos.mean(axis=-2, keepdims=True)
    return float(np.sqrt(np.mean(np.sum((pos - c) ** 2, axis=-1))))


def load_pos(folder: Path) -> np.ndarray | None:
    """[S,L,3] CA-координаты сэмплов, приведённые к Å."""
    chunks = []
    for npz in sorted(folder.glob("batch_*.npz")):
        try:
            d = np.load(npz, allow_pickle=True)
            if "pos" in d:
                p = np.asarray(d["pos"], dtype=np.float64)
                if p.ndim == 3 and p.shape[-1] == 3:
                    chunks.append(p)
        except Exception:
            continue
    if not chunks:
        return None
    pos = np.concatenate(chunks, axis=0)
    if AUTO_UNIT and _rg_mean(pos) < 5.0:   # nm -> Å
        pos = pos * NM_TO_ANG
    return pos


def kabsch(P, Q):
    Pc = P - P.mean(0); Qc = Q - Q.mean(0)
    V, _, Wt = np.linalg.svd(Pc.T @ Qc)
    d = np.sign(np.linalg.det(Wt.T @ V.T))
    R = Wt.T @ np.diag([1.0, 1.0, d]) @ V.T
    Pr = Pc @ R.T
    rmsd = float(np.sqrt(np.mean(np.sum((Pr - Qc) ** 2, axis=1))))
    return rmsd, Pr, Qc


def tm_score(Pr, Qc, L):
    d0 = TM_COEF * (L - 15) ** (1.0 / 3.0) - TM_SHIFT
    if d0 <= 0:
        d0 = 1.0
    dist = np.linalg.norm(Pr - Qc, axis=1)
    return float(np.mean(1.0 / (1.0 + (dist / d0) ** 2)))


def contact_map(X, cutoff):
    L = X.shape[0]
    dist = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
    idx = np.abs(np.subtract.outer(np.arange(L), np.arange(L)))
    return (dist < cutoff) & (idx > MIN_SEQ_SEP)


def q_score(sc, bc):
    nb = bc.sum()
    return float((sc & bc).sum() / nb) if nb else np.nan


def rg(X):
    c = X.mean(0)
    return float(np.sqrt(np.mean(np.sum((X - c) ** 2, axis=1))))


def count_xtc(folder: Path):
    if not HAS_MDA:
        return np.nan
    xtc, topo = folder / "samples.xtc", folder / "topology.pdb"
    if not (xtc.exists() and topo.exists()):
        return np.nan
    try:
        u = mda.Universe(str(topo), str(xtc))
        n = len(u.trajectory); del u; return n
    except Exception:
        return np.nan


# ==================== РАЗМЕТКА ИЗ ИМЕНИ ПАПКИ ====================
def parse_folder(name: str):
    if name == ".ipynb_checkpoints" or "-checkpoint" in name:
        return None
    mr, mp, mc = REGIME_RE.search(name), PATH_RE.search(name), CORE_RE.findall(name)
    if not (mr and mp and mc):
        return None
    regime = mr.group(1).lower()
    core = mc[0]
    mn = NOISE_RE.search(core)
    return dict(regime=regime, task=regime.split("_", 1)[0],
                gen_mode=regime.split("_", 1)[1], path_label=mp.group(1).upper(),
                core=core, noise=float(mn.group(1)) if mn else np.nan)


def bb_rmsd_pair(regime, core):
    pa = BB_DIR / regime / core / f"{core}_pathA.pdb"
    pb = BB_DIR / regime / core / f"{core}_pathB.pdb"
    if not (pa.exists() and pb.exists()):
        return None
    ca, cb = read_ca(pa), read_ca(pb)
    if ca.shape[0] == 0 or ca.shape[0] != cb.shape[0]:
        return None
    return kabsch(ca, cb)[0]


# ==================== СЛОЙ 1: per-sample / per-ensemble ====================
def process_ensemble(folder, info):
    pos = load_pos(folder)
    if pos is None or pos.shape[0] == 0:
        return None, []
    sp = BB_DIR / info["regime"] / info["core"] / f"{info['core']}_path{info['path_label']}.pdb"
    pp = BB_DIR / info["regime"] / info["core"] / f"{info['core']}_path{'B' if info['path_label']=='A' else 'A'}.pdb"
    self_bb = read_ca(sp) if sp.exists() else None
    if self_bb is None or self_bb.shape[0] == 0 or pos.shape[1] != self_bb.shape[0]:
        return None, []
    L = self_bb.shape[0]
    part_bb = read_ca(pp) if pp.exists() else None
    has_part = part_bb is not None and part_bb.shape[0] == L
    bc_self = contact_map(self_bb, CONTACT_CUTOFF)
    bc_part = contact_map(part_bb, CONTACT_CUTOFF) if has_part else None
    rg_self = rg(self_bb)

    rows = []
    for s in range(pos.shape[0]):
        X = pos[s]
        sc = contact_map(X, CONTACT_CUTOFF)
        r_self, Pr, Qc = kabsch(X, self_bb)
        rec = dict(folder=folder.name, sample_idx=s, **{k: info[k] for k in
                 ["regime", "task", "gen_mode", "path_label", "noise"]},
                   rmsd_self=r_self, tm_self=tm_score(Pr, Qc, L),
                   q_self=q_score(sc, bc_self), rg_ratio=rg(X) / rg_self if rg_self else np.nan)
        if has_part:
            r_cr, Prc, Qcc = kabsch(X, part_bb)
            rec.update(rmsd_cross=r_cr, tm_cross=tm_score(Prc, Qcc, L),
                       q_cross=q_score(sc, bc_part),
                       pref="partner" if r_cr < r_self else "self")
        else:
            rec.update(rmsd_cross=np.nan, tm_cross=np.nan, q_cross=np.nan, pref="self")
        rows.append(rec)

    df = pd.DataFrame(rows)
    ens = dict(folder=folder.name, **{k: info[k] for k in
             ["regime", "task", "gen_mode", "path_label", "noise"]},
             n_raw=len(df), n_xtc=count_xtc(folder), seq_len=L,
             rmsd_self_mean=df.rmsd_self.mean(), rmsd_self_median=df.rmsd_self.median(),
             rmsd_self_min=df.rmsd_self.min(), rmsd_self_std=df.rmsd_self.std(),
             frac_lt2=(df.rmsd_self < 2).mean(), frac_lt3=(df.rmsd_self < 3).mean(),
             tm_self_mean=df.tm_self.mean(), q_self_mean=df.q_self.mean(),
             rg_ratio_mean=df.rg_ratio.mean(),
             partner_pref=(df.pref == "partner").mean())
    if has_part:
        ens.update(rmsd_cross_mean=df.rmsd_cross.mean(), tm_cross_mean=df.tm_cross.mean(),
                   q_cross_mean=df.q_cross.mean())
    else:
        ens.update(rmsd_cross_mean=np.nan, tm_cross_mean=np.nan, q_cross_mean=np.nan)
    return ens, rows


def layer1(folders):
    ens_rows, samp_rows, skip = [], [], 0
    for f in tqdm(folders, desc="Layer1 RMSD/TM/Q"):
        info = parse_folder(f.name)
        if info is None:
            skip += 1; continue
        ens, rows = process_ensemble(f, info)
        if ens is None:
            skip += 1; continue
        ens_rows.append(ens); samp_rows.extend(rows)
    df_e = pd.DataFrame(ens_rows); df_s = pd.DataFrame(samp_rows)
    df_e.to_csv(PER_ENS, index=False); df_s.to_csv(PER_SAMPLE, index=False)
    print(f"Layer1: ансамблей={len(df_e)} сэмплов={len(df_s)} пропущено={skip}")
    return df_e


# ==================== СЛОЙ 2: multistate pairs ====================
def layer2(df_e):
    keys = ["regime", "gen_mode", "noise", "folder"]
    df_e = df_e.copy()
    df_e["core"] = df_e["folder"].map(lambda n: (CORE_RE.findall(n) or [None])[0])
    pairs, incomplete = [], 0
    grp = df_e.groupby(["regime", "gen_mode", "noise", "core"], sort=False)
    for _, g in tqdm(grp, desc="Layer2 pairs", total=grp.ngroups):
        g = g.set_index("path_label")
        if "A" not in g.index or "B" not in g.index:
            incomplete += 1; continue
        a, b = g.loc["A"], g.loc["B"]
        bb = bb_rmsd_pair(a["regime"], a["core"])
        if bb is None:
            incomplete += 1; continue
        succ = bool(a["tm_self_mean"] > T_TM_SELF and a["tm_cross_mean"] > T_TM_CROSS and
                    b["tm_self_mean"] > T_TM_SELF and b["tm_cross_mean"] > T_TM_CROSS and
                    bb > T_DISTINCT)
        pairs.append(dict(regime=a["regime"], task=a["task"], gen_mode=a["gen_mode"],
                          noise=a["noise"], core=a["core"], bb_rmsd_AB=bb,
                          tm_self_A=a["tm_self_mean"], tm_cross_A=a["tm_cross_mean"],
                          tm_self_B=b["tm_self_mean"], tm_cross_B=b["tm_cross_mean"],
                          partner_pref_A=a["partner_pref"], partner_pref_B=b["partner_pref"],
                          n_A=a["n_raw"], n_B=b["n_raw"], multistate_success=int(succ)))
    pdf = pd.DataFrame(pairs)
    pdf["bb_bucket"] = pd.cut(pdf["bb_rmsd_AB"], bins=BB_EDGES, labels=BB_LABELS,
                              right=False, include_lowest=True)
    pdf.to_csv(PAIRS_CSV, index=False)
    print(f"Layer2: полных пар={len(pdf)} неполных={incomplete} "
          f"success={pdf['multistate_success'].mean()*100:.1f}%")
    return pdf


# ==================== СВОДНЫЕ ТАБЛИЧКИ ====================
def write_stats(df_e, pdf):
    g = df_e.groupby("regime")
    sr = pd.DataFrame({
        "n": g.size(), "rmsd_self_mean": g.rmsd_self_mean.mean(),
        "rmsd_self_median": g.rmsd_self_median.median(), "rmsd_self_min": g.rmsd_self_min.min(),
        "frac_lt2": g.frac_lt2.mean(), "frac_lt3": g.frac_lt3.mean(),
        "tm_self_mean": g.tm_self_mean.mean(), "q_self_mean": g.q_self_mean.mean(),
        "rg_ratio_mean": g.rg_ratio_mean.mean(), "partner_pref": g.partner_pref.mean(),
        "rmsd_cross_mean": g.rmsd_cross_mean.mean(), "n_xtc_mean": g.n_xtc.mean(),
    }).round(3).reindex([r for r in REGIMES if r in g.groups])
    sr.to_csv(STATS_REG)
    print("\n=== STATS BY REGIME (RMSD/TM) ===\n" + sr.to_string())

    ms = (pdf.groupby(["task", "gen_mode"])
          .agg(n_pairs=("multistate_success", "size"),
               success_rate=("multistate_success", "mean"),
               mean_bb=("bb_rmsd_AB", "mean")).round(3).reset_index())
    ms["success_rate"] = (ms["success_rate"] * 100).round(1)
    ms.to_csv(MS_STATS, index=False)
    print("\n=== MULTISTATE SUCCESS BY TASK x MODE ===\n" + ms.to_string(index=False))
    return sr, ms


# ==================== ФИГУРЫ: RMSD/TM (FigA-D) ====================
def clean(ax, xl="", yl=""):
    if xl: ax.set_xlabel(xl)
    if yl: ax.set_ylabel(yl)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25, ls="--"); ax.grid(axis="x", alpha=0.12, ls="--")


def fig_a(df):
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.2), sharex=True, sharey=True, squeeze=False)
    XMAX, bins = 8.0, np.arange(0, 8.25, 0.25)
    for j, p in enumerate(["A", "B"]):
        for i, t in enumerate(TASKS):
            ax = axes[j][i]; sub = df[(df.path_label == p) & (df.task == t)]
            for m in ["default", "sim_eps"]:
                s = sub[sub["gen_mode"] == m]
                if s.empty: continue
                v = s["rmsd_self_median"].to_numpy()
                ax.hist(v[v <= XMAX], bins=bins, alpha=0.6, label=m,
                        color=MODE_COLOR[m], edgecolor="white", lw=0.4)
                no = int((v > XMAX).sum())
                if no:
                    ax.annotate(f">{XMAX:g}A: {no}", xy=(0.97, 0.95), xycoords="axes fraction",
                                ha="right", va="top", fontsize=8, color="#C0392B",
                                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#C0392B",
                                          lw=0.6, alpha=0.85))
            ax.axvline(2, color="#1E8449", ls="--", lw=1.2); ax.axvline(3.5, color="#C0392B", ls="--", lw=1.2)
            ax.set_xlim(0, XMAX); ax.set_title(f"{t} | path {p}")
            clean(ax, "median self-RMSD (A)" if j == 1 else "", "ensembles" if i == 0 else "")
            if j == 0 and i == 0: ax.legend(title="regime", loc="upper right")
    fig.suptitle("FigA: self-RMSD  (green <2A success, red >3.5A failure)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95]); _save(fig, "eval_figA_rmsd")


def fig_b(df):
    order = [r for r in REGIMES if r in set(df.regime)]
    fig, ax = plt.subplots(figsize=(12.5, 5.2))
    sns.boxplot(df, x="regime", y="tm_self_mean", order=order, hue="gen_mode",
                palette=MODE_COLOR, width=0.72, linewidth=1.1, fliersize=2.5, ax=ax)
    ax.axhline(0.5, color="#C0392B", ls="--", lw=1.2, label="TM=0.5 (same fold)")
    ax.axhline(0.7, color="#1E8449", ls="--", lw=1.2, label="TM=0.7")
    ax.set_ylim(0, 1.03); ax.set_ylabel("mean self-TM-score")
    ax.set_title("FigB: self-TM by regime (length-normalized -> task3 comparable)")
    plt.setp(ax.get_xticklabels(), rotation=28, ha="right", fontsize=9); clean(ax)
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=4, fontsize=9)
    fig.tight_layout(rect=[0, 0.08, 1, 1]); _save(fig, "eval_figB_tm")


def fig_c(df):
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 7.4), sharey=True, squeeze=False)
    for j, p in enumerate(["A", "B"]):
        for i, t in enumerate(TASKS):
            ax = axes[j][i]; sub = df[(df.path_label == p) & (df.task == t)]
            noises = sorted(sub["noise"].dropna().unique())
            if not noises:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"{t} | path {p}"); clean(ax); continue
            xp = np.arange(len(noises))
            for m in ["default", "sim_eps"]:
                s = sub[sub["gen_mode"] == m]
                if s.empty: continue
                med = s.groupby("noise")["partner_pref"].median().reindex(noises)
                ax.plot(xp, med.to_numpy() * 100, "-o", color=MODE_COLOR[m], lw=2.0, ms=4.5, label=m)
            ax.axhline(50, color="0.35", ls=":", lw=1.2)
            ax.annotate("50% = inversion", xy=(0.98, 51), xycoords=("axes fraction", "data"),
                        ha="right", va="bottom", fontsize=7.5, color="0.3")
            ax.set_xticks(xp); ax.set_xticklabels([f"{n:g}" for n in noises], rotation=45, ha="right", fontsize=8)
            ax.set_ylim(-3, 103); ax.set_title(f"{t} | path {p}")
            clean(ax, "noise scale" if j == 1 else "", "% samples closer to partner" if i == 0 else "")
            if j == 0 and i == 0: ax.legend(title="regime", loc="upper left")
    fig.suptitle("FigC: state-mixing - fraction nearer PARTNER backbone (per task)", fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94]); _save(fig, "eval_figC_mixing")


def fig_d(df):
    fig, axes = plt.subplots(1, 2, figsize=(16.5, 4.8))
    for ax, src in zip(axes, ["A", "B"]):
        dst = "B" if src == "A" else "A"
        sub = df[(df.path_label == src) & df.rmsd_cross_mean.notna()]
        if sub.empty:
            ax.text(0.5, 0.5, "no pair", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"cross-RMSD: {src} -> {dst}"); continue
        hm = sub.pivot_table(index="regime", columns="noise", values="rmsd_cross_mean", aggfunc="median")
        hm = hm.reindex([r for r in REGIMES if r in hm.index]).sort_index(axis=1)
        sns.heatmap(hm, annot=True, fmt=".1f", cmap="RdYlGn_r", vmin=0, vmax=10,
                    linewidths=0.6, linecolor="white", annot_kws={"fontsize": 7},
                    cbar_kws={"label": "median cross-RMSD (A)", "shrink": 0.85}, ax=ax)
        ax.set_title(f"cross-RMSD: ensemble {src} -> backbone {dst}", fontsize=11)
        ax.set_xlabel("noise scale"); ax.set_ylabel("")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=9)
    fig.suptitle("FigD: low cross-RMSD = states A,B degenerate", fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.90]); _save(fig, "eval_figD_cross")


# ==================== ФИГУРЫ: MULTISTATE (ms Fig1-4) ====================
def ms_fig1(pdf):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), sharey=True, squeeze=False)
    xmax = float(np.nanpercentile(pdf.bb_rmsd_AB, 99)) + 0.5; bins = np.arange(0, xmax, 0.5)
    for i, t in enumerate(TASKS):
        ax = axes[0][i]; sub = pdf[pdf.task == t]
        for m in ["default", "sim_eps"]:
            s = sub[sub.gen_mode == m]
            if s.empty: continue
            ax.hist(s.bb_rmsd_AB, bins=bins, alpha=0.6, label=m, color=MODE_COLOR[m], edgecolor="white", lw=0.4)
        ax.axvline(T_DISTINCT, color="#C0392B", ls="--", lw=1.4)
        ax.annotate(f"T_distinct={T_DISTINCT:g}", xy=(T_DISTINCT + 0.1, 0.95),
                    xycoords=("data", "axes fraction"), fontsize=8, color="#C0392B", va="top")
        ax.set_xlim(0, xmax); ax.set_title(t)
        clean(ax, "backbone RMSD A-B (A)", "pairs" if i == 0 else "")
        if i == 0: ax.legend(title="regime")
    fig.suptitle("msFig1: state distinctness - calibrate T_distinct here", fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93]); _save(fig, "eval_ms_fig1_bb_hist")


def ms_fig2(pdf):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), sharey=True, squeeze=False)
    sc = None
    for ax, (lab, xc, yc) in zip(axes[0], [("seq_A", "tm_self_A", "tm_cross_A"),
                                            ("seq_B", "tm_self_B", "tm_cross_B")]):
        sc = ax.scatter(pdf[xc], pdf[yc], c=pdf.bb_rmsd_AB, cmap="RdYlGn", vmin=0, vmax=10,
                        s=10, alpha=0.6, edgecolors="none")
        ax.axvline(T_TM_SELF, color="#1E8449", ls="--", lw=1.2)
        ax.axhline(T_TM_CROSS, color="#C0392B", ls="--", lw=1.2)
        ax.add_patch(plt.Rectangle((T_TM_SELF, T_TM_CROSS), 1 - T_TM_SELF, 1 - T_TM_CROSS,
                                   fill=False, edgecolor="black", lw=1.4, ls=":"))
        ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02); ax.set_title(f"{lab} (box = TM success zone)")
        clean(ax, f"tm_self ({lab}->own)", f"tm_cross ({lab}->partner)" if ax is axes[0][0] else "")
    cb = fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.85, pad=0.02)
    cb.set_label("backbone RMSD A-B (A)\ngreen=distinct red=degenerate")
    fig.suptitle("msFig2: plasticity vs degeneracy - top-right folds into both; colour = distinctness",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 0.92, 0.92]); _save(fig, "eval_ms_fig2_plasticity")


def ms_fig3(pdf, model_pdf=None):
    order = [l for l in BB_LABELS if l in set(pdf.bb_bucket.astype(str))]
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.4), sharey=True, squeeze=False)
    for i, t in enumerate(TASKS):
        ax = axes[0][i]
        for m in ["default", "sim_eps"]:
            sub = pdf[(pdf.task == t) & (pdf.gen_mode == m)]
            if sub.empty: continue
            rate = sub.groupby("bb_bucket", observed=False).multistate_success.mean()
            xs = [order.index(l) for l in order if l in rate.index]
            ys = [rate[l] * 100 for l in order if l in rate.index]
            ax.plot(xs, ys, "-o", color=MODE_COLOR[m], lw=2.0, ms=5, label=f"baseline {m}")
            if model_pdf is not None:
                sm = model_pdf[(model_pdf.task == t) & (model_pdf.gen_mode == m)]
                if not sm.empty:
                    rm = sm.groupby("bb_bucket", observed=False).multistate_success.mean()
                    xm = [order.index(l) for l in order if l in rm.index]
                    ym = [rm[l] * 100 for l in order if l in rm.index]
                    ax.plot(xm, ym, "--s", color="#8E44AD", lw=2.0, ms=5, label=f"model {m}")
        ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=45, ha="right", fontsize=8)
        ax.set_ylim(-3, 103); ax.set_title(t)
        clean(ax, "backbone RMSD A-B bucket (A)", "multistate success (%)" if i == 0 else "")
        if i == 0: ax.legend(fontsize=8)
    fig.suptitle("msFig3: success vs state distinctness - model must sit ABOVE baseline at large RMSD",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92]); _save(fig, "eval_ms_fig3_success_vs_bb")


def ms_fig4(pdf):
    piv = (pdf.pivot_table(index="regime", columns="noise", values="multistate_success", aggfunc="mean") * 100)
    piv = piv.reindex([r for r in REGIMES if r in piv.index]).sort_index(axis=1)
    fig, ax = plt.subplots(figsize=(max(9, 0.55 * piv.shape[1]), 0.7 * piv.shape[0] + 1.6))
    sns.heatmap(piv, annot=True, fmt=".0f", cmap="RdYlGn", vmin=0, vmax=100,
                linewidths=0.6, linecolor="white", annot_kws={"fontsize": 7},
                cbar_kws={"label": "success rate (%)", "shrink": 0.85}, ax=ax)
    ax.set_title("msFig4: multistate success by regime x noise (red = pair not designable)", fontsize=11.5)
    ax.set_xlabel("noise scale"); ax.set_ylabel("")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=9)
    fig.tight_layout(); _save(fig, "eval_ms_fig4_success_grid")


def _save(fig, stem):
    fig.savefig(OUT / f"{stem}.png"); fig.savefig(OUT / f"{stem}.pdf"); plt.close(fig)


# ==================== MAIN ====================
def main():
    folders = sorted([p for p in BIO_DIR.iterdir() if p.is_dir()
                      and p.name != ".ipynb_checkpoints" and "-checkpoint" not in p.name])
    if MS_LIMIT > 0:
        folders = folders[:MS_LIMIT]
    print(f"Ансамблей к обработке: {len(folders)}  |  MDAnalysis для n_xtc: {'да' if HAS_MDA else 'НЕТ (n_xtc=NaN)'}")
    if folders:
        pr = load_pos(folders[0])
        if pr is not None:
            print(f"[unit-check] pos shape={pr.shape}, median Rg={_rg_mean(pr):.2f} A (ожидаем ~15-25)")

    df_e = layer1(folders)
    if MS_LIMIT > 0:
        df_e = df_e.sample(min(MS_LIMIT, len(df_e)), random_state=0)
    pdf = layer2(df_e)

    model_pdf = None
    if MODEL_PAIRS and Path(MODEL_PAIRS).exists():
        model_pdf = pd.read_csv(MODEL_PAIRS)
        model_pdf["bb_bucket"] = pd.cut(model_pdf.bb_rmsd_AB, bins=BB_EDGES, labels=BB_LABELS,
                                        right=False, include_lowest=True)
        print(f"[model] наложил кривую модели ({len(model_pdf)} пар)")

    write_stats(df_e, pdf)
    fig_a(df_e); fig_b(df_e); fig_c(df_e); fig_d(df_e)
    ms_fig1(pdf); ms_fig2(pdf); ms_fig3(pdf, model_pdf); ms_fig4(pdf)
    print("\nВСЁ. Таблички и фигуры в:", OUT)


if __name__ == "__main__":
    main()