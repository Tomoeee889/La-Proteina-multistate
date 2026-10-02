#!/usr/bin/env python3
"""
Пере-разметка результатов BioEmu ПОЛНОСТЬЮ из имён папок (без хэш-маппинга).

Читает bioemu_states_summary.csv (n_states_xtc / n_raw_npz там надёжны),
выкидывает checkpoint-мусор, парсит regime/noise/path из имени папки
починенными regex, перестраивает группу-таблицу и 3 фигуры.

Идемпотентен: старые колонки разметки удаляются до concat, есть assert
на дубли, так что повторный запуск не сломается.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==================== ПУТИ ====================
BASE = Path("/home/domain/aristowi/la-proteina-main")
SUMMARY = BASE / "bioemu_states_summary.csv"
OUT = BASE / "bioemu_plots"
OUT.mkdir(parents=True, exist_ok=True)

ENRICHED = OUT / "bioemu_states_summary_v2.csv"
GROUP = OUT / "bioemu_states_group_table_v2.csv"
STATS = OUT / "bioemu_states_stats_v2.csv"

FIG1 = OUT / "fig1_summary_v2.png"
FIG2 = OUT / "fig2_noise_effect_v2.png"
FIG3 = OUT / "fig3_heatmap_noiseB_v2.png"

# ==================== РЕГУЛЯРКИ (починенные) ====================
# regime: task1_default / task2_sim_eps / ...
REGIME_RE = re.compile(r"(task\d+_(?:default|sim_eps))", re.I)
# noise: требую _ или - после слова noise, чтобы не цеплять мусор
NOISE_RE = re.compile(r"noise[_-]([0-9]+(?:\.[0-9]+)?)", re.I)
# path: lookahead вместо \b, потому что '_' — word-символ и \b после 'A'
#        в 'pathA_best' НЕ срабатывал (это и был баг старой разметки)
PATH_RE = re.compile(r"path[_-]?([AB])(?=[_-]|$)", re.I)

# Колонки разметки, которые надо стереть до concat (идемпотентность)
META_COLS = ["regime", "task", "noise", "path_label", "noise_label", "mode"]

# ==================== СТИЛЬ ====================
sns.set_theme(style="whitegrid", context="talk", font="DejaVu Sans")
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "legend.frameon": False,
})

REGIMES = [
    "task1_default", "task1_sim_eps",
    "task2_default", "task2_sim_eps",
    "task3_default", "task3_sim_eps",
]
RCOLOR = {r: c for r, c in zip(REGIMES, sns.color_palette("tab10", 6))}
MODE_COLOR = {"default": "#2E86C1", "sim_eps": "#E67E22"}


# ==================== ПАРСИНГ ====================
def parse_folder(name: str):
    """Возвращает dict с метаданными или None (мусор / не распарсилось)."""
    if not name or name == ".ipynb_checkpoints":
        return None
    if "-checkpoint" in name:
        return None

    m_reg = REGIME_RE.search(name)
    m_noi = NOISE_RE.search(name)
    m_pat = PATH_RE.search(name)
    if not (m_reg and m_noi and m_pat):
        return None

    regime = m_reg.group(1).lower()
    task = regime.split("_", 1)[0]
    return {
        "regime": regime,
        "task": task,
        "noise": float(m_noi.group(1)),
        "path_label": m_pat.group(1).upper(),
    }


# ==================== ОСНОВНОЙ ПРОЦЕСС ====================
def main():
    if not SUMMARY.exists():
        raise FileNotFoundError(f"Не найден входной файл: {SUMMARY}")

    df = pd.read_csv(SUMMARY)
    print(f"Строк в summary: {len(df)}")

    # КРИТИЧНО: в старом summary уже есть regime/task из сломанного
    # хэш-маппинга. Стираем ВСЕ колонки разметки до concat, иначе
    # получатся дубли и pivot_table упадёт с
    # "Grouper for 'regime' not 1-dimensional".
    dropped = [c for c in META_COLS if c in df.columns]
    if dropped:
        print(f"Стираю старые колонки разметки перед перепарсингом: {dropped}")
        df = df.drop(columns=dropped)

    parsed = df["folder"].map(parse_folder)
    ok = parsed.notna()
    print(f"Распарсилось из имени папки: {int(ok.sum())}  "
          f"(мусор/непонятное: {int((~ok).sum())})")

    df = df[ok].copy()
    meta = pd.DataFrame(parsed[ok].tolist(), index=df.index)
    df = pd.concat([df, meta], axis=1)

    # Страховка от дублей (на случай будущего изменения схемы)
    dup = df.columns[df.columns.duplicated()].tolist()
    assert not dup, f"Дубли колонок после concat: {dup}"

    df["n_states_xtc"] = pd.to_numeric(df["n_states_xtc"], errors="coerce")
    df["n_raw_npz"] = pd.to_numeric(df["n_raw_npz"], errors="coerce")
    df["noise_label"] = df["noise"].map(lambda x: f"{x:g}")

    # ---------- ДИАГНОСТИКА СИММЕТРИИ (детектор честности фикса) ----------
    print("\n" + "=" * 70)
    print("СИММЕТРИЯ A/B ПОСЛЕ ФИКСА (ожидаем ~поровну в каждом regime)")
    print("=" * 70)
    sym = (df.pivot_table(index="regime", columns="path_label",
                          values="folder", aggfunc="count")
             .fillna(0).astype(int))
    for col in ["A", "B"]:
        if col not in sym.columns:
            sym[col] = 0
    sym = sym[["A", "B"]]
    sym["total"] = sym.sum(axis=1)
    sym = sym.reindex([r for r in REGIMES if r in sym.index])
    print(sym.to_string())
    print("\nОжидаемый эталон: 500/500 везде, кроме task3_* = 350/350.")
    print("Если сошлось — фикс чист, fig2/fig3 показывают реальный эффект шума.")

    # ---------- СОХРАНЕНИЕ ENRICHED ----------
    df.to_csv(ENRICHED, index=False)
    print(f"\nEnriched v2 сохранён: {ENRICHED}")

    # ---------- СВОДНАЯ СТАТИСТИКА ПО REGIME ----------
    plot_df = df[df["n_states_xtc"].notna()].copy()
    g = plot_df.groupby("regime")["n_states_xtc"]
    stats = pd.DataFrame({
        "n_sequences": g.count(),
        "mean_states": g.mean(),
        "median_states": g.median(),
        "min_states": g.min(),
        "max_states": g.max(),
        "std_states": g.std(),
        "frac_ge95": (plot_df.assign(_=plot_df["n_states_xtc"] >= 95)
                      .groupby("regime")["_"].mean()),
    }).round(3).reindex([r for r in REGIMES if r in plot_df["regime"].unique()])
    stats.to_csv(STATS, index=True)
    print(f"Статистика по regime сохранена: {STATS}")
    print("\n" + "=" * 70)
    print("СТАТИСТИКА ПО REGIME (n_states_xtc)")
    print("=" * 70)
    print(stats.to_string())

    # ---------- ГРУППА-ТАБЛИЦА v2 ----------
    grp = (plot_df.groupby(["task", "path_label", "regime"])["n_states_xtc"]
           .agg(n_sequences="count", mean_states="mean", median_states="median",
                min_states="min", max_states="max", std_states="std")
           .round(3).reset_index())
    grp.to_csv(GROUP, index=False)
    print(f"\nГруппа-таблица v2 сохранена: {GROUP}")
    print("\n" + grp.to_string(index=False))

    # ---------- ФИГУРЫ ----------
    plot_fig1(plot_df, stats)
    plot_fig2(plot_df)
    plot_fig3(plot_df)

    print("\n" + "=" * 70)
    print("ГОТОВО. Фигуры:")
    for f in [FIG1, FIG2, FIG3]:
        print(f"  {f}")
    print("=" * 70)


# ==================== FIGURE 1: сводка по regime ====================
def plot_fig1(plot_df: pd.DataFrame, stats: pd.DataFrame):
    print("\nСтрою fig1 (сводка по regime)...")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(stats))
    colors = [RCOLOR[r] for r in stats.index]

    axes[0].bar(x, stats["mean_states"], yerr=stats["std_states"], capsize=4,
                color=colors, edgecolor="black", linewidth=0.6)
    axes[0].axhline(100, color="#1E8449", ls="--", lw=1.3, label="100 (полный ансамбль)")
    axes[0].axhline(95, color="#C0392B", ls="--", lw=1.3, label="95")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(stats.index, rotation=30, ha="right")
    axes[0].set_ylim(0, 108)
    axes[0].set_ylabel("Среднее валидных структур")
    axes[0].set_title("(a) Средний размер ансамбля ± std")
    axes[0].legend(loc="lower right")

    axes[1].bar(x, stats["frac_ge95"] * 100, color=colors,
                edgecolor="black", linewidth=0.6)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(stats.index, rotation=30, ha="right")
    axes[1].set_ylim(0, 100)
    axes[1].set_ylabel("% последовательностей с ≥95 структур")
    axes[1].set_title("(b) Доля «полных» ансамблей")

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG1)
    plt.close(fig)
    print(f"  сохранено: {FIG1}")


# ==================== FIGURE 2: эффект шума ====================
def plot_fig2(plot_df: pd.DataFrame):
    print("Строю fig2 (эффект шума: B сверху = реальный, A снизу = контроль)...")
    tasks = sorted(plot_df["task"].unique())
    fig, axes = plt.subplots(2, len(tasks),
                             figsize=(4.6 * len(tasks), 8.4),
                             sharey=True, squeeze=False)

    for j, p in enumerate(["B", "A"]):
        for i, t in enumerate(tasks):
            ax = axes[j][i]
            sub = plot_df[(plot_df["path_label"] == p) & (plot_df["task"] == t)].copy()
            if sub.empty or sub["noise"].isna().all():
                ax.text(0.5, 0.5, "нет данных", ha="center", va="center",
                        transform=ax.transAxes)
                ax.set_title(f"{t} | path {p}")
                continue

            sub["mode"] = sub["regime"].str.replace(t + "_", "", regex=False)
            noises = sorted(sub["noise"].dropna().unique())

            for mode in ["default", "sim_eps"]:
                sm = sub[sub["mode"] == mode]
                if sm.empty:
                    continue
                med, q1, q3 = [], [], []
                for nz in noises:
                    v = sm.loc[sm["noise"] == nz, "n_states_xtc"]
                    if len(v) == 0:
                        med.append(np.nan); q1.append(np.nan); q3.append(np.nan)
                        continue
                    med.append(v.median())
                    q1.append(v.quantile(0.25))
                    q3.append(v.quantile(0.75))
                xs = np.arange(len(noises))
                ax.fill_between(xs, q1, q3, color=MODE_COLOR[mode], alpha=0.18)
                ax.plot(xs, med, "-o", color=MODE_COLOR[mode], lw=2.2, ms=5,
                        label=mode)

            ax.set_xticks(np.arange(len(noises)))
            ax.set_xticklabels([f"{n:g}" for n in noises], rotation=45,
                               ha="right", fontsize=8)
            ax.axhline(100, color="#1E8449", ls="--", lw=1.0)
            ax.axhline(95, color="#C0392B", ls="--", lw=1.0)
            ax.set_ylim(0, 108)
            suf = "  (шум реален)" if p == "B" else "  (контроль)"
            ax.set_title(f"{t} | path {p}{suf}", fontsize=11)
            if i == 0:
                ax.set_ylabel("Медиана валидных структур")
            ax.spines[["top", "right"]].set_visible(False)
            if j == 0 and i == 0:
                ax.legend(frameon=False, fontsize=9, title="режим")

    fig.tight_layout()
    fig.savefig(FIG2)
    plt.close(fig)
    print(f"  сохранено: {FIG2}")


# ==================== FIGURE 3: heatmap путь B ====================
def plot_fig3(plot_df: pd.DataFrame):
    print("Строю fig3 (heatmap regime × noise, путь B)...")
    hm = (plot_df[plot_df["path_label"] == "B"]
          .pivot_table(index="regime", columns="noise",
                       values="n_states_xtc", aggfunc="median"))
    hm = hm.reindex([r for r in REGIMES if r in hm.index])

    fig, ax = plt.subplots(figsize=(max(8, 0.5 * hm.shape[1]),
                                    0.6 * hm.shape[0] + 1.5))
    sns.heatmap(hm, annot=True, fmt=".0f", cmap="RdYlGn", vmin=0, vmax=100,
                linewidths=0.5, linecolor="white",
                cbar_kws={"label": "медиана структур"}, ax=ax)
    ax.set_title("Heatmap: медиана валидных структур (путь B) — regime × noise",
                 fontsize=13)
    ax.set_xlabel("noise scale")
    ax.set_ylabel("")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG3)
    plt.close(fig)
    print(f"  сохранено: {FIG3}")


if __name__ == "__main__":
    main()