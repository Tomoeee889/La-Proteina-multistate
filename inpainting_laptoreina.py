"""
Запускает La-Proteina motif scaffolding (indexed, dual-path)
на датасете из build_inpainting_dataset.py.

Ключевая логика:
  - gap.pdb содержит ДВА фрагмента мотива (до и после дырки)
  - Ренумеруем gap.pdb с 1 → contig строится по реальным счётчикам
  - motif_min_length = motif_max_length = n_residues из metadata
    (это TOTAL длина = мотив + scaffold, именно это ждёт La-Proteina)

Поддерживает resume.

Запуск:
    cd /home/domain/aristowi/la-proteina-main
    python run_inpainting_scaffolding.py
"""

import sys
import shutil
import subprocess
import pandas as pd
from pathlib import Path

# ─── Пути ────────────────────────────────────────────────────────────────────
PROJECT_ROOT   = Path("/home/domain/aristowi/la-proteina-main")
DATASET_DIR    = PROJECT_ROOT / "inpainting_dataset"
METADATA_TSV   = DATASET_DIR / "metadata.tsv"
OUT_ROOT       = PROJECT_ROOT / "inference_inpainting"
MOTIF_PDB_DIR  = PROJECT_ROOT / "motif_benchmark_pdb_files"

PYTHON         = "/home/domain/aristowi/mambaforge/envs/laproteina_env/bin/python"
CKPT_NAME      = "LD4_motif_idx_aa.ckpt"
AE_CKPT        = "./checkpoints_laproteina/AE3_motif.ckpt"

NSAMPLES        = 1
MAX_PER_BATCH   = 2
DUAL_PATH_ALPHA = 0.65
NSTEPS          = 500

# ─── Вспомогательные функции ─────────────────────────────────────────────────

def load_done_samples(csv_path: Path) -> set:
    if not csv_path.exists():
        return set()
    try:
        df = pd.read_csv(csv_path)
        return set(df[df["status"].isin(["ok", "already_done"])]["sample_id"].tolist())
    except Exception as e:
        print(f"[WARN] не удалось прочитать предыдущий CSV: {e}")
        return set()


def read_gap_info(path: Path) -> dict:
    info = {}
    with open(path) as f:
        for line in f:
            if ":" in line:
                k, v = line.split(":", 1)
                info[k.strip()] = v.strip()
    return info


def renumber_and_count(src: Path, dst: Path, chain_id: str):
    """
    Ренумеровывает остатки gap.pdb с 1 последовательно.
    Возвращает количество остатков в каждом непрерывном сегменте.

    gap.pdb содержит два фрагмента (до и после дырки).
    Определяем границу сегментов по разрыву в оригинальной нумерации.

    Возвращает: (n_before, n_after) — число остатков в первом и втором сегменте.
    """
    from Bio.PDB import PDBParser, PDBIO
    from Bio.PDB.Polypeptide import is_aa

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("p", str(src))

    orig_resseqs = []
    for model in structure:
        for chain in model:
            if chain.id != chain_id:
                continue
            for residue in chain:
                if is_aa(residue, standard=True) and residue.get_id()[0] == " ":
                    orig_resseqs.append(residue.get_id()[1])
        break

    if not orig_resseqs:
        return None, None

    # Находим разрыв в нумерации — это граница между двумя фрагментами мотива
    # Разрыв = там где resseq[i+1] - resseq[i] > 1
    split_idx = None
    for i in range(len(orig_resseqs) - 1):
        if orig_resseqs[i + 1] - orig_resseqs[i] > 1:
            split_idx = i + 1  # индекс первого остатка второго сегмента
            break

    if split_idx is None:
        # Нет разрыва — весь gap.pdb один кусок (не должно быть, но обработаем)
        # Делим пополам как fallback
        split_idx = len(orig_resseqs) // 2

    n_before = split_idx
    n_after  = len(orig_resseqs) - split_idx

    # Ренумеруем последовательно с 1
    for model in structure:
        for chain in model:
            if chain.id != chain_id:
                continue
            i = 1
            for residue in chain:
                if is_aa(residue, standard=True) and residue.get_id()[0] == " ":
                    residue.id = (' ', i, ' ')
                    i += 1
        break

    from Bio.PDB import PDBIO
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(dst))

    return n_before, n_after


def build_contig(chain_id: str, n_before: int, n_after: int, gap_size: int) -> str:
    """
    Строит contig по ренумерованным остаткам.

    После ренумерации:
      - первый сегмент: остатки 1 .. n_before
      - scaffold: gap_size новых остатков
      - второй сегмент: остатки n_before+1 .. n_before+n_after

    Пример: n_before=70, gap_size=35, n_after=25
      → "A1-70/35-35/A71-95"
    """
    seg1_start = 1
    seg1_end   = n_before
    seg2_start = n_before + 1
    seg2_end   = n_before + n_after
    return (f"{chain_id}{seg1_start}-{seg1_end}"
            f"/{gap_size}-{gap_size}"
            f"/{chain_id}{seg2_start}-{seg2_end}")


def write_config(config_path: Path, sample_id: str,
                 contig: str, motif_pdb: str,
                 total_len: int, run_name: str):
    """
    motif_min_length = motif_max_length = total_len = n_residues из metadata.

    La-Proteina ожидает TOTAL длину (мотив + scaffold).
    Scaffold в contig зафиксирован как 'gap_size-gap_size',
    поэтому единственная валидная комбинация — это total_len.
    Устанавливаем min=max=total_len → ровно одна комбинация, нет ошибок.
    """
    yaml_text = f"""defaults:
  - inference_base
  - generation: motif
  - _self_

run_name_: {run_name}
ckpt_name: {CKPT_NAME}
autoencoder_ckpt_path: {AE_CKPT}

generation:
  args:
    dual_path_alpha: {DUAL_PATH_ALPHA}
    nsteps: {NSTEPS}

  model:
    bb_ca:
      simulation_step_params:
        center_every_step: False

  dataset:
    motif_task_name: {sample_id}
    nsamples: {NSAMPLES}
    max_nsamples_per_batch: {MAX_PER_BATCH}

    motif_dict_cfg:
      {sample_id}:
        contig_string: "{contig}"
        motif_pdb_path: {motif_pdb}
        motif_only: false
        motif_min_length: {total_len}
        motif_max_length: {total_len}
        segment_order: "A"
        atom_selection_mode: "all_atom"
"""
    config_path.write_text(yaml_text)


# ─── Основной цикл ────────────────────────────────────────────────────────────

def main():
    if not METADATA_TSV.exists():
        print(f"[ERROR] Не найден {METADATA_TSV}")
        sys.exit(1)

    MOTIF_PDB_DIR.mkdir(parents=True, exist_ok=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(METADATA_TSV, sep="\t")
    print(f"Загружено {len(meta)} образцов из metadata.tsv")

    out_csv  = OUT_ROOT / "scaffolding_results.csv"
    done_set = load_done_samples(out_csv)
    print(f"Уже успешно обработано: {len(done_set)} — пропускаем\n")

    prev_results = []
    if out_csv.exists():
        try:
            prev_results = pd.read_csv(out_csv).to_dict("records")
        except Exception:
            pass

    new_results = []
    skipped = 0

    for _, row in meta.iterrows():
        sample_id = row["sample_id"]
        chain_id  = row["chain"]
        gap_size  = int(row["gap_size"])
        # total_len = мотив + scaffold = n_residues (оригинальная длина цепи)
        total_len = int(row["n_residues"])

        if sample_id in done_set:
            continue

        sample_dir = DATASET_DIR / sample_id
        gap_info_f = sample_dir / "gap_info.txt"
        gap_f      = sample_dir / "gap.pdb"

        missing = [f.name for f in [gap_info_f, gap_f] if not f.exists()]
        if missing:
            print(f"[SKIP] {sample_id} — нет файлов: {missing}")
            skipped += 1
            continue

        print(f"── {sample_id}  (gap_size={gap_size}, total_len={total_len})")

        # Ренумеруем gap.pdb → motif_pdb_dest и узнаём размеры сегментов
        motif_pdb_dest = MOTIF_PDB_DIR / f"{sample_id}.pdb"
        n_before, n_after = renumber_and_count(gap_f, motif_pdb_dest, chain_id)

        if n_before is None:
            print(f"   [SKIP] не удалось прочитать остатки из gap.pdb")
            skipped += 1
            continue

        # Проверка консистентности
        n_motif = n_before + n_after
        if n_motif + gap_size != total_len:
            print(f"   [WARN] n_motif({n_motif}) + gap_size({gap_size}) = "
                  f"{n_motif + gap_size} ≠ total_len({total_len}) из metadata")
            # Используем реальный подсчёт как total_len
            total_len = n_motif + gap_size

        contig = build_contig(chain_id, n_before, n_after, gap_size)

        print(f"   n_before:  {n_before}  n_after: {n_after}")
        print(f"   contig:    {contig}")
        print(f"   total_len: {total_len}")

        # Пишем конфиг
        run_name    = f"inpainting_{sample_id}"
        config_name = f"inpainting_{sample_id}"
        config_path = PROJECT_ROOT / "configs" / f"{config_name}.yaml"
        write_config(config_path, sample_id, contig,
                     str(motif_pdb_dest), total_len, run_name)

        # Пропускаем если уже есть результаты
        out_dir = PROJECT_ROOT / "inference" / run_name
        if out_dir.exists() and any(out_dir.glob("**/*_pathA.pdb")):
            print(f"   [SKIP] уже есть результаты")
            new_results.append({"sample_id": sample_id, "contig": contig,
                                "total_len": total_len, "status": "already_done"})
            pd.DataFrame(prev_results + new_results).to_csv(out_csv, index=False)
            continue

        # Запуск La-Proteina
        cmd = [PYTHON, "proteinfoundation/generate.py",
               "--config_name", config_name, "--job_id", "0"]
        print(f"   Запускаем...")

        try:
            proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT),
                                  timeout=600, text=True)
            status = "ok" if proc.returncode == 0 else f"error_{proc.returncode}"
        except subprocess.TimeoutExpired:
            status = "timeout"
        except Exception as e:
            status = f"exception: {e}"

        print(f"   Статус: {status}\n")
        new_results.append({"sample_id": sample_id, "contig": contig,
                            "total_len": total_len, "status": status})

        # Сохраняем после каждого образца
        pd.DataFrame(prev_results + new_results).to_csv(out_csv, index=False)

    # Финальное сохранение
    df = pd.DataFrame(prev_results + new_results)
    df.to_csv(out_csv, index=False)

    print("\n" + "=" * 55)
    print(f"Новых обработано : {len(new_results)}  |  Пропущено: {skipped}")
    print(f"Всего в CSV      : {len(df)}")
    if len(df):
        ok  = df["status"].isin(["ok", "already_done"]).sum()
        err = (~df["status"].isin(["ok", "already_done"])).sum()
        print(f"Успешно итого    : {ok}  |  Ошибок итого: {err}")
        if err:
            failed = df[~df["status"].isin(["ok", "already_done"])]
            print(failed[["sample_id", "status"]].to_string(index=False))
    print(f"Результаты       : {out_csv}")
    print("=" * 55)


if __name__ == "__main__":
    main()