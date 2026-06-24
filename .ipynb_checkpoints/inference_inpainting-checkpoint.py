"""
Запускает La-Proteina motif scaffolding (indexed, dual-path)
на датасете из build_inpainting_dataset.py.

Для каждого образца:
  1. Читает gap_info.txt → строит contig string
  2. Строит motif_dict_cfg для La-Proteina
  3. Запускает inference → получает pathA.pdb и pathB.pdb (две конформации петли)

Запуск:
    cd /home/domain/aristowi/la-proteina-main
    python run_inpainting_scaffolding.py

Требования:
    - La-Proteina окружение активировано
    - dual_path_alpha прописан в configs/inference_base.yaml
"""

import os
import sys
import re
import subprocess
import json
import shutil
import pandas as pd
from pathlib import Path

# ─── Пути ────────────────────────────────────────────────────────────────────
PROJECT_ROOT   = Path("/home/domain/aristowi/la-proteina-main")
DATASET_DIR    = PROJECT_ROOT / "inpainting_dataset"
METADATA_TSV   = DATASET_DIR / "metadata.tsv"
OUT_ROOT       = PROJECT_ROOT / "inference_inpainting"
MOTIF_PDB_DIR  = PROJECT_ROOT / "motif_benchmark_pdb_files"  # La-Proteina ожидает pdbs здесь

PYTHON         = "/home/domain/aristowi/mambaforge/envs/laproteina_env/bin/python"
CKPT_NAME      = "LD4_motif_idx_aa.ckpt"   # indexed all-atom модель
AE_CKPT        = "./checkpoints_laproteina/AE3_motif.ckpt"

# Параметры генерации
NSAMPLES       = 1    # сколько пар (pathA, pathB) на каждый gap
MAX_PER_BATCH  = 2
DUAL_PATH_ALPHA= 0.9   # mixing коэффициент
NSTEPS         = 800

# ─── Чтение gap_info ─────────────────────────────────────────────────────────

def read_gap_info(gap_info_path: Path) -> dict:
    """Читает gap_info.txt → dict."""
    info = {}
    with open(gap_info_path) as f:
        for line in f:
            line = line.strip()
            if ":" in line:
                key, val = line.split(":", 1)
                info[key.strip()] = val.strip()
    return info


def get_pdb_resseq_range(pdb_path: Path, chain_id: str = None):
    """
    Читает PDB файл и возвращает список residue sequence numbers (resseq)
    стандартных аминокислот в первой (или указанной) цепи.
    """
    from Bio.PDB import PDBParser
    from Bio.PDB.Polypeptide import is_aa

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("prot", str(pdb_path))

    for model in structure:
        for chain in model:
            if chain_id and chain.id != chain_id:
                continue
            residues = [r for r in chain
                        if is_aa(r, standard=True) and r.get_id()[0] == " "]
            if residues:
                resseqs = [r.get_id()[1] for r in residues]
                return chain.id, resseqs
    return None, []


def build_contig_string(gap_info: dict, chain_id: str, all_resseqs: list) -> str:
    """
    Строит contig string для La-Proteina из gap_info.

    gap.pdb содержит два фрагмента: до дырки и после.
    Мотив = оба фрагмента (фиксированные координаты).
    Scaffold = gap_size новых остатков между ними.

    Пример результата:
        "A1-45/30-30/A76-120"
        (мотив A1-45, генерируем 30 остатков, мотив A76-120)
    """
    gap_start_idx = int(gap_info["gap_start_idx"])  # 0-based, первый удалённый
    gap_end_idx   = int(gap_info["gap_end_idx"])    # 0-based, первый после дырки
    gap_size      = int(gap_info["gap_size"])

    # resseq до дырки (в original)
    before_resseqs = all_resseqs[:gap_start_idx]
    after_resseqs  = all_resseqs[gap_end_idx:]

    if not before_resseqs or not after_resseqs:
        return None

    # Конструируем contig
    # Формат La-Proteina: Chain + resseq_start + "-" + resseq_end
    before_start = before_resseqs[0]
    before_end   = before_resseqs[-1]
    after_start  = after_resseqs[0]
    after_end    = after_resseqs[-1]

    contig = (
        f"{chain_id}{before_start}-{before_end}"
        f"/{gap_size}-{gap_size}"
        f"/{chain_id}{after_start}-{after_end}"
    )
    return contig


def build_motif_dict_entry(sample_id: str, gap_info: dict,
                            contig: str, motif_pdb_path: str) -> dict:
    """Строит запись для motif_dict_cfg в конфиге La-Proteina."""
    gap_size    = int(gap_info["gap_size"])
    total_res   = int(gap_info["total_residues"])
    gap_start   = int(gap_info["gap_start_idx"])
    gap_end     = int(gap_info["gap_end_idx"])

    before_len  = gap_start
    after_len   = total_res - gap_end
    motif_total = before_len + after_len  # остатки мотива

    # Итоговая длина белка фиксирована = total_res (мотив + gap)
    total_length = total_res

    return {
        "contig_string":   contig,
        "motif_pdb_path":  motif_pdb_path,
        "motif_only":      False,          # gap.pdb содержит только часть белка
        "motif_min_length": total_length,
        "motif_max_length": total_length,
        "segment_order":   "A",
        "atom_selection_mode": "all_atom",
    }


# ─── Генерация YAML конфига ───────────────────────────────────────────────────

def write_inference_config(config_path: Path, sample_id: str,
                           motif_entry: dict, run_name: str):
    """
    Пишет временный YAML конфиг для одного образца.
    Наследует от inference_motif_idx_aa.yaml.
    """
    # Экранируем пути для YAML
    motif_pdb = motif_entry["motif_pdb_path"].replace("\\", "/")

    yaml_content = f"""defaults:
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
        contig_string: "{motif_entry['contig_string']}"
        motif_pdb_path: {motif_pdb}
        motif_only: {str(motif_entry['motif_only']).lower()}
        motif_min_length: {motif_entry['motif_min_length']}
        motif_max_length: {motif_entry['motif_max_length']}
        segment_order: "{motif_entry['segment_order']}"
        atom_selection_mode: "{motif_entry['atom_selection_mode']}"
"""
    config_path.write_text(yaml_content)


# ─── Основной цикл ────────────────────────────────────────────────────────────

def main():
    if not METADATA_TSV.exists():
        print(f"[ERROR] Не найден {METADATA_TSV}")
        print("Сначала запусти build_inpainting_dataset.py")
        sys.exit(1)

    MOTIF_PDB_DIR.mkdir(parents=True, exist_ok=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(METADATA_TSV, sep="\t")
    print(f"Загружено {len(meta)} образцов из metadata.tsv\n")

    results = []
    skipped = 0

    for _, row in meta.iterrows():
        sample_id  = row["sample_id"]       # e.g. "1ubq_A"
        pdb_id     = row["pdb_id"]
        chain_id   = row["chain"]

        sample_dir = DATASET_DIR / sample_id
        gap_info_f = sample_dir / "gap_info.txt"
        original_f = sample_dir / "original.pdb"
        gap_f      = sample_dir / "gap.pdb"

        # Проверяем что все файлы есть
        missing = [f for f in [gap_info_f, original_f, gap_f] if not f.exists()]
        if missing:
            print(f"[SKIP] {sample_id} — нет файлов: {[f.name for f in missing]}")
            skipped += 1
            continue

        print(f"── {sample_id}")

        # Читаем gap_info
        gap_info = read_gap_info(gap_info_f)

        # Получаем resseq из original.pdb (полная нумерация)
        _, all_resseqs = get_pdb_resseq_range(original_f, chain_id)
        if not all_resseqs:
            print(f"   [SKIP] не удалось прочитать resseq из original.pdb")
            skipped += 1
            continue

        # Строим contig string
        contig = build_contig_string(gap_info, chain_id, all_resseqs)
        if not contig:
            print(f"   [SKIP] не удалось построить contig")
            skipped += 1
            continue

        print(f"   contig: {contig}")

        # Копируем gap.pdb в папку мотивов La-Proteina
        motif_pdb_dest = MOTIF_PDB_DIR / f"{sample_id}.pdb"
        shutil.copy2(gap_f, motif_pdb_dest)

        # Строим запись для motif_dict
        motif_entry = build_motif_dict_entry(
            sample_id, gap_info, contig, str(motif_pdb_dest)
        )

        # Пишем временный конфиг
        run_name    = f"inpainting_{sample_id}"
        config_name = f"inpainting_{sample_id}"
        config_path = PROJECT_ROOT / "configs" / f"{config_name}.yaml"
        write_inference_config(config_path, sample_id, motif_entry, run_name)

        print(f"   config: {config_path.name}")

        # Проверяем не запускали ли уже
        out_dir = PROJECT_ROOT / "inference" / run_name
        if out_dir.exists() and any(out_dir.glob("**/*_pathA.pdb")):
            print(f"   [SKIP] уже есть результаты в {out_dir}")
            results.append({
                "sample_id": sample_id,
                "contig":    contig,
                "status":    "already_done",
                "out_dir":   str(out_dir),
            })
            continue

        # Запускаем La-Proteina inference
        cmd = [
            PYTHON, "proteinfoundation/generate.py",
            "--config_name", config_name,
            "--job_id", "0",
        ]
        print(f"   Запускаем: {' '.join(cmd)}")

        try:
            proc = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                capture_output=False,   # выводим в терминал
                text=True,
                timeout=600,            # 10 минут на образец
            )
            status = "ok" if proc.returncode == 0 else f"error_{proc.returncode}"
        except subprocess.TimeoutExpired:
            status = "timeout"
        except Exception as e:
            status = f"exception: {e}"

        print(f"   Статус: {status}\n")

        results.append({
            "sample_id": sample_id,
            "contig":    contig,
            "status":    status,
            "out_dir":   str(out_dir),
        })

        # Удаляем временный конфиг
        # config_path.unlink(missing_ok=True)  # раскомментируй если хочешь чистоту

    # Сводка
    df_res = pd.DataFrame(results)
    out_csv = OUT_ROOT / "scaffolding_results.csv"
    df_res.to_csv(out_csv, index=False)

    print("\n" + "=" * 55)
    print(f"Готово!")
    print(f"  Обработано : {len(results)}")
    print(f"  Пропущено  : {skipped}")
    if len(results):
        ok = df_res[df_res["status"] == "ok"]
        print(f"  Успешно    : {len(ok)}")
        err = df_res[df_res["status"] != "ok"]
        if len(err):
            print(f"  Ошибок     : {len(err)}")
            print(err[["sample_id","status"]].to_string(index=False))
    print(f"  Результаты : {out_csv}")
    print("=" * 55)


if __name__ == "__main__":
    main()