"""
Сборка датасета для inpainting.

Для каждого белка:
  1. Скачивает PDB структуру
  2. Выбирает случайную аминокислоту не у края
  3. Удаляет 25-40 соседних аминокислот вокруг неё
  4. Сохраняет: original.pdb и gap.pdb (с дыркой)

Требования:
    pip install biopython requests tqdm

Запуск:
    python build_inpainting_dataset.py
"""

import os
import re
import random
import time
import requests
import numpy as np
from pathlib import Path
from tqdm import tqdm

from Bio.PDB import PDBParser, PDBIO, Select
from Bio.PDB.Polypeptide import is_aa

# ─── Параметры ───────────────────────────────────────────────────────────────
OUT_DIR       = Path("/home/domain/aristowi/la-proteina-main/inpainting_dataset")
N_STRUCTURES  = 100          # сколько структур хотим
GAP_MIN       = 30           # минимум удаляемых остатков
GAP_MAX       = 45           # максимум удаляемых остатков
MIN_LENGTH    = 80           # минимальная длина цепи (чтобы было куда вырезать)
MAX_LENGTH    = 250          # максимальная длина цепи
MAX_RESOLUTION= 2.5          # максимальная разрешённость (Å)
SEED          = 42
SLEEP_SEC     = 0.1          # пауза между запросами к PDB API

random.seed(SEED)
np.random.seed(SEED)

# ─── Вспомогательные классы ──────────────────────────────────────────────────

class ResidueSelect(Select):
    """Сохраняет только указанные residue_id."""
    def __init__(self, keep_ids):
        self.keep_ids = set(keep_ids)

    def accept_residue(self, residue):
        return 1 if residue.get_id() in self.keep_ids else 0


# ─── Функции ─────────────────────────────────────────────────────────────────

def query_pdb_api(n_results: int = 500) -> list:
    """
    Запрашивает список PDB ID через RCSB Search API.
    Фильтры: белок, одна цепь, X-ray, разрешение <= MAX_RESOLUTION,
             длина MIN_LENGTH–MAX_LENGTH.
    """
    url = "https://search.rcsb.org/rcsbsearch/v2/query"
    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "entity_poly.rcsb_entity_polymer_type",
                        "operator": "exact_match",
                        "value": "Protein"
                    }
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "exptl.method",
                        "operator": "exact_match",
                        "value": "X-RAY DIFFRACTION"
                    }
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.resolution_combined",
                        "operator": "less_or_equal",
                        "value": MAX_RESOLUTION
                    }
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.deposited_polymer_monomer_count",
                        "operator": "greater_or_equal",
                        "value": MIN_LENGTH
                    }
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.deposited_polymer_monomer_count",
                        "operator": "less_or_equal",
                        "value": MAX_LENGTH
                    }
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.polymer_entity_count_protein",
                        "operator": "equals",
                        "value": 1
                    }
                }
            ]
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": 0, "rows": n_results},
            "sort": [{"sort_by": "score", "direction": "desc"}]
        }
    }

    resp = requests.post(url, json=query, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    ids = [hit["identifier"] for hit in data.get("result_set", [])]
    print(f"PDB API вернул {len(ids)} структур")
    return ids


def download_pdb(pdb_id: str, out_path: Path) -> bool:
    """Скачивает PDB файл с RCSB."""
    if out_path.exists():
        return True
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200 and len(resp.content) > 100:
            out_path.write_bytes(resp.content)
            return True
    except Exception as e:
        print(f"  [WARN] download {pdb_id}: {e}")
    return False


def get_chain_residues(structure, min_len: int, max_len: int):
    """
    Возвращает список стандартных аминокислотных остатков
    из первой подходящей цепи структуры.
    Возвращает (chain_id, [residue, ...]) или None.
    """
    for model in structure:
        for chain in model:
            residues = [r for r in chain if is_aa(r, standard=True)
                        and r.get_id()[0] == " "]  # исключаем HETATM
            if min_len <= len(residues) <= max_len:
                return chain.id, residues
    return None


def make_gap(residues: list, gap_min: int, gap_max: int):
    """
    Выбирает случайный центр и возвращает:
      keep_ids   — id остатков которые сохраняются
      gap_ids    — id остатков которые удалены
      center_idx — индекс центрального остатка
      gap_size   — фактический размер дырки
    """
    n         = len(residues)
    gap_size  = random.randint(gap_min, gap_max)
    half      = gap_size // 2

    # Центр должен быть достаточно далеко от краёв
    margin    = half + 2
    if n <= 2 * margin:
        return None

    center    = random.randint(margin, n - margin - 1)
    start     = center - half
    end       = start + gap_size  # [start, end)

    gap_ids   = {residues[i].get_id() for i in range(start, min(end, n))}
    keep_ids  = {r.get_id() for r in residues if r.get_id() not in gap_ids}

    return keep_ids, gap_ids, center, gap_size, start, end


def save_pdb(structure, chain_id: str, residue_ids, out_path: Path):
    """Сохраняет структуру, оставляя только указанные остатки в цепи."""
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(out_path), ResidueSelect(residue_ids))


def write_metadata(meta_path: Path, records: list):
    """Сохраняет metadata.tsv."""
    import csv
    if not records:
        return
    with open(meta_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=records[0].keys(), delimiter="\t")
        w.writeheader()
        w.writerows(records)


# ─── Главная функция ──────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_dir = OUT_DIR / "raw_pdb"
    raw_dir.mkdir(exist_ok=True)

    # 1. Получить список PDB ID
    print("Запрашиваем PDB API...")
    try:
        pdb_ids = query_pdb_api(n_results=N_STRUCTURES * 3)  # берём с запасом
    except Exception as e:
        print(f"[ERROR] PDB API: {e}")
        print("Используем fallback-список...")
        # Небольшой ручной fallback если API недоступен
        pdb_ids = [
            "1UBQ","1VII","2HHB","1L2Y","1BDD","1ENH","2KHO","1MBA",
            "1RNB","2LZM","1HRC","1AKE","2CRO","1FKF","1IGD","1WIT",
            "2OVN","1ZWF","1KMB","2WXC","1POH","1FAS","1HEL","2ACE",
            "3LZT","1LMB","2RNM","1PGB","1FRD","2TRX","1OHO","1XNB",
        ]
        random.shuffle(pdb_ids)

    random.shuffle(pdb_ids)

    # 2. Скачиваем и обрабатываем
    parser   = PDBParser(QUIET=True)
    metadata = []
    done     = 0
    skipped  = 0

    print(f"\nОбрабатываем структуры (цель: {N_STRUCTURES})...\n")

    for pdb_id in tqdm(pdb_ids, desc="PDB structures"):
        if done >= N_STRUCTURES:
            break

        pdb_id  = pdb_id.lower()
        raw_pdb = raw_dir / f"{pdb_id}.pdb"

        # Скачать
        ok = download_pdb(pdb_id, raw_pdb)
        if not ok:
            skipped += 1
            continue
        time.sleep(SLEEP_SEC)

        # Парсить
        try:
            structure = parser.get_structure(pdb_id, str(raw_pdb))
        except Exception as e:
            skipped += 1
            continue

        # Найти подходящую цепь
        result = get_chain_residues(structure, MIN_LENGTH, MAX_LENGTH)
        if result is None:
            skipped += 1
            continue
        chain_id, residues = result

        # Создать gap
        gap_result = make_gap(residues, GAP_MIN, GAP_MAX)
        if gap_result is None:
            skipped += 1
            continue
        keep_ids, gap_ids, center_idx, gap_size, gap_start, gap_end = gap_result

        # Папка для этого образца
        sample_dir = OUT_DIR / f"{pdb_id}_{chain_id}"
        sample_dir.mkdir(exist_ok=True)

        # Сохранить оригинал (только выбранная цепь, все остатки)
        all_ids = {r.get_id() for r in residues}
        save_pdb(structure, chain_id, all_ids,
                 sample_dir / "original.pdb")

        # Сохранить структуру с дыркой
        save_pdb(structure, chain_id, keep_ids,
                 sample_dir / "gap.pdb")

        # Записать info о gap
        gap_residue_ids = sorted([r.get_id()[1] for r in residues
                                   if r.get_id() in gap_ids])
        center_resseq = residues[center_idx].get_id()[1]

        info_path = sample_dir / "gap_info.txt"
        with open(info_path, "w") as f:
            f.write(f"pdb_id:        {pdb_id}\n")
            f.write(f"chain:         {chain_id}\n")
            f.write(f"total_residues:{len(residues)}\n")
            f.write(f"gap_size:      {gap_size}\n")
            f.write(f"center_resseq: {center_resseq}\n")
            f.write(f"gap_start_idx: {gap_start}\n")
            f.write(f"gap_end_idx:   {gap_end}\n")
            f.write(f"gap_resseq:    {','.join(map(str, gap_residue_ids))}\n")

        metadata.append({
            "sample_id":      f"{pdb_id}_{chain_id}",
            "pdb_id":         pdb_id,
            "chain":          chain_id,
            "n_residues":     len(residues),
            "gap_size":       gap_size,
            "gap_start_idx":  gap_start,
            "gap_end_idx":    gap_end,
            "center_resseq":  center_resseq,
            "gap_resseq":     ",".join(map(str, gap_residue_ids)),
        })

        done += 1

    # 3. Сохранить metadata
    meta_path = OUT_DIR / "metadata.tsv"
    write_metadata(meta_path, metadata)

    # 4. Итог
    print(f"\n{'='*55}")
    print(f"Готово!")
    print(f"  Успешно обработано : {done}")
    print(f"  Пропущено          : {skipped}")
    print(f"  Выходная папка     : {OUT_DIR}")
    print(f"  Metadata           : {meta_path}")
    print(f"{'='*55}")
    print(f"\nСтруктура папок:")
    print(f"  {OUT_DIR}/")
    print(f"  ├── metadata.tsv          — сводная таблица всех образцов")
    print(f"  ├── raw_pdb/              — скачанные PDB файлы")
    print(f"  └── <pdbid>_<chain>/")
    print(f"       ├── original.pdb     — полная структура цепи")
    print(f"       ├── gap.pdb          — структура с вырезанным регионом")
    print(f"       └── gap_info.txt     — какие остатки удалены")

    # 5. Краткая статистика по gap_size
    if metadata:
        import statistics
        sizes = [m["gap_size"] for m in metadata]
        print(f"\nСтатистика gap_size:")
        print(f"  mean   = {statistics.mean(sizes):.1f}")
        print(f"  median = {statistics.median(sizes):.1f}")
        print(f"  min    = {min(sizes)}")
        print(f"  max    = {max(sizes)}")


if __name__ == "__main__":
    main()