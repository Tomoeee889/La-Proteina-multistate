"""
dataset_mlp_pairs.py
Часть 2: Формирование всех типов пар с явной меткой типа
"""

import os
import random
import torch
import requests
from pathlib import Path
from collections import defaultdict
from typing import Dict, List
from tqdm import tqdm
from loguru import logger
import sys

# ====================== НАСТРОЙКА ЛОГГЕРА ======================
log_dir = Path("./mlp_model_dataset/logs")
log_dir.mkdir(parents=True, exist_ok=True)

logger.remove()

logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level:8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - {message}",
    level="INFO",
    colorize=True
)

logger.add(
    log_dir / "pairs.log",
    level="DEBUG",
    encoding="utf-8",
    rotation="50 MB",
    retention="10 days"
)

logger.info("Логгер инициализирован")
logger.info(f"Логи будут сохраняться в: {log_dir / 'pairs.log'}")

# ====================== КОНФИГУРАЦИЯ ======================
CFG = {
    "seed": 42,
    "registry_path": "./mlp_model_dataset/latents/registry.pt",
    "pairs_path": "./mlp_model_dataset/train_pairs.pt",

    "n_homolog_pairs": 12000,
    "n_random_pairs": 6000,
    "n_nmr_pairs": 10000,
    "n_multistate_pairs": 10000,

    "min_length": 80,
    "max_length": 400,
    "length_tolerance": 8,

    "homolog_min_seqid": 0.3,
    "homolog_max_seqid": 0.75,
    "max_pairs_per_cluster": 6,
    "max_pairs_per_uniprot": 10,

    "multistate_min_rmsd": 1.3,
    "multistate_min_seqid": 0.85,

    "alpha_min": 0.35,
    "alpha_max": 0.90,
    "noise_scale_z": 0.12,
    "noise_scale_ca": 0.06,
}

# ====================== ВСПОМОГАТЕЛЬНЫЕ ======================
def sequence_identity(seq1: str, seq2: str) -> float:
    min_len = min(len(seq1), len(seq2))
    if min_len == 0: return 0.0
    matches = sum(a == b for a, b in zip(seq1[:min_len], seq2[:min_len]))
    return matches / min_len


def ca_rmsd(coords_a: torch.Tensor, coords_b: torch.Tensor) -> float:
    n = min(len(coords_a), len(coords_b))
    diff = coords_a[:n] - coords_b[:n]
    return torch.sqrt((diff ** 2).sum(-1).mean()).item() * 10.0


def fetch_uniprot_mapping(pdb_ids: List[str]) -> Dict[str, str]:
    logger.info(f"Получаем UniProt mapping для {len(pdb_ids)} структур...")
    mapping = {}
    url = "https://data.rcsb.org/graphql"
    batch_size = 100

    for i in tqdm(range(0, len(pdb_ids), batch_size), desc="UniProt mapping"):
        batch = pdb_ids[i:i + batch_size]
        ids_str = '["' + '","'.join([x.lower() for x in batch]) + '"]'
        query = f"""
        {{
          entries(entry_ids: {ids_str}) {{
            rcsb_id
            polymer_entities {{
              rcsb_polymer_entity_container_identifiers {{
                uniprot_ids
              }}
            }}
          }}
        }}
        """
        try:
            r = requests.post(url, json={"query": query}, timeout=20)
            data = r.json().get("data", {}).get("entries", [])
            for entry in data:
                pdb_id = entry.get("rcsb_id", "").upper()
                for entity in entry.get("polymer_entities", []):
                    uniprot_ids = entity.get("rcsb_polymer_entity_container_identifiers", {}).get("uniprot_ids", [])
                    if uniprot_ids:
                        mapping[pdb_id] = uniprot_ids[0]
                        break
        except Exception:
            continue

    logger.success(f"Получено UniProt ID для {len(mapping)} структур")
    return mapping


# ====================== ФОРМИРОВАНИЕ ПАР ======================

def form_nmr_pairs(registry: Dict, max_pairs: int) -> List[dict]:
    """NMR pairs — случайный выбор двух моделей из одного ансамбля"""
    by_base = defaultdict(list)
    for full_id in registry.keys():
        base = full_id.split('_model')[0]
        by_base[base].append(full_id)

    pairs = []
    for models in by_base.values():
        if len(models) < 2:
            continue
        models = sorted(models)
        random.shuffle(models)
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                pairs.append({
                    "a": models[i],
                    "b": models[j],
                    "type": "NMR"
                })
                if len(pairs) >= max_pairs:
                    logger.success(f"NMR pairs: сформировано {len(pairs)}")
                    return pairs
    logger.success(f"Сформировано {len(pairs)} NMR pairs")
    return pairs


def form_multistate_pairs(registry: Dict, max_pairs: int) -> List[dict]:
    """Multistate pairs — только между разными PDB ID"""
    logger.info("Формирование Multistate pairs (X-ray / разные эксперименты)...")
    xray_like_pdbs = [pdb for pdb in registry.keys() if "_model" not in pdb]
    uniprot_map = fetch_uniprot_mapping(xray_like_pdbs)

    by_uniprot = defaultdict(list)
    for pdb_id in xray_like_pdbs:
        uniprot = uniprot_map.get(pdb_id.upper())
        if uniprot:
            by_uniprot[uniprot].append(pdb_id)

    pairs = []
    for uniprot_id, pdbs in tqdm(by_uniprot.items(), desc="Multistate"):
        if len(pdbs) < 2:
            continue

        count = 0  # счётчик пар для этой UniProt группы
        pdbs = sorted(pdbs)

        for i in range(len(pdbs)):
            for j in range(i + 1, len(pdbs)):
                a, b = pdbs[i], pdbs[j]

                seqid = sequence_identity(registry[a]["sequence"], registry[b]["sequence"])
                if seqid < CFG["multistate_min_seqid"]:
                    continue

                rmsd = ca_rmsd(registry[a]["coords"], registry[b]["coords"])
                if rmsd < CFG["multistate_min_rmsd"]:
                    continue

                pairs.append({"a": a, "b": b, "type": "Multistate"})
                count += 1

                if count >= CFG["max_pairs_per_uniprot"]:
                    break

            if count >= CFG["max_pairs_per_uniprot"]:
                break

            if len(pairs) >= max_pairs:
                logger.success(f"Multistate pairs: достигнут общий лимит ({len(pairs)})")
                return pairs

    logger.success(f"Сформировано {len(pairs)} Multistate pairs")
    return pairs


def form_homolog_pairs(registry: Dict, max_pairs_per_cluster: int = 6) -> List[dict]:
    """Homolog pairs через mmseqs2 — только между РАЗНЫМИ PDB ID"""
    logger.info("Запуск кластеризации mmseqs2 для homolog pairs...")

    # 1. Создаём fasta
    fasta_path = Path("temp_homolog.fasta")
    with open(fasta_path, "w") as f:
        for pdb_id, data in registry.items():
            f.write(f">{pdb_id}\n{data['sequence']}\n")

    # 2. Кластеризация
    cluster_dir = Path("temp_mmseqs_cluster")
    cluster_dir.mkdir(exist_ok=True)

    cmd = (
        f"mmseqs easy-cluster {fasta_path} {cluster_dir}/cluster {cluster_dir}/tmp "
        f"--min-seq-id 0.3 --cov-mode 0 -c 0.8 -s 7.0 --threads 8 --cluster-mode 0"
    )
    os.system(cmd)

    # 3. Читаем кластеры
    clusters = defaultdict(list)
    cluster_tsv = cluster_dir / "cluster_cluster.tsv"

    if not cluster_tsv.exists():
        logger.error("mmseqs2 не создал TSV файл")
        return []

    with open(cluster_tsv) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                rep, member = parts
                clusters[rep].append(member)

    # 4. Формируем пары ТОЛЬКО между разными базовыми PDB ID
    pairs = []
    for rep_id, members in tqdm(clusters.items(), desc="Формирование homolog pairs"):
        if len(members) < 2:
            continue

        # Извлекаем базовый PDB ID для каждой модели
        base_members = [(m, m.split('_model')[0]) for m in members]

        random.shuffle(base_members)
        count = 0

        for i in range(len(base_members)):
            for j in range(i + 1, len(base_members)):
                pdb_a, base_a = base_members[i]
                pdb_b, base_b = base_members[j]

                # ← Ключевой фильтр: не берём модели одного PDB ID
                if base_a == base_b:
                    continue

                pairs.append({
                    "a": pdb_a,
                    "b": pdb_b,
                    "type": "Homolog"
                })
                count += 1
                if count >= max_pairs_per_cluster:
                    break
            if count >= max_pairs_per_cluster:
                break

        if len(pairs) >= CFG["n_homolog_pairs"]:
            break

    logger.success(f"Сформировано {len(pairs)} Homolog pairs (только разные PDB ID)")
    return pairs[:CFG["n_homolog_pairs"]]


def form_random_pairs(registry: Dict, max_pairs: int) -> List[dict]:
    """Random pairs"""
    logger.info("Формирование Random pairs...")
    by_length = defaultdict(list)
    for pdb_id, data in registry.items():
        by_length[data["n"]].append(pdb_id)

    pairs = []
    pair_set = set()
    pdb_ids = list(registry.keys())

    for _ in tqdm(range(max_pairs * 25), desc="Random pairs", leave=False):
        a = random.choice(pdb_ids)
        n_a = registry[a]["n"]
        candidates = []
        for d in range(-CFG["length_tolerance"], CFG["length_tolerance"] + 1):
            candidates.extend(by_length.get(n_a + d, []))
        candidates = [c for c in candidates if c != a]
        if not candidates:
            continue
        b = random.choice(candidates)
        key = tuple(sorted([a, b]))
        if key not in pair_set:
            pairs.append({"a": a, "b": b, "type": "Random"})
            pair_set.add(key)
        if len(pairs) >= max_pairs:
            break

    logger.success(f"Сформировано {len(pairs)} Random pairs")
    return pairs


# ====================== MAIN ======================
if __name__ == "__main__":
    random.seed(CFG["seed"])
    torch.manual_seed(CFG["seed"])

    logger.info("Загружаем registry...")
    registry = torch.load(CFG["registry_path"], map_location="cpu")
    logger.success(f"Загружено структур: {len(registry)}")

    logger.info("Начинаем формирование пар...")

    nmr_pairs       = form_nmr_pairs(registry, CFG["n_nmr_pairs"])
    multistate_pairs = form_multistate_pairs(registry, CFG["n_multistate_pairs"])
    homolog_pairs   = form_homolog_pairs(registry, CFG["max_pairs_per_cluster"])
    random_pairs    = form_random_pairs(registry, CFG["n_random_pairs"])

    all_pairs = nmr_pairs + multistate_pairs + homolog_pairs + random_pairs
    random.shuffle(all_pairs)

    logger.success(f"Всего сформировано пар: {len(all_pairs)}")
    logger.info(f"Распределение → NMR: {len(nmr_pairs)} | Multistate: {len(multistate_pairs)} | "
                f"Homolog: {len(homolog_pairs)} | Random: {len(random_pairs)}")

    # Сохраняем в новом формате
    torch.save({
        "pairs": all_pairs,           # список словарей с "type"
        "cfg": CFG,
        "registry_size": len(registry)
    }, CFG["pairs_path"])

    logger.success(f"Файл с парами успешно сохранён → {CFG['pairs_path']}")
    logger.success("Каждая пара теперь содержит поле 'type'")