"""
dataset_mlp_pairs.py
Полная версия БЕЗ параллелизма — максимально безопасно для сервера
"""

import os
import random
import torch
import numpy as np
import requests
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional
from tqdm import tqdm
from loguru import logger
import sys
import pymol2
from Bio.Align import PairwiseAligner

# ====================== ЛОГГЕР ======================
log_dir = Path("./logs")
log_dir.mkdir(parents=True, exist_ok=True)
logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{level:8}</level> | {message}", 
           level="INFO", colorize=True)
logger.add(log_dir / "pairs.log", level="INFO", encoding="utf-8", rotation="50 MB")

logger.info("Запуск безопасной версии БЕЗ параллелизма")

# ====================== КОНФИГУРАЦИЯ ======================
CFG = {
    "seed": 42,
    "registry_path": "/home/domain/data/aristowi/mlp_dataset_laproteina/latents/registry.pt",
    "pairs_path": "./train_pairs.pt",
    "pdb_dir": "/home/domain/data/aristowi/mlp_dataset_laproteina/pdb",
    "nmr_split_dir": "/home/domain/data/aristowi/mlp_dataset_laproteina/pdb/nmr_split",

    "n_homolog_pairs": 0,      # уменьшил для безопасности
    "n_random_pairs": 0,
    "n_nmr_pairs": 10000,
    "n_multistate_pairs": 0,

    "min_length": 60,
    "max_length": 400,
    "length_tolerance": 15,

    "homolog_min_seqid": 0.95,
    "homolog_max_seqid": 1.00,
    "homolog_min_rmsd": 2.5,
    "homolog_max_rmsd": 20.0,
    "max_pairs_per_cluster": 35,

    "multistate_min_seqid": 0.83,
    "multistate_max_seqid": 1.0,
    "multistate_min_rmsd": 1.5,
    "multistate_max_rmsd": 20.0,
    "max_pairs_per_uniprot": 50,

    "nmr_min_rmsd": 2.0,
    "nmr_max_rmsd": 12.0,
    "max_nmr_pairs_per_ensemble": 6,

    "random_max_rmsd": 11.0,
}

# ====================== ВЫРАВНИВАТЕЛЬ ПОСЛЕДОВАТЕЛЕЙ ======================
# Глобальный экземпляр — создаётся один раз, работает быстрее
_aligner = PairwiseAligner()
_aligner.mode = 'global'
_aligner.match_score = 2
_aligner.mismatch_score = -1
_aligner.open_gap_score = -2
_aligner.extend_gap_score = -0.5

# ====================== ОДНА СЕССИЯ PyMOL ======================
pymol_session = None

def init_pymol():
    """Инициализация PyMOL сессии — максимально совместимая версия."""
    global pymol_session
    if pymol_session is None:
        try:
            pymol_session = pymol2.PyMOL()
            pymol_session.start()
            
            # 🔥 Только универсальные настройки — каждая в try/except
            for setting_name, setting_value in [
                ("internal_feedback", 0),  # отключить внутренний фидбек (есть везде)
                ("auto_zoom", 0),          # не зумить при загрузке
                ("retain_order", 1),       # сохранять порядок атомов
            ]:
                try:
                    pymol_session.cmd.set(setting_name, setting_value)
                except Exception:
                    pass  # игнорируем, если настройка не поддерживается
            
            # 🔥 Отключение сообщений — через feedback (работает стабильнее чем set)
            try:
                pymol_session.cmd.feedback("disable", "all", "warnings")
                pymol_session.cmd.feedback("disable", "all", "errors")
            except Exception:
                pass
                
            logger.info("✓ PyMOL сессия инициализирована")
            
        except Exception as e:
            logger.error(f"✗ Не удалось инициализировать PyMOL: {e}")
            raise
            
    return pymol_session

def compute_pymol_rmsd(path_a: str, path_b: str) -> Optional[float]:
    """Расчёт RMSD через единственную сессию PyMOL"""
    global pymol_session
    try:
        pymol_session.cmd.delete("all")
        pymol_session.cmd.load(path_a, "mol_a")
        pymol_session.cmd.load(path_b, "mol_b")
        rmsd = pymol_session.cmd.align("mol_a and name CA", "mol_b and name CA")[0]
        return float(rmsd) if 0 < rmsd < 50 else None
    except Exception as e:
        logger.warning(f"PyMOL ошибка: {e}")
        return None


# ====================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ======================
def sequence_identity(seq1: str, seq2: str) -> float:
    if not seq1 or not seq2:
        return 0.0

    if abs(len(seq1) - len(seq2)) > CFG["length_tolerance"] * 3:
        return 0.0

    min_check = min(30, len(seq1), len(seq2))
    if min_check > 0:
        fast_sim = sum(a == b for a, b in zip(seq1[:min_check], seq2[:min_check])) / min_check
        if fast_sim < 0.3:
            return 0.0

    try:
        al = _aligner.align(seq1, seq2)[0]
        matches = 0
        aligned = 0

        # al.aligned = [(start1, end1), ...], [(start2, end2), ...]
        for (s1, e1), (s2, e2) in zip(al.aligned[0], al.aligned[1]):
            block_len = min(e1 - s1, e2 - s2)
            for k in range(block_len):
                aligned += 1
                if seq1[s1 + k] == seq2[s2 + k]:
                    matches += 1

        return matches / aligned if aligned > 0 else 0.0
    except Exception as e:
        logger.warning(f"sequence_identity failed: {e}")
        return 0.0


def get_pdb_path(pdb_id: str) -> str:
    if '_model' in pdb_id:
        return os.path.join(CFG["nmr_split_dir"], f"{pdb_id}.pdb")
    else:
        return os.path.join(CFG["pdb_dir"], "x-ray", f"{pdb_id[:4].lower()}.pdb")

def fetch_uniprot_mapping(pdb_ids: List[str]) -> Dict[str, str]:
    mapping = {}
    url = "https://data.rcsb.org/graphql"
    batch_size = 80
    for i in tqdm(range(0, len(pdb_ids), batch_size), desc="UniProt mapping"):
        batch = pdb_ids[i:i + batch_size]
        ids_str = '["' + '","'.join([x.lower() for x in batch]) + '"]'
        query = f"""{{
          entries(entry_ids: {ids_str}) {{
            rcsb_id
            polymer_entities {{
              rcsb_polymer_entity_container_identifiers {{
                uniprot_ids
              }}
            }}
          }}
        }}"""
        try:
            r = requests.post(url, json={"query": query}, timeout=25)
            r.raise_for_status()
            data = r.json().get("data", {}).get("entries", [])
            for entry in data:
                pdb_id = entry.get("rcsb_id", "").upper()
                for entity in entry.get("polymer_entities", []):
                    uniprot_ids = entity.get("rcsb_polymer_entity_container_identifiers", {}).get("uniprot_ids", [])
                    if uniprot_ids:
                        mapping[pdb_id] = uniprot_ids[0]
                        break
        except:
            continue
    logger.success(f"Получено UniProt ID для {len(mapping)} структур")
    return mapping
# ====================== ФОРМИРОВАНИЕ ПАР ======================

def form_nmr_pairs(registry: Dict, max_pairs: int) -> List[dict]:
    logger.info(f"[NMR] Запуск: цель {max_pairs} пар")
    pairs = []
    checked = 0
    by_base = defaultdict(list)
    for fid in registry:
        if registry[fid].get("experiment_type") == "NMR":
            by_base[fid.split('_model')[0]].append(fid)

    pbar = tqdm(total=max_pairs, desc="[NMR] Пары", file=sys.stderr, 
                unit="pair", leave=True, disable=not sys.stderr.isatty())

    for base, models in by_base.items():
        if len(pairs) >= max_pairs: break
        if len(models) < 2: continue
        models = random.sample(models, min(len(models), CFG["max_nmr_pairs_per_ensemble"]))
        
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                if len(pairs) >= max_pairs: break
                checked += 1
                a, b = models[i], models[j]
                pa, pb = get_pdb_path(a), get_pdb_path(b)
                if not (os.path.isfile(pa) and os.path.isfile(pb)): continue

                rmsd = compute_pymol_rmsd(pa, pb)
                if rmsd is not None and CFG["nmr_min_rmsd"] <= rmsd <= CFG["nmr_max_rmsd"]:
                    pairs.append({"a": a, "b": b, "type": "NMR", "rmsd": round(rmsd, 3)})
                    pbar.update(1)

                if checked % 200 == 0:
                    rate = len(pairs) / checked * 100 if checked > 0 else 0
                    pbar.set_postfix_str(f"проверено={checked} | успех={rate:.1f}%")

    pbar.close()
    logger.success(f"[NMR] ✅ Сформировано {len(pairs)} пар (проверено: {checked} кандидатов)")
    return pairs


def form_multistate_pairs(registry: Dict, max_pairs: int) -> List[dict]:
    logger.info(f"[Multistate] Запуск: цель {max_pairs} пар")
    xray = [p for p in registry if registry[p].get("experiment_type") == "X-RAY"]
    sample = random.sample(xray, min(5000, len(xray)))
    umap = fetch_uniprot_mapping(sample)
    
    by_up = defaultdict(list)
    for p in sample:
        u = umap.get(p.upper())
        if u: by_up[u].append(p)
        
    pairs = []
    checked = 0
    up_cnt = defaultdict(int)
    
    pbar = tqdm(total=max_pairs, desc="[Multistate] Пары", file=sys.stderr, 
                unit="pair", leave=True, disable=not sys.stderr.isatty())

    for uid, pdbs in by_up.items():
        if len(pairs) >= max_pairs: break
        if len(pdbs) < 2: continue
        pdbs.sort()
        
        for i in range(len(pdbs)):
            for j in range(i + 1, len(pdbs)):
                if len(pairs) >= max_pairs: break
                checked += 1
                a, b = pdbs[i], pdbs[j]
                if a[:4] == b[:4]: continue
                if abs(registry[a]["n"] - registry[b]["n"]) > CFG["length_tolerance"]: continue
                
                seqid = sequence_identity(registry[a]["sequence"], registry[b]["sequence"])
                if not (CFG["multistate_min_seqid"] <= seqid <= CFG["multistate_max_seqid"]): continue
                
                pa, pb = get_pdb_path(a), get_pdb_path(b)
                if not (os.path.isfile(pa) and os.path.isfile(pb)): continue
                
                rmsd = compute_pymol_rmsd(pa, pb)
                if rmsd is None or not (CFG["multistate_min_rmsd"] <= rmsd <= CFG["multistate_max_rmsd"]): continue
                
                if up_cnt[uid] >= CFG["max_pairs_per_uniprot"]: continue
                
                pairs.append({"a": a, "b": b, "type": "Multistate", "rmsd": round(rmsd, 3)})
                up_cnt[uid] += 1
                pbar.update(1)
                
                if checked % 200 == 0:
                    rate = len(pairs) / checked * 100 if checked > 0 else 0
                    pbar.set_postfix_str(f"проверено={checked} | успех={rate:.1f}%")

    pbar.close()
    logger.success(f"[Multistate] ✅ Сформировано {len(pairs)} пар (проверено: {checked} кандидатов)")
    return pairs


def form_homolog_pairs(registry: Dict, target_count: int) -> List[dict]:
    logger.info(f"[Homolog] Запуск: цель {target_count} пар")
    fasta_path = Path("temp_homolog.fasta")
    with open(fasta_path, "w") as f:
        for pid, d in registry.items():
            if len(d["sequence"]) >= CFG["min_length"]:
                f.write(f">{pid}\n{d['sequence']}\n")

    cdir = Path("temp_mmseqs_cluster")
    cdir.mkdir(exist_ok=True)
    os.system(f"mmseqs easy-cluster {fasta_path} {cdir}/c {cdir}/t --min-seq-id 0.4 --cov-mode 0 -c 0.65 -s 7.5 --threads 10 > /dev/null 2>&1")
    
    clusters = defaultdict(list)
    tsv = cdir / "c_cluster.tsv"
    if not tsv.exists():
        logger.error("[Homolog] mmseqs не создал TSV"); return []
    with open(tsv) as f:
        for line in f:
            p = line.strip().split("\t")
            if len(p) == 2: clusters[p[0]].append(p[1])
            
    pairs = []
    seen = set()
    checked = 0
    
    pbar = tqdm(total=target_count, desc="[Homolog] Пары", file=sys.stderr, 
                unit="pair", leave=True, disable=not sys.stderr.isatty())

    for members in clusters.values():
        if len(pairs) >= target_count: break
        if len(members) < 2: continue
        random.shuffle(members)
        cnt = 0
        
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                if len(pairs) >= target_count: break
                checked += 1
                a, b = members[i], members[j]
                ba = a.split('_model')[0] if '_model' in a else a
                bb = b.split('_model')[0] if '_model' in b else b
                if ba == bb and '_model' in a and '_model' in b: continue
                if abs(registry[a]["n"] - registry[b]["n"]) > 40: continue
                
                si = sequence_identity(registry[a]["sequence"], registry[b]["sequence"])
                if not (CFG["homolog_min_seqid"] <= si <= CFG["homolog_max_seqid"]): continue
                
                pa, pb = get_pdb_path(a), get_pdb_path(b)
                if not (os.path.isfile(pa) and os.path.isfile(pb)): continue
                
                rmsd = compute_pymol_rmsd(pa, pb)
                if rmsd is None or not (CFG["homolog_min_rmsd"] <= rmsd <= CFG["homolog_max_rmsd"]): continue
                
                key = tuple(sorted([a, b]))
                if key in seen: continue
                
                pairs.append({"a": a, "b": b, "type": "Homolog", "rmsd": round(rmsd, 3)})
                seen.add(key)
                pbar.update(1)
                cnt += 1
                
                if checked % 300 == 0:
                    rate = len(pairs) / checked * 100 if checked > 0 else 0
                    pbar.set_postfix_str(f"проверено={checked} | успех={rate:.1f}%")
            if cnt >= CFG["max_pairs_per_cluster"]: break

    pbar.close()
    logger.success(f"[Homolog] ✅ Сформировано {len(pairs)} пар (проверено: {checked} кандидатов)")
    return pairs


def form_random_pairs(registry: Dict, max_pairs: int) -> List[dict]:
    logger.info(f"[Random] Запуск: цель {max_pairs} пар")

    xray = [p for p in registry if registry[p].get("experiment_type") == "X-RAY"]
    if len(xray) < 2:
        return []

    by_len = defaultdict(list)
    for p in xray:
        by_len[registry[p]["n"]].append(p)

    pairs = []
    pair_set = set()
    used_counts = defaultdict(int)
    checked = 0
    attempts = 0
    max_attempts = max_pairs * 40
    
    pbar = tqdm(total=max_pairs, desc="[Random] Пары", file=sys.stderr, 
                unit="pair", leave=True, disable=not sys.stderr.isatty())

    while len(pairs) < max_pairs and attempts < max_attempts:
        attempts += 1
        checked += 1
        
        weights = [1.0 / (used_counts.get(p, 0) + 1) for p in xray]
        a = random.choices(xray, weights=weights)[0]
        n_a = registry[a]["n"]
        
        pool = [c for d in range(-CFG["length_tolerance"], CFG["length_tolerance"]+1) 
                for c in by_len.get(n_a + d, []) if c != a]
        if not pool: continue
        
        weights_pool = [1.0 / (used_counts.get(c, 0) + 1) for c in pool]
        b = random.choices(pool, weights=weights_pool)[0]
        
        key = tuple(sorted([a, b]))
        if key in pair_set: continue
        
        pa, pb = get_pdb_path(a), get_pdb_path(b)
        if not (os.path.isfile(pa) and os.path.isfile(pb)): continue
        
        rmsd = compute_pymol_rmsd(pa, pb)
        if rmsd is None or rmsd > CFG["random_max_rmsd"]: continue
        
        pairs.append({"a": a, "b": b, "type": "Random", "rmsd": round(rmsd, 3)})
        pair_set.add(key)
        used_counts[a] = used_counts.get(a, 0) + 1
        used_counts[b] = used_counts.get(b, 0) + 1
        pbar.update(1)
        
        if checked % 500 == 0:
            rate = len(pairs) / checked * 100 if checked > 0 else 0
            pbar.set_postfix_str(f"проверено={checked} | успех={rate:.1f}%")

    pbar.close()
    logger.success(f"[Random] ✅ Сформировано {len(pairs)} пар (попыток: {attempts})")
    return pairs


# ====================== MAIN ======================
if __name__ == "__main__":
    random.seed(CFG["seed"])
    torch.manual_seed(CFG["seed"])

    logger.info("Загружаем registry...")
    registry = torch.load(CFG["registry_path"], map_location="cpu")
    logger.success(f"Загружено структур: {len(registry)}")

    logger.info("Начинаем формирование пар...")

    init_pymol()   # инициализируем PyMOL один раз

    nmr_pairs = form_nmr_pairs(registry, CFG["n_nmr_pairs"])
    multistate_pairs = form_multistate_pairs(registry, CFG["n_multistate_pairs"])
    homolog_pairs = form_homolog_pairs(registry, CFG["n_homolog_pairs"])
    random_pairs = form_random_pairs(registry, CFG["n_random_pairs"])

    all_pairs = nmr_pairs + multistate_pairs + homolog_pairs + random_pairs
    random.shuffle(all_pairs)

    logger.success(f"Всего сформировано пар: {len(all_pairs)}")
    logger.info(f"Распределение → NMR: {len(nmr_pairs)} | Multistate: {len(multistate_pairs)} | "
                f"Homolog: {len(homolog_pairs)} | Random: {len(random_pairs)}")

    torch.save({"pairs": all_pairs, "cfg": CFG, "registry_size": len(registry)}, CFG["pairs_path"])
    logger.success(f"Файл успешно сохранён → {CFG['pairs_path']}")