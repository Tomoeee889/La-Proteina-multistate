"""
dataset_mlp_downloading.py
Часть 1: Скачивание PDB (X-ray + NMR) + PDBFlex + строгое извлечение латентов
"""

import os
import random
import requests
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import torch
from tqdm import tqdm
from loguru import logger
from Bio.PDB import PDBParser
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
    log_dir / "dataset_download_{time:YYYY-MM-DD}.log",
    rotation="10 MB",
    retention="7 days",
    level="DEBUG",
    encoding="utf-8"
)

logger.info("Логгер инициализирован.")

# ====================== КОНФИГУРАЦИЯ ======================
CFG = {
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    "min_length": 80,
    "max_length": 400,
    "max_resolution": 2.5,

    "n_xray_to_query": 15000,
    "n_nmr_to_query": 5000,
    "n_pdbflex_to_query": 4000,

    "pdb_dir": "./mlp_model_dataset/pdb",
    "latents_dir": "./mlp_model_dataset/latents",
    "registry_path": "./mlp_model_dataset/latents/registry.pt",

    # ← Убедись, что путь к чекпоинту правильный!
    "ae_ckpt": "./checkpoints_laproteina/AE1_ucond_512.ckpt",

    "cache_latents": True,
    "save_summary": True,
}

# Добавляем корень проекта
root = os.path.abspath(".")
sys.path.insert(0, root)

try:
    from proteinfoundation.partial_autoencoder.autoencoder import AutoEncoder
    from proteinfoundation.utils.pdb_utils import from_pdb_file
    from proteinfoundation.utils.coors_utils import ang_to_nm
    from openfold.np import residue_constants
    USE_LAPROTEINA = True
    logger.info("La-Proteina модули загружены")
except ImportError as e:
    USE_LAPROTEINA = False
    logger.error(f"La-Proteina модули не найдены! Ошибка: {e}")
    raise

# ====================== RCSB QUERY ======================
def query_rcsb(experiment: str, min_len: int, max_len: int, max_resolution: float = 2.5, n_results: int = 8000) -> List[str]:
    url = "https://search.rcsb.org/rcsbsearch/v2/query"
    
    nodes = [
        {"type": "terminal", "service": "text", "parameters": {
            "attribute": "exptl.method", "operator": "exact_match", "value": experiment}},
        {"type": "terminal", "service": "text", "parameters": {
            "attribute": "rcsb_entry_info.deposited_polymer_monomer_count", "operator": "greater_or_equal", "value": min_len}},
        {"type": "terminal", "service": "text", "parameters": {
            "attribute": "rcsb_entry_info.deposited_polymer_monomer_count", "operator": "less_or_equal", "value": max_len}},
        {"type": "terminal", "service": "text", "parameters": {
            "attribute": "rcsb_entry_info.polymer_entity_count_protein", "operator": "equals", "value": 1}},
    ]

    if "X-RAY" in experiment.upper():
        nodes.append({"type": "terminal", "service": "text", "parameters": {
            "attribute": "rcsb_entry_info.resolution_combined", "operator": "less_or_equal", "value": max_resolution}})

    all_ids = []
    batch_size = 1000
    for start in range(0, n_results, batch_size):
        rows = min(batch_size, n_results - start)
        query = {
            "query": {"type": "group", "logical_operator": "and", "nodes": nodes},
            "return_type": "entry",
            "request_options": {"paginate": {"start": start, "rows": rows}}
        }
        try:
            r = requests.post(url, json=query, timeout=30)
            r.raise_for_status()
            ids = [item["identifier"] for item in r.json().get("result_set", [])]
            all_ids.extend(ids)
            if len(ids) < rows:
                break
        except Exception as e:
            logger.error(f"RCSB query failed ({experiment}), start={start}: {e}")
            break

    logger.info(f"RCSB → найдено {len(all_ids)} структур ({experiment})")
    return all_ids[:n_results]


# ====================== PDBFlex ======================
def fetch_pdbflex_structures(max_total: int = 4000, max_per_cluster: int = 6) -> List[str]:
    logger.info(f"Запрашиваем до {max_total} структур из PDBFlex...")

    pdbs = set()
    seen_clusters = set()

    seed_pdbs = [
        "1a50","1ake","1bmf","1bta","1c2r","1d5r","1d8t","1e2b","1f6m","1g6n",
        "1gcn","1h9z","1j5p","1k8k","1l2y","1m40","1nls","1o1s","1p7t","1q2y",
        "1r6a","1su4","1tca","1ubi","1v4x","1w7w","2acy","2b3i","2c4f","2dn2",
        "2f4j","2gs2","2h3f","2j4k","2n2h","3j9m","4hn4","4hpj","4hpx","4kkx",
        "5e5r","6n4o","6vxx","7s1o","1tim","1gfl","2wfi","3p0g","4hhb","5lzs"
    ]

    api_url = "http://pdbflex.org/php/api/representatives.php"

    for pdb in tqdm(seed_pdbs, desc="PDBFlex clusters"):
        if len(pdbs) >= max_total:
            break
        for chain in ["A", "B", "C"]:
            try:
                r = requests.get(f"{api_url}?pdbID={pdb}&chainID={chain}", timeout=12)
                if r.status_code == 200:
                    try:
                        members = r.json()
                        if isinstance(members, list) and len(members) > 1:
                            cluster_name = members[0]
                            if cluster_name in seen_clusters:
                                continue
                            seen_clusters.add(cluster_name)
                            selected = members[:max_per_cluster]
                            pdbs.update([m.upper() for m in selected])
                    except:
                        pass
            except:
                continue

    final_list = list(pdbs)[:max_total]
    logger.success(f"PDBFlex: собрано {len(final_list)} уникальных PDB ID")
    return final_list


# ====================== DOWNLOAD ======================
def download_pdb(pdb_id: str, out_dir: str) -> Optional[str]:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{pdb_id.lower()}.pdb")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path

    urls = [
        f"https://files.rcsb.org/download/{pdb_id.lower()}.pdb",
        f"https://files.rcsb.org/view/{pdb_id.upper()}.pdb",
    ]
    for url in urls:
        try:
            r = requests.get(url, timeout=20)
            if r.status_code == 200 and r.text.strip():
                with open(path, "w") as f:
                    f.write(r.text)
                logger.debug(f"Скачан {pdb_id}")
                return path
        except Exception:
            pass
    logger.warning(f"Не удалось скачать {pdb_id}")
    return None


# ====================== NMR SPLIT ======================
def split_nmr_models(pdb_path: str, out_dir: str, pdb_id: str) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    with open(pdb_path, "r") as f:
        lines = f.readlines()

    model_lines = []
    current_model = []
    header_lines = []
    in_model = False

    for line in lines:
        record = line[:6].strip()
        if record in ("HEADER", "TITLE", "COMPND", "SOURCE", "REMARK", "SEQRES"):
            header_lines.append(line)
        elif record == "MODEL":
            in_model = True
            current_model = []
        elif record == "ENDMDL":
            in_model = False
            model_lines.append(current_model)
            current_model = []
        elif in_model:
            current_model.append(line)

    if not model_lines:
        return [pdb_path]

    out_paths = []
    for idx, m_lines in enumerate(model_lines):
        out_name = f"{pdb_id}_model_{idx:03d}.pdb"
        out_path = os.path.join(out_dir, out_name)
        if not (os.path.exists(out_path) and os.path.getsize(out_path) > 0):
            with open(out_path, "w") as f:
                f.writelines(header_lines)
                f.writelines(m_lines)
                f.write("END\n")
        out_paths.append(out_path)

    logger.debug(f"{pdb_id}: разбит на {len(out_paths)} моделей")
    return out_paths


# ====================== LOAD BATCH ======================
def load_pdb_as_batch(pdb_path: str, device: str) -> Optional[Dict]:
    if not USE_LAPROTEINA:
        return None
    try:
        prot = from_pdb_file(pdb_path)
    except Exception as e:
        logger.debug(f"from_pdb_file failed for {pdb_path}: {e}")
        return None

    coords = torch.tensor(prot.atom_positions, dtype=torch.float32)
    coord_mask = torch.tensor(prot.atom_mask, dtype=torch.float32)
    aatype = getattr(prot, 'aatype', None)
    if aatype is None:
        return None

    res_type = torch.tensor(aatype, dtype=torch.long)
    coords_nm = ang_to_nm(coords)
    mask = coord_mask[:, 1].bool()

    res_type = res_type[mask]
    coords = coords[mask]
    coords_nm = coords_nm[mask]
    coord_mask = coord_mask[mask]
    mask = coord_mask[:, 1].bool()

    valid_res = res_type < 20
    res_type = res_type[valid_res]
    coords = coords[valid_res]
    coords_nm = coords_nm[valid_res]
    coord_mask = coord_mask[valid_res]
    mask = mask[valid_res]

    n = coords.shape[0]
    if n < CFG["min_length"] or n > CFG["max_length"]:
        return None

    return {
        "coords": coords.unsqueeze(0).to(device),
        "coords_nm": coords_nm.unsqueeze(0).to(device),
        "coord_mask": coord_mask.unsqueeze(0).to(device),
        "residue_type": res_type.unsqueeze(0).to(device),
        "mask": mask.unsqueeze(0).to(device),
        "mask_dict": {
            "coords": coord_mask.unsqueeze(0).unsqueeze(-1).expand(1, n, 37, 3).bool().to(device),
            "residue_type": mask.unsqueeze(0).to(device),
        },
    }


# ====================== STRICT EXTRACT LATENT ======================
def extract_latent(
    pdb_id: str,
    pdb_path: str,
    device: str,
    autoencoder,
    cache_dir: str,
) -> Dict:
    if not USE_LAPROTEINA or autoencoder is None:
        raise RuntimeError(f"{pdb_id}: La-Proteina AutoEncoder не загружен")

    cache_path = os.path.join(cache_dir, f"{pdb_id}.pt")
    if CFG.get("cache_latents", True) and os.path.exists(cache_path):
        try:
            return torch.load(cache_path, map_location="cpu")
        except Exception:
            if os.path.exists(cache_path):
                os.remove(cache_path)

    batch = load_pdb_as_batch(pdb_path, device)
    if batch is None:
        raise RuntimeError(f"{pdb_id}: не удалось создать batch")

    try:
        with torch.no_grad():
            enc_out = autoencoder.encode(batch)

            z = enc_out["z_latent"].squeeze(0).cpu()
            bb_ca = batch["coords_nm"].squeeze(0)[:, 1, :].cpu()
            res_type = batch["residue_type"].squeeze(0).cpu()
            sequence = "".join(
                residue_constants.restypes[r.item()] if r.item() < len(residue_constants.restypes) else "X"
                for r in res_type
            )

            data = {
                "z": z,
                "coords": bb_ca,
                "n": z.shape[0],
                "pdb_id": pdb_id,
                "sequence": sequence,
            }

            if CFG.get("cache_latents", True):
                os.makedirs(cache_dir, exist_ok=True)
                torch.save(data, cache_path)

            logger.debug(f"{pdb_id}: успешно закодировано ({data['n']} остатков)")
            return data
    except Exception as e:
        raise RuntimeError(f"{pdb_id}: ошибка энкодинга La-Proteina: {e}") from e


# ====================== BUILD REGISTRY ======================
def build_registry(pdb_ids: List[str], pdb_dir: str, latents_dir: str, device: str, autoencoder, experiment_type: str) -> Dict:
    registry = {}
    cache_dir = os.path.join(latents_dir, experiment_type.lower())

    paths = {}
    for pdb_id in tqdm(pdb_ids, desc=f"Download {experiment_type}"):
        path = download_pdb(pdb_id, os.path.join(pdb_dir, experiment_type.lower()))
        if path:
            paths[pdb_id] = path

    logger.info(f"Скачано: {len(paths)} из {len(pdb_ids)} ({experiment_type})")

    nmr_split_dir = os.path.join(pdb_dir, "nmr_split")

    for pdb_id, path in tqdm(paths.items(), desc=f"Encode {experiment_type}"):
        try:
            if experiment_type == "NMR":
                model_paths = split_nmr_models(path, nmr_split_dir, pdb_id)
                if len(model_paths) < 2:
                    continue
            else:
                model_paths = [path]

            for model_path in model_paths:
                entry_id = os.path.splitext(os.path.basename(model_path))[0]
                data = extract_latent(entry_id, model_path, device, autoencoder, cache_dir)
                data["experiment_type"] = experiment_type
                registry[entry_id] = data
        except Exception as e:
            logger.error(f"{pdb_id} пропущен: {e}")

    logger.success(f"Валидных структур после энкодера: {len(registry)} ({experiment_type})")
    return registry


# ====================== MAIN ======================
if __name__ == "__main__":
    random.seed(CFG["seed"])
    torch.manual_seed(CFG["seed"])

    for d in [CFG["pdb_dir"], CFG["latents_dir"]]:
        os.makedirs(d, exist_ok=True)

    # === ЗАГРУЗКА АВТОЭНКОДЕРА ===
    autoencoder = None
    if USE_LAPROTEINA and os.path.exists(CFG["ae_ckpt"]):
        try:
            autoencoder = AutoEncoder.load_from_checkpoint(CFG["ae_ckpt"], strict=False)
            autoencoder = autoencoder.to(CFG["device"]).eval()
            for p in autoencoder.parameters():
                p.requires_grad = False
            logger.info("Автоэнкодер загружен успешно")
        except Exception as e:
            logger.error(f"Не удалось загрузить автоэнкодер: {e}")
            raise
    else:
        logger.error("AutoEncoder не найден! Проверьте путь к чекпоинту.")
        raise FileNotFoundError(f"Чекпоинт не найден: {CFG['ae_ckpt']}")

    # === Скачивание ===
    logger.info("Запрос X-ray структур из RCSB...")
    xray_ids = query_rcsb("X-RAY DIFFRACTION", CFG["min_length"], CFG["max_length"], CFG["max_resolution"], CFG["n_xray_to_query"])

    logger.info("Запрос NMR структур из RCSB...")
    nmr_ids = query_rcsb("SOLUTION NMR", CFG["min_length"], CFG["max_length"], 99.0, CFG["n_nmr_to_query"])

    logger.info("Запрос гибких структур из PDBFlex...")
    pdbflex_ids = fetch_pdbflex_structures(CFG["n_pdbflex_to_query"])

    # === Создание registry ===
    xray_reg = build_registry(xray_ids, CFG["pdb_dir"], CFG["latents_dir"], CFG["device"], autoencoder, "X-RAY")
    nmr_reg = build_registry(nmr_ids, CFG["pdb_dir"], CFG["latents_dir"], CFG["device"], autoencoder, "NMR")
    pdbflex_reg = build_registry(pdbflex_ids, CFG["pdb_dir"], CFG["latents_dir"], CFG["device"], autoencoder, "PDBFlex")

    registry = {**xray_reg, **nmr_reg, **pdbflex_reg}

    os.makedirs(os.path.dirname(CFG["registry_path"]), exist_ok=True)
    torch.save(registry, CFG["registry_path"])

    logger.success(f"Registry успешно создан!")
    logger.success(f"Всего структур: {len(registry)}")
    logger.success(f"Путь: {CFG['registry_path']}")