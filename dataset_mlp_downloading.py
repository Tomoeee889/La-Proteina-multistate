"""
dataset_mlp_downloading.py
Часть 1: Скачивание PDB (X-ray + NMR) + строгое извлечение латентов
Только одноцепочечные + гомо-олигомеры (берём одну цепь)
С поддержкой aria2 для быстрого скачивания
"""

import os
import random
import requests
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import torch
from tqdm import tqdm
from loguru import logger
from Bio.PDB import PDBParser
import sys
import concurrent.futures

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

    "min_length": 60,
    "max_length": 400,
    "max_resolution": 2.5,

    "n_xray_to_query": 100000,
    "n_nmr_to_query": 20000,

    "pdb_dir": "/home/domain/data/aristowi/mlp_dataset_laproteina/pdb",
    "latents_dir": "/home/domain/data/aristowi/mlp_dataset_laproteina/latents",
    "registry_path": "/home/domain/data/aristowi/mlp_dataset_laproteina/latents/registry.pt",

    "ae_ckpt": "./checkpoints_laproteina/AE1_ucond_512.ckpt",
    "cache_latents": True,

    "use_aria2": True,          # ← Включи/выключи aria2 здесь
    "aria2_max_concurrent": 32,
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
            "attribute": "rcsb_entry_info.deposited_polymer_monomer_count", 
            "operator": "greater_or_equal", "value": min_len}},
        {"type": "terminal", "service": "text", "parameters": {
            "attribute": "rcsb_entry_info.deposited_polymer_monomer_count", 
            "operator": "less_or_equal", "value": max_len}},
        {"type": "terminal", "service": "text", "parameters": {
            "attribute": "rcsb_entry_info.polymer_entity_count_protein", 
            "operator": "equals", "value": 1}},
    ]

    if "X-RAY" in experiment.upper():
        nodes.append({"type": "terminal", "service": "text", "parameters": {
            "attribute": "rcsb_entry_info.resolution_combined", 
            "operator": "less_or_equal", "value": max_resolution}})

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

# ====================== DOWNLOAD (с aria2) ======================
import concurrent.futures

def download_pdb(pdb_id: str, out_dir: str) -> Optional[str]:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{pdb_id.lower()}.pdb")

    if os.path.exists(path) and os.path.getsize(path) > 1000:
        return path

    url = f"https://files.rcsb.org/download/{pdb_id.lower()}.pdb"

    # === АГРЕССИВНЫЙ aria2 ===
    if CFG.get("use_aria2", True):
        try:
            cmd = [
                "aria2c",
                "--quiet=true",
                "--continue=true",
                "--max-concurrent-downloads=1",      # 1 файл за раз (параллельность будет выше)
                "--split=32",
                "--max-connection-per-server=16",
                "--min-split-size=1M",
                "--timeout=60",
                "--connect-timeout=30",
                "--max-tries=5",
                "--retry-wait=3",
                f"--dir={out_dir}",
                f"--out={pdb_id.lower()}.pdb",
                url
            ]
            result = subprocess.run(cmd, timeout=90, capture_output=True)
            if result.returncode == 0 and os.path.exists(path) and os.path.getsize(path) > 1000:
                logger.debug(f"Скачан (aria2) {pdb_id}")
                return path
        except Exception as e:
            logger.debug(f"aria2 не сработал: {e}")

    # Fallback
    try:
        r = requests.get(url, timeout=40)
        if r.status_code == 200 and len(r.text) > 1000:
            with open(path, "w") as f:
                f.write(r.text)
            logger.debug(f"Скачан (requests) {pdb_id}")
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

# ====================== LOAD BATCH (только одноцепочечные + гомо-олигомеры) ======================
def load_pdb_as_batch(pdb_path: str, device: str) -> Optional[Dict]:
    if not USE_LAPROTEINA:
        return None
    try:
        prot = from_pdb_file(pdb_path)
    except Exception as e:
        logger.debug(f"from_pdb_file failed for {pdb_path}: {e}")
        return None

    chain_ids = np.unique(prot.chain_index)
    n_chains = len(chain_ids)

    if n_chains == 1:
        # Одноцепочечная — всё хорошо
        pass
    elif n_chains > 1:
        # Проверяем, гомо-олигомер ли это
        sequences = []
        for ch in chain_ids:
            mask = prot.chain_index == ch
            seq = "".join(
                residue_constants.restypes[r] if r < len(residue_constants.restypes) else "X"
                for r in prot.aatype[mask]
            )
            sequences.append(seq)

        if len(set(sequences)) == 1:
            # Гомо-олигомер — берём только первую цепь
            logger.debug(f"Гомо-олигомер ({n_chains} цепей) → берём только первую цепь: {pdb_path}")
            first_chain = chain_ids[0]
            mask = prot.chain_index == first_chain
            prot = prot[mask]  # оставляем только атомы первой цепи
        else:
            # Гетеро-структура — отбрасываем
            logger.debug(f"Гетеро-структура ({n_chains} разных цепей) → пропущен: {pdb_path}")
            return None
    else:
        return None

    # Обычная обработка (уже с одной цепью)
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

    # Параллельное скачивание
    paths = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:   # ← вот тут параллельность
        future_to_id = {executor.submit(download_pdb, pdb_id, os.path.join(pdb_dir, experiment_type.lower())): pdb_id 
                        for pdb_id in pdb_ids}
        
        for future in tqdm(concurrent.futures.as_completed(future_to_id), total=len(pdb_ids), desc=f"Download {experiment_type}"):
            pdb_id = future_to_id[future]
            try:
                path = future.result()
                if path:
                    paths[pdb_id] = path
            except Exception as e:
                logger.error(f"Ошибка при скачивании {pdb_id}: {e}")

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

    # === Создание registry ===
    xray_reg = build_registry(xray_ids, CFG["pdb_dir"], CFG["latents_dir"], CFG["device"], autoencoder, "X-RAY")
    nmr_reg = build_registry(nmr_ids, CFG["pdb_dir"], CFG["latents_dir"], CFG["device"], autoencoder, "NMR")

    registry = {**xray_reg, **nmr_reg}

    os.makedirs(os.path.dirname(CFG["registry_path"]), exist_ok=True)
    torch.save(registry, CFG["registry_path"])

    logger.success(f"Registry успешно создан!")
    logger.success(f"Всего структур: {len(registry)}")
    logger.success(f"Путь: {CFG['registry_path']}")