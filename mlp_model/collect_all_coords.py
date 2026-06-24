#!/usr/bin/env python3
"""
Скрипт для добавления полных 3D-координат (Atom37) из PDB файлов в registry.pt
"""
import torch
import os
import glob
from tqdm import tqdm
from Bio.PDB import PDBParser
import numpy as np

# ==========================================
# ⚙️ НАСТРОЙКИ (измени под себя)
# ==========================================
PAIRS_PATH = "./train_pairs.pt"
REGISTRY_PATH = "/home/domain/data/aristowi/mlp_dataset_laproteina/latents/registry.pt"
PDB_DIR = "/home/domain/data/aristowi/mlp_dataset_laproteina/pdb/nmr_split/"  # Папка с .pdb файлами
OUTPUT_PATH = REGISTRY_PATH.replace(".pt", "_with_full_coords.pt")

# Ключи для новых данных в registry
KEY_COORDS_37 = "coords_37"      # [N, 37, 3] в Ангстремах
KEY_ATOM_MASK = "atom_mask_37"   # [N, 37] (1.0 если атом существует)

# Стандартный порядок 37 атомов (совместим с AlphaFold / La-Proteina)
ATOM37_TYPES = [
    "N", "CA", "C", "CB", "O",
    "CG", "CG1", "CG2", "OG", "OG1",
    "SG", "CD", "CD1", "CD2", "ND1", "ND2",
    "OD1", "OD2", "SD", "CE", "CE1", "CE2", "CE3",
    "NE", "NE1", "NE2", "OE1", "OE2", "CH2", "NH1", "NH2",
    "OH", "CZ", "CZ2", "CZ3", "NZ", "OXT"
]
ATOM37_IDX = {atom: i for i, atom in enumerate(ATOM37_TYPES)}
# ==========================================

def parse_pdb_to_atom37(pdb_path: str):
    """Парсит PDB и возвращает тензоры [N, 37, 3] и маску [N, 37]"""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("prot", pdb_path)
    model = structure[0]  # Берём первую модель

    coords_list = []
    mask_list = []

    # Предполагаем одну цепь или берём первую
    chain = next(model.get_chains())
    for residue in chain:
        # Пропускаем гетероатомы (вода, лиганды)
        if residue.get_id()[0] != " ":
            continue
            
        res_coords = np.zeros((37, 3), dtype=np.float32)
        res_mask = np.zeros(37, dtype=np.float32)
        
        for atom in residue:
            atom_name = atom.get_name().strip()
            if atom_name in ATOM37_IDX:
                idx = ATOM37_IDX[atom_name]
                res_coords[idx] = atom.get_coord()  # Координаты в Å
                res_mask[idx] = 1.0
                
        coords_list.append(res_coords)
        mask_list.append(res_mask)

    return np.array(coords_list, dtype=np.float32), np.array(mask_list, dtype=np.float32)


def main():
    print("📦 Загружаем train_pairs...")
    pairs_data = torch.load(PAIRS_PATH, map_location="cpu")
    pairs = pairs_data["pairs"] if isinstance(pairs_data, dict) else pairs_data
    
    # Собираем уникальные ID белков
    unique_ids = set()
    for p in pairs:
        unique_ids.add(p["a"])
        unique_ids.add(p["b"])
    unique_ids = sorted(list(unique_ids))
    print(f"🔍 Найдено {len(unique_ids)} уникальных белков в парах")

    print("📦 Загружаем registry...")
    registry = torch.load(REGISTRY_PATH, map_location="cpu")
    updated_count = 0

    print(f"🔍 Парсим PDB файлы и обновляем registry...")
    for prot_id in tqdm(unique_ids, desc="Обработка белков"):
        if prot_id in registry:
            # Ищем PDB файл (поддерживаем .pdb, .pdb.gz, .ent)
            pdb_candidates = glob.glob(os.path.join(PDB_DIR, f"{prot_id}*"))
            pdb_candidates += glob.glob(os.path.join(PDB_DIR, f"{prot_id.lower()}*"))
            
            if not pdb_candidates:
                print(f"\n⚠️  PDB не найден для {prot_id}. Пропускаю.")
                continue
                
            pdb_path = pdb_candidates[0]  # Берём первый найденный
            
            try:
                coords_37, atom_mask = parse_pdb_to_atom37(pdb_path)
                
                # Добавляем в registry
                registry[prot_id][KEY_COORDS_37] = torch.from_numpy(coords_37)
                registry[prot_id][KEY_ATOM_MASK] = torch.from_numpy(atom_mask)
                updated_count += 1
            except Exception as e:
                print(f"\n❌ Ошибка парсинга {prot_id}: {e}")
        else:
            print(f"\n⚠️  Белок {prot_id} отсутствует в registry. Пропускаю.")

    # Сохраняем с бэкапом оригинала
    backup_path = REGISTRY_PATH + ".bak"
    if not os.path.exists(backup_path):
        os.rename(REGISTRY_PATH, backup_path)
        print(f"💾 Создан бэкап оригинала: {backup_path}")
        
    torch.save(registry, OUTPUT_PATH)
    print(f"\n✅ Готово! Обновлено {updated_count} белков.")
    print(f"📂 Сохранено в: {OUTPUT_PATH}")
    print("💡 Не забудь указать OUTPUT_PATH в train_mlp.py вместо старого registry!")

if __name__ == "__main__":
    main()