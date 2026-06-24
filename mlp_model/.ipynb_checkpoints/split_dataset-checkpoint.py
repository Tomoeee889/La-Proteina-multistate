# split_dataset.py
import torch
from sklearn.model_selection import train_test_split

PAIRS_PATH = "./train_pairs_clean.pt"
print("📦 Загружаем чистые пары...")
data = torch.load(PAIRS_PATH)
pairs = data["pairs"]

print("🔪 Разделяем: 80% train, 10% val, 10% test")
train_pairs, temp = train_test_split(pairs, test_size=0.2, random_state=42)
val_pairs, test_pairs = train_test_split(temp, test_size=0.5, random_state=42)

torch.save({"pairs": train_pairs}, "train_pairs_split.pt")
torch.save({"pairs": val_pairs}, "val_pairs_split.pt")
torch.save({"pairs": test_pairs}, "test_pairs_split.pt")

print(f"✅ Train: {len(train_pairs)} | Val: {len(val_pairs)} | Test: {len(test_pairs)}")