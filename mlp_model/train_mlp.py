import sys
import os
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger

PROJECT_ROOT = "/home/domain/aristowi/la-proteina-main"
sys.path.insert(0, PROJECT_ROOT)

from proteinfoundation.partial_autoencoder.autoencoder import AutoEncoder
from proteinfoundation.flow_matching.product_space_flow_matcher import ProductSpaceFlowMatcher
from mlp_trainer import MLP_Trainer

def mlp_collate_fn(batch):
    max_len = max(item["z1"].shape[0] for item in batch)
    collated = {
        "z1": [], "coords1": [], "mask1": [], "seq1": [],
        "z2": [], "coords2": [], "mask2": [], "seq2": [],
        "coords1_full": [], "coords2_full": [],
        "atom_mask1": [], "atom_mask2": []
    }
    for item in batch:
        n = item["z1"].shape[0]
        for k in ["z1", "z2"]:
            pad = torch.zeros(max_len, item[k].shape[1], dtype=item[k].dtype)
            pad[:n] = item[k]
            collated[k].append(pad)
        for k in ["seq1", "seq2"]:
            pad = torch.zeros(max_len, dtype=item[k].dtype)
            pad[:n] = item[k]
            collated[k].append(pad)
        for k in ["coords1", "coords2"]:
            pad = torch.zeros(max_len, 3, dtype=item[k].dtype)
            pad[:n] = item[k]
            collated[k].append(pad)
        for k in ["mask1", "mask2"]:
            pad = torch.zeros(max_len, dtype=torch.bool)
            pad[:n] = item[k]
            collated[k].append(pad)
        for k in ["coords1_full", "coords2_full"]:
            if item.get(k) is not None:
                pad = torch.zeros(max_len, 37, 3, dtype=item[k].dtype)
                c = min(item[k].shape[0], max_len)
                pad[:c] = item[k][:c]
                collated[k].append(pad)
            else:
                collated[k].append(None)
        for k in ["atom_mask1", "atom_mask2"]:
            if item.get(k) is not None:
                pad = torch.zeros(max_len, 37, dtype=item[k].dtype)
                c = min(item[k].shape[0], max_len)
                pad[:c] = item[k][:c]
                collated[k].append(pad)
            else:
                collated[k].append(None)
    for k, v in collated.items():
        if all(x is None for x in v):
            collated[k] = None
        else:
            if any(x is None for x in v):
                sample = [x for x in v if x is not None][0]
                v = [torch.zeros_like(sample) if x is None else x for x in v]
            collated[k] = torch.stack(v)
    return collated

class MLPDataset(Dataset):
    def __init__(self, pairs_path, registry):
        self.pairs = torch.load(pairs_path, map_location="cpu")["pairs"]
        self.registry = registry
    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx):
        p = self.pairs[idx]
        i1, i2 = self.registry[p["a"]], self.registry[p["b"]]
        AA_TO_INT = {"A":0,"R":1,"N":2,"D":3,"C":4,"Q":5,"E":6,"G":7,"H":8,"I":9,"L":10,"K":11,"M":12,"F":13,"P":14,"S":15,"T":16,"W":17,"Y":18,"V":19}
        def str_to_tensor(seq_str, target_len):
            if not isinstance(seq_str, str) or not seq_str: return torch.zeros(target_len, dtype=torch.long)
            indices = [AA_TO_INT.get(aa.upper(), 0) for aa in seq_str]
            seq_tensor = torch.tensor(indices, dtype=torch.long)
            if len(seq_tensor) > target_len: return seq_tensor[:target_len]
            elif len(seq_tensor) < target_len: return torch.nn.functional.pad(seq_tensor, (0, target_len - len(seq_tensor)))
            return seq_tensor
        n1, n2 = i1["coords"].shape[0], i2["coords"].shape[0]
        seq1 = str_to_tensor(i1.get("sequence", ""), n1)
        seq2 = str_to_tensor(i2.get("sequence", ""), n2)
        return {
            "z1": i1["z"].float(), "coords1": i1["coords"].float(), "mask1": torch.ones(n1, dtype=torch.bool), "seq1": seq1,
            "z2": i2["z"].float(), "coords2": i2["coords"].float(), "mask2": torch.ones(n2, dtype=torch.bool), "seq2": seq2,
            "coords1_full": i1.get("coords_37", torch.zeros(1)).float(), "coords2_full": i2.get("coords_37", torch.zeros(1)).float(),
            "atom_mask1": i1.get("atom_mask_37", torch.zeros(1)).float(), "atom_mask2": i2.get("atom_mask_37", torch.zeros(1)).float(),
        }

if __name__ == "__main__":
    AE_CKPT_PATH = "/home/domain/aristowi/la-proteina-main/checkpoints_laproteina/AE1_ucond_512.ckpt"
    LD_CKPT_PATH = "/home/domain/aristowi/la-proteina-main/checkpoints_laproteina/LD2_ucond_tri_512.ckpt"
    REGISTRY_PATH = "/home/domain/data/aristowi/mlp_dataset_laproteina/latents/registry_with_full_coords.pt"

    registry = torch.load(REGISTRY_PATH, map_location="cpu")
    torch.set_float32_matmul_precision('medium')

    print("Loading AutoEncoder...")
    autoencoder = AutoEncoder.load_from_checkpoint(AE_CKPT_PATH, strict=False)
    autoencoder.eval()
    for p in autoencoder.parameters(): p.requires_grad = False
    print(f"   Latent dim = {autoencoder.latent_dim}")

    print("Loading Flow Matcher...")
    flow_matcher = ProductSpaceFlowMatcher.load_from_checkpoint(LD_CKPT_PATH, strict=False)
    flow_matcher.eval()
    for p in flow_matcher.parameters(): p.requires_grad = False

    # Patch config for full_simulation compatibility
    if not hasattr(flow_matcher, 'cfg_exp') or flow_matcher.cfg_exp is None:
        flow_matcher.cfg_exp = OmegaConf.create({})
    OmegaConf.set_struct(flow_matcher.cfg_exp, False)
    if not hasattr(flow_matcher.cfg_exp, 'model') or flow_matcher.cfg_exp.model is None:
        flow_matcher.cfg_exp.model = OmegaConf.create({
            "bb_ca": {"schedule": {"mode": "log", "p": 2.0}, "gt": {"mode": "1/t", "p": 1.0, "clamp_val": None},
                      "simulation_step_params": {"sampling_mode": "sc", "sc_scale_noise": 0.1, "sc_scale_score": 1.0, "t_lim_ode": 0.98, "t_lim_ode_below": 0.02, "center_every_step": True}},
            "local_latents": {"schedule": {"mode": "power", "p": 2.0}, "gt": {"mode": "tan", "p": 1.0, "clamp_val": None},
                              "simulation_step_params": {"sampling_mode": "sc", "sc_scale_noise": 0.1, "sc_scale_score": 1.0, "t_lim_ode": 0.98, "t_lim_ode_below": 0.02, "center_every_step": False}}
        })
    if "product_flowmatcher" in flow_matcher.cfg_exp:
        flow_matcher.cfg_exp.product_flowmatcher.local_latents.dim = autoencoder.latent_dim
    OmegaConf.set_struct(flow_matcher.cfg_exp, True)
    print("✅ cfg_exp.model patched")

    mlp_trainer = MLP_Trainer(autoencoder=autoencoder, flow_matcher=flow_matcher, latent_dim=autoencoder.latent_dim)

    train_ds = MLPDataset("train_pairs_split.pt", registry)
    val_ds = MLPDataset("val_pairs_split.pt", registry)
    test_ds = MLPDataset("test_pairs_split.pt", registry)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0, collate_fn=mlp_collate_fn, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0, collate_fn=mlp_collate_fn, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=0, collate_fn=mlp_collate_fn, pin_memory=True)

    checkpoint_callback = ModelCheckpoint(monitor="val/total_loss", dirpath="checkpoints/mlp_mixer/", filename="mlp-{epoch:02d}-{val/total_loss:.4f}", save_top_k=3, mode="min")
    early_stopping = EarlyStopping(monitor="val/total_loss", patience=25, mode="min", min_delta=0.001, check_finite=True)

    pl_trainer = L.Trainer(
        max_epochs=40, accelerator="gpu", devices=1,
        precision="bf16-mixed", # ✅ Lightning автоматически управляет AMP
        logger=[CSVLogger(save_dir="csv_logs", name="mlp_mixer"), TensorBoardLogger("tb_logs", name="mlp_mixer")],
        callbacks=[checkpoint_callback, early_stopping],
        log_every_n_steps=10,
        gradient_clip_val=0.5, gradient_clip_algorithm="norm",
        accumulate_grad_batches=8, # Эффективный batch = 64
        check_val_every_n_epoch=1, limit_val_batches=50,
    )

    print("Starting training...")
    pl_trainer.fit(mlp_trainer, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print("Starting test...")
    pl_trainer.test(mlp_trainer, dataloaders=test_loader)
    print("✅ Done.")