import os
import hydra
import lightning as L

L.seed_everything(43)
version_base = hydra.__version__
config_path = "/home/domain/aristowi/la-proteina-main/configs/dataset"
hydra.initialize_config_dir(config_dir=f"{config_path}/pdb", version_base=version_base)

cfg = hydra.compose(
    config_name="pdb_train_ucond.yaml",
    return_hydra_config=True,
)
pdb_datamodule = hydra.utils.instantiate(cfg.datamodule)
print("prepare_data start")
pdb_datamodule.prepare_data()
print("prepare_data done")

print("setup start")
pdb_datamodule.setup("fit")
print("setup done")

print("creating dataloader")
pdb_train_dataloader = pdb_datamodule.train_dataloader()
print("dataloader created")