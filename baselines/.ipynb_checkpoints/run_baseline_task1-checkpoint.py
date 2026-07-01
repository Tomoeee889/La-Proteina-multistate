"""
Baseline Task 1: Unconditional generation with small noise neighborhoods.
Generates pairs of structures with different noise_scale for path B.
Uses original La-Proteina pipeline without MLP and without path mixing.

Usage (from la-proteina-main/ directory):
    python baselines/run_baseline_task1.py --noise_scale 0.1 --num_pairs 50
"""

import os
import sys

# Change working directory to project root so all paths work correctly
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
os.chdir(PROJECT_ROOT)

# Add project root to path for imports
sys.path.insert(0, PROJECT_ROOT)

# Fix module name mismatch: proteina.py imports from mlp_model_dataset,
# but the folder is named mlp_model
MLP_MODEL_PATH = os.path.join(PROJECT_ROOT, "mlp_model")
if os.path.exists(MLP_MODEL_PATH):
    # Create mlp_model_dataset package structure
    import types
    mlp_dataset_pkg = types.ModuleType('mlp_model_dataset')
    mlp_dataset_pkg.__path__ = [MLP_MODEL_PATH]
    sys.modules['mlp_model_dataset'] = mlp_dataset_pkg
    
    # Import mlp_mixer module and register it
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "mlp_model_dataset.mlp_mixer",
        os.path.join(MLP_MODEL_PATH, "mlp_mixer.py")
    )
    if spec and spec.loader:
        mlp_mixer_module = importlib.util.module_from_spec(spec)
        sys.modules['mlp_model_dataset.mlp_mixer'] = mlp_mixer_module
        spec.loader.exec_module(mlp_mixer_module)

import argparse
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import lightning as L
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from loguru import logger

from proteinfoundation.datasets.gen_dataset import GenDataset
from proteinfoundation.proteina import Proteina
from proteinfoundation.utils.pdb_utils import write_prot_to_pdb


def load_model(cfg: DictConfig) -> Proteina:
    """Load model and configure inference without MLP."""
    ckpt_path = cfg.ckpt_path
    ckpt_name = cfg.ckpt_name
    ckpt_file = os.path.join(ckpt_path, ckpt_name)
    logger.info(f"Using checkpoint {ckpt_file}")
    assert os.path.exists(ckpt_file), f"Not a valid checkpoint {ckpt_file}"

    autoencoder_ckpt_path = cfg.get("autoencoder_ckpt_path", None)
    model = Proteina.load_from_checkpoint(
        ckpt_file,
        strict=False,
        autoencoder_ckpt_path=autoencoder_ckpt_path,
    )

    # Disable MLP for baseline
    model.configure_inference(
        inf_cfg=cfg.generation,
        nn_ag=None,
        mlp_mixer=None,
    )

    logger.info("MLP disabled for baseline experiment")
    return model


class NoiseScaleWrapper:
    """
    Wraps model.predict_step to inject init_noise_scale parameter
    into full_simulation call.
    """
    def __init__(self, model: Proteina, noise_scale: float):
        self.model = model
        self.noise_scale = noise_scale
        self._original_predict_step = model.predict_step

    def patched_predict_step(self, batch, batch_idx):
        """Patched predict_step that passes init_noise_scale to full_simulation."""
        original_full_sim = self.model.flow_matcher.full_simulation

        def patched_full_sim(*args, **kwargs):
            kwargs["init_noise_scale"] = self.noise_scale
            kwargs["dual_path_alpha"] = 1.0  # No path mixing
            kwargs["mlp_mixer"] = None  # MLP disabled
            kwargs["mlp_t_threshold"] = 1.1  # MLP not applied
            return original_full_sim(*args, **kwargs)

        self.model.flow_matcher.full_simulation = patched_full_sim
        try:
            result = self._original_predict_step(batch, batch_idx)
        finally:
            self.model.flow_matcher.full_simulation = original_full_sim

        return result


def save_baseline_predictions(
    root_path: str,
    predictions: List[List[Tuple[torch.Tensor]]],
    noise_scale: float,
    job_id: int = 0,
) -> None:
    """Save generated samples with noise_scale in directory name."""
    predictions_flat = [sample for sublist in predictions for sample in sublist]

    samples_per_length = defaultdict(int)
    for j, pred in enumerate(predictions_flat):
        dual_path = len(pred) == 4
        coors_atom37, residue_type = pred[0], pred[1]
        n = coors_atom37.shape[-3]

        dir_name = f"noise_{noise_scale:.2f}_job_{job_id}_n_{n}_id_{samples_per_length[n]}"
        samples_per_length[n] += 1
        sample_root_path = os.path.join(root_path, dir_name)
        os.makedirs(sample_root_path, exist_ok=False)

        # Save path A
        fname = dir_name + ("_pathA.pdb" if dual_path else ".pdb")
        pdb_path = os.path.join(sample_root_path, fname)
        write_prot_to_pdb(
            prot_pos=coors_atom37.float().detach().cpu().numpy(),
            aatype=residue_type.detach().cpu().numpy(),
            file_path=pdb_path,
            overwrite=True,
            no_indexing=True,
        )

        # Save path B if dual path
        if dual_path:
            coors_atom37_B, residue_type_B = pred[2], pred[3]
            fname_B = dir_name + "_pathB.pdb"
            pdb_path_B = os.path.join(sample_root_path, fname_B)
            write_prot_to_pdb(
                prot_pos=coors_atom37_B.float().detach().cpu().numpy(),
                aatype=residue_type_B.detach().cpu().numpy(),
                file_path=pdb_path_B,
                overwrite=True,
                no_indexing=True,
            )


def main():
    parser = argparse.ArgumentParser(
        description="Baseline Task 1: Unconditional generation with noise perturbation"
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default="inference_ucond_tri",
        help="Name of the config yaml file",
    )
    parser.add_argument(
        "--noise_scale",
        type=float,
        required=True,
        help="Noise perturbation coefficient for path B (x_B = x_A + scale * noise)",
    )
    parser.add_argument(
        "--num_pairs",
        type=int,
        default=50,
        help="Number of pairs to generate",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="baselines/task1_unconditional",
        help="Directory for results (relative to project root)",
    )
    parser.add_argument(
        "--job_id",
        type=int,
        default=0,
        help="Job id for splitting",
    )
    args = parser.parse_args()

    # Setup logging
    logger.add(
        sys.stdout,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {file}:{line} | {message}",
    )

    logger.info(f"Starting baseline Task 1 with noise_scale={args.noise_scale}")
    logger.info(f"Number of pairs: {args.num_pairs}")
    logger.info(f"Output directory: {args.output_dir}")

    # Load config using Hydra - use absolute path
    config_path = os.path.join(PROJECT_ROOT, "configs")
    with hydra.initialize(config_path=config_path, version_base=hydra.__version__):
        cfg = hydra.compose(config_name=args.config_name)

    # Override parameters
    cfg.generation.dataset.nsamples = args.num_pairs
    cfg.generation.dataset.nlens_cfg.nres_lens = [120]
    cfg.generation.args.dual_path_alpha = 1.0  # No path mixing
    cfg.generation.args.mlp_t_threshold = 1.1  # MLP not applied

    # Set seed
    L.seed_everything(cfg.seed + args.job_id)

    # Create output directory
    root_path = os.path.join(PROJECT_ROOT, args.output_dir)
    os.makedirs(root_path, exist_ok=True)

    # Load model
    model = load_model(cfg)

    # Wrap predict_step to inject noise_scale
    wrapper = NoiseScaleWrapper(model, args.noise_scale)
    model.predict_step = wrapper.patched_predict_step

    # Create dataset and dataloader
    dataset = GenDataset(**cfg.generation.dataset)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Run inference using Lightning Trainer
    trainer = L.Trainer(accelerator="gpu", devices=1)
    predictions = trainer.predict(model, dataloader)

    # Save results
    save_baseline_predictions(
        root_path=root_path,
        predictions=predictions,
        noise_scale=args.noise_scale,
        job_id=args.job_id,
    )

    logger.info(f"Generation completed. Results saved in: {root_path}")


if __name__ == "__main__":
    main()