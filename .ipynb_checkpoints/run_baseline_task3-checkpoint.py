"""
Baseline Task 3: Conditional generation with LARGE noise neighborhoods.
Generates pairs of structures with different noise_scale for path B, using motif scaffolding.
Uses original La-Proteina pipeline without MLP and without path mixing.

Usage (from la-proteina-main/ directory):
    DATA_PATH=./data python run_baseline_task3.py --noise_scale 5.0 --num_pairs 50
"""

import os
import sys

# Add mlp_model to path (needed by proteina.py)
PROJECT_ROOT = os.getcwd()
MLP_MODEL_PATH = os.path.join(PROJECT_ROOT, "mlp_model")
if os.path.exists(MLP_MODEL_PATH):
    sys.path.insert(0, PROJECT_ROOT)
    import importlib.util
    import types
    mlp_dataset_pkg = types.ModuleType('mlp_model_dataset')
    mlp_dataset_pkg.__path__ = [MLP_MODEL_PATH]
    sys.modules['mlp_model_dataset'] = mlp_dataset_pkg
    
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
from typing import List, Tuple

import numpy as np
import torch
import lightning as L
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from loguru import logger

from proteinfoundation.datasets.gen_dataset import GenDataset
from proteinfoundation.proteina import Proteina
from proteinfoundation.utils.pdb_utils import write_prot_to_pdb


CONFIG_DIR = os.path.join(PROJECT_ROOT, "configs")


def load_model(cfg):
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

    model.configure_inference(
        inf_cfg=cfg.generation,
        nn_ag=None,
        mlp_mixer=None,
    )

    logger.info("MLP disabled for baseline experiment")
    return model


def find_flow_matcher(model):
    """Find the flow matcher attribute dynamically."""
    for attr_name in ['flow_matcher', 'fm', 'product_flow_matcher']:
        if hasattr(model, attr_name):
            attr = getattr(model, attr_name)
            if hasattr(attr, 'full_simulation'):
                return attr
    for attr_name in dir(model):
        attr = getattr(model, attr_name, None)
        if attr is not None and hasattr(attr, 'full_simulation'):
            return attr
    raise AttributeError("Could not find flow matcher with full_simulation method")


def patch_full_simulation(model, noise_scale):
    """
    Monkey-patch full_simulation to inject init_noise_scale and disable MLP/mixing.
    """
    fm = find_flow_matcher(model)
    logger.info(f"Found flow matcher: {type(fm).__name__}")
    original_full_sim = fm.full_simulation

    def patched_full_sim(batch, *args, **kwargs):
        kwargs["init_noise_scale"] = noise_scale
        kwargs["dual_path_alpha"] = 1.0
        kwargs["mlp_mixer"] = None
        kwargs["mlp_t_threshold"] = 1.1
        return original_full_sim(batch, *args, **kwargs)

    fm.full_simulation = patched_full_sim


def save_baseline_predictions(
    root_path,
    predictions,
    noise_scale,
    job_id=0,
):
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

        fname = dir_name + ("_pathA.pdb" if dual_path else ".pdb")
        pdb_path = os.path.join(sample_root_path, fname)
        write_prot_to_pdb(
            prot_pos=coors_atom37.float().detach().cpu().numpy(),
            aatype=residue_type.detach().cpu().numpy(),
            file_path=pdb_path,
            overwrite=True,
            no_indexing=True,
        )

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
        description="Baseline Task 3: Conditional generation with LARGE noise perturbation"
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default="inference_motif",
        help="Name of the config yaml file",
    )
    parser.add_argument(
        "--noise_scale",
        type=float,
        required=True,
        help="Noise perturbation coefficient for path B (LARGE values)",
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
        default="baselines/task3_conditional_large_sim_eps",  # <-- ИЗМЕНЕНО
        help="Directory for results",
    )
    parser.add_argument(
        "--motif_name",
        type=str,
        required=True,
        help="Name of the motif (will be added to motif_dict_cfg)",
    )
    parser.add_argument(
        "--motif_pdb",
        type=str,
        required=True,
        help="Path to motif PDB file",
    )
    parser.add_argument(
        "--contig_string",
        type=str,
        required=True,
        help="Contig string specifying the motif residues",
    )
    parser.add_argument(
        "--motif_min_length",
        type=int,
        default=158,
        help="Minimum total length (motif + scaffold)",
    )
    parser.add_argument(
        "--motif_max_length",
        type=int,
        default=158,
        help="Maximum total length (motif + scaffold)",
    )
    parser.add_argument(
        "--job_id",
        type=int,
        default=0,
        help="Job id for splitting",
    )
    args = parser.parse_args()

    logger.add(
        sys.stdout,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {file}:{line} | {message}",
    )

    logger.info(f"Project root: {PROJECT_ROOT}")
    logger.info(f"Config dir: {CONFIG_DIR}")
    logger.info(f"Starting baseline Task 3 with noise_scale={args.noise_scale}")
    logger.info(f"Motif name: {args.motif_name}")
    logger.info(f"Motif PDB: {args.motif_pdb}")
    logger.info(f"Contig string: {args.contig_string}")
    logger.info(f"Motif length range: [{args.motif_min_length}, {args.motif_max_length}]")
    logger.info(f"Number of pairs: {args.num_pairs}")
    logger.info(f"Output directory: {args.output_dir}")

    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name=args.config_name)

    OmegaConf.set_struct(cfg, False)

    cfg.generation.dataset.nsamples = args.num_pairs
    cfg.generation.args.dual_path_alpha = 1.0
    cfg.generation.args.mlp_t_threshold = 1.1
    cfg.generation.dataset.motif_task_name = args.motif_name

    cfg.generation.dataset.motif_dict_cfg[args.motif_name] = {
        "contig_string": args.contig_string,
        "motif_pdb_path": os.path.abspath(args.motif_pdb),
        "motif_only": False,
        "motif_min_length": args.motif_min_length,
        "motif_max_length": args.motif_max_length,
        "segment_order": "A",
        "atom_selection_mode": "all_atom",
    }

    OmegaConf.set_struct(cfg, True)

    L.seed_everything(cfg.seed + args.job_id)

    output_path = os.path.join(PROJECT_ROOT, args.output_dir)
    os.makedirs(output_path, exist_ok=True)

    model = load_model(cfg)
    patch_full_simulation(model, args.noise_scale)

    dataset = GenDataset(**cfg.generation.dataset)

    if getattr(dataset, "motif_task_name", None) is not None and dataset.motif_masks[0] is not None:
        logger.info(f"SUCCESS: motif-conditioned sampling active for '{dataset.motif_task_name}', mask shape {dataset.motif_masks[0].shape}")
    else:
        logger.warning("WARNING: motif conditioning NOT active — falling back to length-centric sampling!")

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    trainer = L.Trainer(accelerator="gpu", devices=1)
    predictions = trainer.predict(model, dataloader)

    save_baseline_predictions(
        root_path=output_path,
        predictions=predictions,
        noise_scale=args.noise_scale,
        job_id=args.job_id,
    )

    logger.info(f"Generation completed. Results saved in: {output_path}")


if __name__ == "__main__":
    main()