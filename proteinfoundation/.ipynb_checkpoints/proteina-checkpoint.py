import os
import random
from functools import partial
from typing import Dict, List, Literal, Tuple, Union

import lightning as L
import numpy as np
import torch
from jaxtyping import Bool, Float
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from loguru import logger
from torch import Tensor
from omegaconf import OmegaConf
from mlp_model.mlp_mixer import MLP_Mixer

from proteinfoundation.flow_matching.product_space_flow_matcher import (
    ProductSpaceFlowMatcher,
)
from proteinfoundation.nn.local_latents_transformer import LocalLatentsTransformer
from proteinfoundation.nn.local_latents_transformer_unindexed import LocalLatentsTransformerMotifUidx
from proteinfoundation.partial_autoencoder.autoencoder import AutoEncoder
from proteinfoundation.utils.coors_utils import nm_to_ang, trans_nm_to_atom37
from proteinfoundation.utils.pdb_utils import (
    create_full_prot,
    to_pdb,
)

import sys

# Добавляем папку с mlp_mixer.py в путь поиска
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
mlp_path = os.path.join(project_root, "mlp_model_dataset")
if mlp_path not in sys.path:
    sys.path.insert(0, mlp_path)

# Пытаемся импортировать
try:
    from mlp_mixer import MLP_Mixer
    print("✅ MLP_Mixer class imported")
except ImportError:
    MLP_Mixer = None
    print("⚠️ Warning: Could not import MLP_Mixer. Check mlp_model_dataset/mlp_mixer.py")

@rank_zero_only
def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)


class Proteina(L.LightningModule):
    def __init__(self, cfg_exp, store_dir=None, autoencoder_ckpt_path=None, mlp_mixer_ckpt_path=None, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.cfg_exp = cfg_exp
        self.inf_cfg = None  # Only for inference runs
        self.validation_output_lens = {}
        self.validation_output_data = []
        self.store_dir = store_dir if store_dir is not None else "./tmp"
        self.val_path_tmp = os.path.join(self.store_dir, "val_samples")
        create_dir(self.val_path_tmp)

        self.metric_factory = None

        if autoencoder_ckpt_path is not None:
            # Allow adding new keys
            logger.info(f"Manually setting autoencoder_ckpt_path to {autoencoder_ckpt_path}")
            OmegaConf.set_struct(cfg_exp, False)
            # Update the configuration with the new key-value pair
            cfg_exp.autoencoder_ckpt_path = autoencoder_ckpt_path
            # Re-enable struct mode if needed
            OmegaConf.set_struct(cfg_exp, True)

        self.autoencoder, latent_dim = self.load_autoencoder(cfg_exp, freeze_params=True)
        self.autoencoder, latent_dim = self.load_autoencoder(cfg_exp, freeze_params=True)
        
        # 🔹 1. ОПРЕДЕЛЯЕМ ПУТЬ К ЧЕКПОИНТУ (ЭТОЙ СТРОКИ НЕ ХВАТАЛО!)
        # Сначала проверяем аргумент функции, потом конфиг
        target_mlp_path = mlp_mixer_ckpt_path
        if target_mlp_path is None:
            # OmegaConf или dict
            target_mlp_path = getattr(cfg_exp, "mlp_mixer_ckpt_path", None)
            if target_mlp_path is None and isinstance(cfg_exp, dict):
                target_mlp_path = cfg_exp.get("mlp_mixer_ckpt_path")

        # 🔹 2. ЗАГРУЗКА МОДЕЛИ
        # Проверяем, что путь есть И что класс MLP_Mixer доступен (импортирован)
        if target_mlp_path is not None and 'MLP_Mixer' in globals() and MLP_Mixer is not None:
            print(f"🚀 Loading MLP Mixer from: {target_mlp_path}")
            try:
                # Инициализируем архитектуру
                mlp_latent_dim = latent_dim if latent_dim is not None else cfg_exp.product_flowmatcher.local_latents.get("dim", 8)
                self.mlp_mixer = MLP_Mixer(latent_dim=mlp_latent_dim)
                
                # Загружаем чекпоинт
                ckpt = torch.load(target_mlp_path, map_location="cpu", weights_only=False)
                
                # Извлекаем state_dict
                state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
                
                # Фильтруем ключи (убираем префикс 'mlp_mixer.')
                mlp_state = {
                    k.replace('mlp_mixer.', ''): v 
                    for k, v in state_dict.items() 
                    if k.startswith('mlp_mixer.')
                }
                
                # Загружаем веса
                self.mlp_mixer.load_state_dict(mlp_state, strict=False)
                self.mlp_mixer.eval()
                for p in self.mlp_mixer.parameters():
                    p.requires_grad = False
                print("✅ MLP Mixer loaded successfully!")
            except Exception as e:
                print(f"❌ Failed to load MLP Mixer: {e}")
                import traceback
                traceback.print_exc()
                self.mlp_mixer = None
        else:
            self.mlp_mixer = None
            if target_mlp_path is None:
                print("ℹ️ MLP Mixer not loaded (path not provided).")
            else:
                print("ℹ️ MLP Mixer not loaded (class not found).")
        
        # Add right latent dimensionality in the config file, needed to instantiate the flow matcher below
        if latent_dim is not None:
            self.latent_dim = latent_dim
        else:
            self.latent_dim = cfg_exp.product_flowmatcher.local_latents.get("dim", 8)
            
        if self.autoencoder is not None:
            try:
                cfg_exp.product_flowmatcher.local_latents.dim = self.latent_dim
            except:
                OmegaConf.set_struct(cfg_exp, False)
                # Update the configuration with the new key-value pair
                cfg_exp.product_flowmatcher.local_latents.dim = self.latent_dim
                # Re-enable struct mode if needed
                OmegaConf.set_struct(cfg_exp, True)

        self.fm = ProductSpaceFlowMatcher(cfg_exp)
        logger.info(f"cfg_exp.nn: {cfg_exp.nn}")

        # Neural network
        if cfg_exp.nn.name == "local_latents_transformer":
            self.nn = LocalLatentsTransformer(**cfg_exp.nn, latent_dim=self.latent_dim)
        elif cfg_exp.nn.name == "local_latents_transformer_motif_uidx":
            self.nn = LocalLatentsTransformerMotifUidx(**cfg_exp.nn, latent_dim=self.latent_dim)
        else:
            raise IOError(f"Wrong nn selected for CAFlow {cfg_exp.nn.name}")

        # Scaling laws stuff
        self.nflops = 0
        self.nsamples_processed = 0
        self.nparams = sum(p.numel() for p in self.nn.parameters() if p.requires_grad)

        self.nn_ag = None

    def load_autoencoder(self, cfg_exp, freeze_params=True):
        """Loads autoencoder, if required."""
        if ("autoencoder_ckpt_path" in cfg_exp):
            # for new runs trained with refactored codebase
            ae_ckp_path = cfg_exp.autoencoder_ckpt_path
        elif ("autoencoder_ckpt_path" in cfg_exp.product_flowmatcher.local_latents):
            # for old runs trained with old codebase
            ae_ckp_path = cfg_exp.product_flowmatcher.local_latents.autoencoder_ckpt_path
        else:
            raise ValueError("No autoencoder checkpoint path provided")

        if ae_ckp_path is None:
            return None, None
        
        logger.info(f"Loading autoencoder from {ae_ckp_path}")
        autoencoder = AutoEncoder.load_from_checkpoint(ae_ckp_path, strict=False)
        if freeze_params:
            for param in autoencoder.parameters():
                param.requires_grad = False
        return autoencoder, autoencoder.latent_dim

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            [p for p in self.parameters() if p.requires_grad], lr=self.cfg_exp.opt.lr
        )
        return optimizer

    def on_save_checkpoint(self, checkpoint):
        """Adds additional variables to checkpoint."""
        checkpoint["nflops"] = self.nflops
        checkpoint["nsamples_processed"] = self.nsamples_processed

    def on_load_checkpoint(self, checkpoint):
        """Loads additional variables from checkpoint."""
        try:
            self.nflops = checkpoint["nflops"]
            self.nsamples_processed = checkpoint["nsamples_processed"]
        except:
            logger.info("Failed to load nflops and nsamples_processed from checkpoint")
            self.nflops = 0
            self.nsamples_processed = 0

    def call_nn(
        self,
        batch: Dict[str, torch.Tensor],
        n_recycle: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Calls NN with recycling. Should this be here or in the NN? Possibly better here,
        in case we want to recycle using decoder for some approach, etc, and this is akin
        to self conditioning, also here.
        Also, if we want to recycle clean sample predictions... Then we'd need this here,
        as the nn does not know about relations between v, x1, ...
        """
        # First call
        nn_out = self.nn(batch)

        # Recycle n_recycle times detaching gradients and updating input
        # Note that recycling is supported by the codebase, but the models provided 
        # with the La-Proteina paper do not use it, nor were trained with it.
        for _ in range(n_recycle):
            x_1_pred = self.fm.nn_out_to_clean_sample_prediction(
                batch=batch, nn_out=nn_out
            )
            batch[f"x_recycle"] = {dm: x_1_pred[dm].detach() for dm in x_1_pred}
            nn_out = self.nn(batch)

        # Final prediction
        return nn_out

    def predict_for_sampling(
        self,
        batch: Dict,
        mode: Literal["full", "ucond"],
        n_recycle: int,
    ) -> Tuple[Union[Dict[str, torch.Tensor], float, None]]:
        """
        This function predicts clean samples for multiple models:
        x_pred, the 'original' model, if mode == full
        x_pred_ucond, the unconditional model, , if mode == ucond

        TODO: Need to update to include autoguidance again

        These predictions will later be used to sample with guidance and autoguidance.

        Args:
            batch: Dict
            mode: str

        Returns:
            x_pred (tensor) for the requested mode
        """
        if mode == "full":
            nn_out = self.call_nn(batch, n_recycle=n_recycle)
        elif mode == "ucond":
            assert "cath_code" in batch or "x_motif" in batch, "Only support CFG when cath_code or x_motif is provided"
            uncond_batch = batch.copy()
            if "cath_code" in uncond_batch:
                uncond_batch.pop("cath_code")
            if "x_motif" in uncond_batch:
                uncond_batch.pop("x_motif")
            nn_out = self.call_nn(uncond_batch, n_recycle=n_recycle)
        else:
            raise IOError(f"Wrong {mode} passed to `predict_for_sampling`")

        return nn_out

    def training_step(self, batch: Dict, batch_idx: int):
        """
        Computes training loss for batch of samples.

        Args:
            batch: Data batch.

        Returns:
            Training loss averaged over batch dimension.
        """
        val_step = batch_idx == -1  # validation step is indicated with batch_idx -1
        log_prefix = "validation_loss" if val_step else "train"

        # Add clean samples for all data modes / spaces we are working on
        batch = self.add_clean_samples(batch)

        # Corrupt the batch
        batch = self.fm.corrupt_batch(batch)  # adds x_1, t, x_0, x_t, mask
        bs, n = batch["mask"].shape

        # Handle conditioning variables
        batch = self.handle_self_cond(
            batch
        )  # self conditioning, adds ["x_sc"] to batch prob 0.5
        batch = self.handle_folding_n_inverse_folding(
            batch
        )  # folding and inverse folding iterations

        # Number of recycling steps
        n_recycle = self.handle_recycling()

        nn_out = self.call_nn(batch, n_recycle=n_recycle)
        losses = self.fm.compute_loss(
            batch=batch,
            nn_out=nn_out,
        )  # Dict[str, Tensor w.batch shape [*]]

        self.log_losses(bs=bs, losses=losses, log_prefix=log_prefix, batch=batch)
        train_loss = sum([torch.mean(losses[k]) for k in losses if "_justlog" not in k])

        self.log(
            f"{log_prefix}/loss",
            train_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=bs,
            sync_dist=True,
            add_dataloader_idx=False,
        )

        if not val_step:  # Don't log these for val step
            self.log_train_loss_n_prog_bar(bs, train_loss)
            self.update_n_log_flops(bs, n)
            self.update_n_log_nsamples_processed(bs)
            self.log_nparams()

        return train_loss

    def add_clean_samples(self, batch: Dict) -> Dict:
        """
        Adds clean sample for all data modes / spaces we are working on. For instance, if we have two
        data modes, bb_ca and local_latents, it adds the clean data to the batch
        x_1 = {
            "bb_ca": Corresponding tensor with clean bb_ca coordinates, shape [b, n, 3]
            "local_latents": Corresponding tensor with clean local_latents, shape [b, n, d]
        }

        Args:
            batch: Batch to add clean samples to.

        Returns:
            Batch with clean sample added.
        """
        batch["x_1"] = {
            dm: self._get_clean_sample(batch, dm)
            for dm in self.cfg_exp.product_flowmatcher
        }
        return batch

    def _get_clean_sample(self, batch: Dict, dm: str) -> torch.Tensor:
        """
        Gets clean sample for a given data mode.

        Args:
            batch: Batch to get clean sample from.
            dm: Data mode to get clean sample for.

        Returns:
            Clean sample for the given data mode.
        """
        if dm == "bb_ca":
            return batch["coords_nm"][:, :, 1, :]  # [b, n, 3]
        elif dm == "local_latents":
            encoded_batch = self.autoencoder.encode(batch)
            # {
            #   "z_latent": latent_sample, shape [b, n, d]
            #   "mean": mean of latent (diag) Gaussian dist, shape [b, n, d]
            #   "log_scale": log standard deviation of latent (diag) Gaussian dist, shape [b, n, d]
            # }
            return encoded_batch["z_latent"]
        else:
            raise ValueError(
                f"Loading clean samples from data mode {dm} not supported."
            )


    def handle_self_cond(self, batch: Dict) -> Dict:
        n_recycle = self.cfg_exp.training.get(
            "n_recycle", 0
        )
        if random.random() > 0.5 and self.cfg_exp.training.self_cond:
            nn_out = self.call_nn(batch, n_recycle=n_recycle)
            x_1_pred = self.fm.nn_out_to_clean_sample_prediction(
                batch=batch, nn_out=nn_out
            )
            batch["x_sc"] = {k: x_1_pred[k].detach() for k in x_1_pred}

        return batch

    def handle_recycling(self):
        n_recycle = self.cfg_exp.training.get("n_recycle", 0)
        if n_recycle == 0:
            return 0
        return random.randint(0, n_recycle)  # 0 and n_recycle included

    def handle_folding_n_inverse_folding(self, batch: Dict) -> Dict:
        """
        With 15% probability either a folding or inverse folding iteration.
        If one such iteration (ie 15% of the times), with 50% probability set
        set folding_mode to true, otherwise set inverse_folding_mode to true.

        For inverse folding, we just provide CA.

        Applies to the whole batch.

        With 85% probability sets both to false.

        Adds entries 'folding_mode' and 'inverse_folding_ca_mode' to batch, with
        values being boolean variables (True or False).
        """
        batch["use_ca_coors_nm_feature"] = False
        batch["use_residue_type_feature"] = False
        prob = self.cfg_exp.training.get("p_folding_n_inv_folding_iters", 0.0)
        r1 = random.random()  # float
        if r1 < prob:  # with p=prob
            r2 = random.random()
            if r2 < 0.5:  # with p=0.5
                batch["use_ca_coors_nm_feature"] = True
            else:
                batch["use_residue_type_feature"] = True
        return batch

    def log_losses(
        self,
        bs: int,
        losses: Dict[str, Float[torch.Tensor, "b"]],
        log_prefix: str,
        batch: Dict,
    ):
        for k in losses:
            log_name = k[: -len("_justlog")] if k.endswith("_justlog") else k

            self.log(
                f"{log_prefix}/loss_{log_name}",
                torch.mean(losses[k]),
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                batch_size=bs,
                sync_dist=True,
                add_dataloader_idx=False,
            )

            if self.cfg_exp.training.get("p_folding_n_inv_folding_iters", 0.0) > 0.0:
                # Log also for folding and inverse folding iters
                # divides by p_aux to account for the fact that for most steps loss will be just zero
                p_aux = self.cfg_exp.training["p_folding_n_inv_folding_iters"] / 2
                loss = torch.mean(losses[k])  # [b]

                f_inv_fold = batch["use_ca_coors_nm_feature"] * 1.0 / p_aux
                self.log(
                    f"{log_prefix}_invfold_ca_iter/loss_{log_name}",
                    loss * f_inv_fold,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                    batch_size=bs,
                    sync_dist=True,
                    add_dataloader_idx=False,
                )

                f_fold = batch["use_residue_type_feature"] * 1.0 / p_aux
                self.log(
                    f"{log_prefix}_fold_iter/loss_{log_name}",
                    loss * f_fold,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                    batch_size=bs,
                    sync_dist=True,
                    add_dataloader_idx=False,
                )

    def log_train_loss_n_prog_bar(self, b: int, train_loss: torch.Tensor):
        self.log(
            f"train_loss",
            train_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=b,
            sync_dist=True,
            add_dataloader_idx=False,
        )

    def log_nparams(self):
        self.log(
            "scaling/nparams",
            self.nparams * 1.0,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            batch_size=1,
            sync_dist=True,
        )  # constant line

    def update_n_log_nsamples_processed(self, b: int):
        self.nsamples_processed = self.nsamples_processed + b * self.trainer.world_size
        self.log(
            "scaling/nsamples_processed",
            self.nsamples_processed * 1.0,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            batch_size=1,
            sync_dist=True,
        )

    def update_n_log_flops(self, b: int, n: int):
        """
        Updates and logs flops, if available
        """
        try:
            nflops_step = self.nn.nflops_computer(
                b, n
            )  # nn should implement this function if we want to see nflops
        except:
            nflops_step = None

        if nflops_step is not None:
            self.nflops = (
                self.nflops + nflops_step * self.trainer.world_size
            )  # Times number of processes so it logs sum across devices
            self.log(
                "scaling/nflops",
                self.nflops * 1.0,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
                batch_size=1,
                sync_dist=True,
            )

    def validation_step(self, batch: Dict, batch_idx: int):
        """
        Validation step.

        Args:
            batch: batch from dataset (see last argument)
            batch_idx: batch index (unused)
        """
        self.validation_step_data(batch, batch_idx)

    def validation_step_data(self, batch: Dict, batch_idx: int):
        """Evaluates the training loss on validation data."""
        with torch.no_grad():
            loss = self.training_step(batch, batch_idx=-1)
            self.validation_output_data.append(loss.item())

    def on_validation_epoch_end(self):
        """
        Takes the samples produced in the validation step, stores them as pdb files, and computes validation metrics.
        It also cleans results.
        """
        self.on_validation_epoch_end_data()

    def on_validation_epoch_end_data(self):
        self.validation_output_data = []

    def configure_inference(self, inf_cfg, nn_ag, mlp_mixer=None):
        """Sets inference config with all sampling parameters required by the method (dt, etc)
        and autoguidance network (or None if not provided)."""
        self.inf_cfg = inf_cfg
        self.nn_ag = nn_ag
        if mlp_mixer is not None:
            self.mlp_mixer = mlp_mixer  # ← добавляем MLP

    # def predict_step(self, batch: Dict, batch_idx: int) -> List[Tuple[torch.tensor]]:
    #     """
    #     Генерация через dual_path + MLP миксинг.
    #     Возвращает две структуры с идентичной последовательностью.
    #     """
    #     self_cond = self.inf_cfg.args.self_cond
    #     nsteps = self.inf_cfg.args.nsteps
    #     guidance_w = self.inf_cfg.args.get("guidance_w", 1.0)
    #     ag_ratio = self.inf_cfg.args.get("ag_ratio", 0.0)
    #     save_trajectory_every = 0
        
    #     fn_predict_for_sampling = partial(
    #         self.predict_for_sampling, n_recycle=self.inf_cfg.get("n_recycle", 0)
    #     )

    #     # 1. Два независимых пути через Flow Matching
    #     gen_samples, extra_info = self.fm.full_simulation(
    #         batch=batch,
    #         predict_for_sampling=fn_predict_for_sampling,
    #         nsteps=nsteps,
    #         nsamples=batch["nsamples"],
    #         n=batch["nres"],
    #         self_cond=self_cond,
    #         sampling_model_args=self.inf_cfg.model,
    #         device=self.device,
    #         save_trajectory_every=save_trajectory_every,
    #         guidance_w=guidance_w,
    #         ag_ratio=ag_ratio,
    #         dual_path_alpha=0.9, # 0.95
    #         init_noise_scale=0.0,
    #         mlp_mixer= getattr(self, "mlp_mixer", None),
    #         mlp_t_threshold=0.95, # 0.95
    #     )
        
    #     z_A, ca_A = gen_samples["local_latents"], gen_samples["bb_ca"]
    #     z_B, ca_B = extra_info["x_B"]["local_latents"], extra_info["x_B"]["bb_ca"]
    #     mask = extra_info["mask"]

    #     # 2. MLP миксинг (если загружен)
    #     if self.mlp_mixer is not None:
    #         with torch.no_grad():
    #             z_cons = self.mlp_mixer(z_A - z_B, z_A, z_B)
    #     else:
    #         z_cons = (z_A + z_B) / 2.0 

    #     # 3. Декодирование с общим латентом, разными координатами
    #     from proteinfoundation.utils.coors_utils import nm_to_ang
    #     out_A = self.autoencoder.decode(z_latent=z_cons, ca_coors_nm=ca_A, mask=mask)
    #     out_B = self.autoencoder.decode(z_latent=z_cons, ca_coors_nm=ca_B, mask=mask)

    #     # 4. Форматирование вывода: две конформации, одна последовательность
    #     gen_list = []
    #     for i in range(batch["nsamples"]):
    #         gen_list.append((
    #             nm_to_ang(out_A["coors_nm"][i]), out_A["residue_type"][i],
    #             nm_to_ang(out_B["coors_nm"][i]), out_B["residue_type"][i]
    #         ))
    #     return gen_list

    #     # 🔹 3. Fallback: если MLP нет, стандартный вывод
    #     sample_prots = self.sample_formatting(x=gen_samples, extra_info=extra_info, ret_mode="coors37_n_aatype")
    #     return [(sample_prots["coors"][i], sample_prots["residue_type"][i]) 
    #             for i in range(sample_prots["coors"].shape[0])]
        
    #     # ---------------------------------------------------------
    #     # Если MLP нет, возвращаем стандартный результат
    #     # ---------------------------------------------------------
    #     sample_prots = self.sample_formatting(
    #         x=gen_samples,
    #         extra_info=extra_info,
    #         ret_mode="coors37_n_aatype",
    #     )
        
    #     generation_list = []
    #     if dual_path_alpha > 0.0 and extra_info.get("x_B") is not None:
    #         sample_prots_B = self.sample_formatting(
    #             x=extra_info["x_B"],
    #             extra_info=extra_info,
    #             ret_mode="coors37_n_aatype",
    #         )
    #         for i in range(sample_prots["coors"].shape[0]):
    #             generation_list.append((
    #                 sample_prots["coors"][i],
    #                 sample_prots["residue_type"][i],
    #                 sample_prots_B["coors"][i],
    #                 sample_prots_B["residue_type"][i],
    #             ))
    #     else:
    #         for i in range(sample_prots["coors"].shape[0]):
    #             generation_list.append(
    #                 (sample_prots["coors"][i], sample_prots["residue_type"][i])
    #             )
        
    #     return generation_list  # List of tupes (coors [n, 37, 3], aatype [n])
    def predict_step(self, batch: Dict, batch_idx: int) -> List[Tuple[torch.tensor]]:
        self_cond       = self.inf_cfg.args.self_cond
        nsteps          = self.inf_cfg.args.nsteps
        guidance_w      = self.inf_cfg.args.get("guidance_w", 1.0)
        ag_ratio        = self.inf_cfg.args.get("ag_ratio",   0.0)
        dual_path_alpha = self.inf_cfg.args.get("dual_path_alpha", 0.0)
        
        # Порог t, начиная с которого MLP применяется в цикле.
        # Если не задан в конфиге — ставим 1.1, т.е. MLP только после цикла (старое поведение).
        mlp_t_threshold = self.inf_cfg.args.get("mlp_t_threshold", 1.1)
        
        fn_predict_for_sampling = partial(
            self.predict_for_sampling, n_recycle=self.inf_cfg.get("n_recycle", 0)
        )
    
        mlp_mixer = getattr(self, "mlp_mixer", None)
    
        gen_samples, extra_info = self.fm.full_simulation(
            batch=batch,
            predict_for_sampling=fn_predict_for_sampling,
            nsteps=nsteps,
            nsamples=batch["nsamples"],
            n=batch["nres"],
            self_cond=self_cond,
            sampling_model_args=self.inf_cfg.model,
            device=self.device,
            save_trajectory_every=0,
            guidance_w=guidance_w,
            ag_ratio=ag_ratio,
            dual_path_alpha=dual_path_alpha,
            init_noise_scale=0.0,
            mlp_mixer=mlp_mixer,           # <-- передаём MLP
            mlp_t_threshold=mlp_t_threshold,  # <-- передаём порог
        )
    
        mask          = extra_info["mask"]
        x_B           = extra_info.get("x_B")
        scaffold_mask = extra_info.get("scaffold_mask")
    
        # Если dual path активен
        if dual_path_alpha > 0.0 and x_B is not None:
            z_A  = gen_samples["local_latents"]
            ca_A = gen_samples["bb_ca"]
            z_B  = x_B["local_latents"]
            ca_B = x_B["bb_ca"]
    
            with torch.no_grad():
                # Если MLP уже применялся в цикле (mlp_t_threshold <= 1.0),
                # то z_A == z_B == z_cons — финальное смешивание необязательно,
                # но для надёжности делаем его снова.
                if mlp_mixer is not None:
                    z_cons = mlp_mixer(z_A - z_B, z_A, z_B)
                else:
                    if scaffold_mask is not None:
                        sm = scaffold_mask[..., None]
                        z_avg = dual_path_alpha * z_A + (1.0 - dual_path_alpha) * z_B
                        z_cons = torch.where(sm, z_avg, z_A)
                    else:
                        z_cons = dual_path_alpha * z_A + (1.0 - dual_path_alpha) * z_B
    
            sample_prots_A = self.sample_formatting(
                x={"local_latents": z_cons, "bb_ca": ca_A},
                extra_info=extra_info,
                ret_mode="coors37_n_aatype",
            )
            sample_prots_B = self.sample_formatting(
                x={"local_latents": z_cons, "bb_ca": ca_B},
                extra_info=extra_info,
                ret_mode="coors37_n_aatype",
            )
    
            return [
                (
                    sample_prots_A["coors"][i], sample_prots_A["residue_type"][i],
                    sample_prots_B["coors"][i], sample_prots_B["residue_type"][i],
                )
                for i in range(sample_prots_A["coors"].shape[0])
            ]
    
        # Дефолт: один путь
        sample_prots = self.sample_formatting(
            x=gen_samples, extra_info=extra_info, ret_mode="coors37_n_aatype"
        )
        return [
            (sample_prots["coors"][i], sample_prots["residue_type"][i])
            for i in range(sample_prots["coors"].shape[0])
        ]

        
    def sample_formatting(
        self,
        x: Dict[str, Tensor],
        extra_info: Dict[str, Tensor],
        ret_mode: str,
    ):
        """
        Given a batch of b samples x produced by the flow matcher, it returns the samples in the requested format (ret_mode).

        Supports `ret_modes` for:
            - `samples` returns the original sample from the flow matcher, a dictionary[str, Tensor].
            for the data modalities, each with batch shape b.
            - `atom37` returns an Tensor of shape [b, n, 37, 3] just for coordinates.
            - `pdb_string` returns a list of dictionaries {"pdb_string": str, "nres": int}, with one dictionary per sample.
            - `coors37_n_aatype` returns a dictionary with keys `coors` (atom37), `residue_type`, and `mask`, and
            values with shapes [b, n, 37, 3] float, [b, n] int, [b, n] boolean, respectively.

        Args:
            x: sample.
            extra_info: a dict with additional things, including:
                - "mask", whose value is boolean of shape [nsamples, n]
                - ...
            ret_mode: target format, for now only supports atom37.

        Returns:
            Sample x in the requested format.
        """
        data_modes = sorted([dm for dm in self.cfg_exp.product_flowmatcher])
        if data_modes == ["bb_ca"]:
            return self._format_sample_bb_ca(
                x=x, ret_mode=ret_mode, mask=extra_info["mask"]
            )
        elif data_modes == ["bb_ca", "local_latents"]:
            return self._format_sample_local_latents(
                x=x, ret_mode=ret_mode, mask=extra_info["mask"]
            )
        else:
            raise NotImplementedError(f"Format {ret_mode} not implemented")

    def _format_sample_bb_ca(
        self,
        x: Dict[str, torch.Tensor],
        ret_mode: str,
        mask: Bool[torch.Tensor, "b n"],
    ):
        if ret_mode == "samples":
            return x

        if ret_mode == "atom37":
            return trans_nm_to_atom37(x["bb_ca"].float())

        elif ret_mode == "coors37_n_aatype":
            coors = (
                trans_nm_to_atom37(x["bb_ca"].float()) * mask[..., None, None]
            )  # [b, n, 37, 3]
            residue_type = torch.zeros_like(coors)[..., 0, 0] * mask  # [b, n]
            return {
                "coors": coors,  # [b, n, 37, 3]
                "residue_type": residue_type.long(),  # [b, n]
                "mask": mask,  # [b, n]
            }

        elif ret_mode == "pdb_string":
            pdb_strings = []

            coors = (
                trans_nm_to_atom37(x["bb_ca"]).float().detach().cpu().numpy()
            )  # [b, n, 37, 3]
            residue_type = np.zeros_like(coors[:, :, 0, 0])  # [b, n]
            atom37_mask = np.zeros_like(coors[:, :, :, 0])  # [b, n, 37]
            atom37_mask[:, :, 1] = 1.0  # [b, n, 37]
            atom37_mask = atom37_mask * mask[..., None]  # [b, n, 37]
            n = coors.shape[-3]

            for i in range(coors.shape[0]):
                prot = create_full_prot(
                    atom37=coors[i, ...],
                    atom37_mask=atom37_mask[i, ...],
                    aatype=residue_type[i, ...],
                )
                pdb_string = to_pdb(prot=prot)
                pdb_strings.append(
                    {
                        "pdb_string": pdb_string,
                        "nres": n,
                    }
                )
            return pdb_strings

        else:
            raise NotImplementedError(
                f"{ret_mode} format for data modes `[bb_ca]` not implemented"
            )

    def _format_sample_local_latents(
        self,
        x: Dict[str, torch.Tensor],
        ret_mode: str,
        mask: Bool[torch.Tensor, "b n"],
    ):
        """
        Given a batch of b samples consisting on `bb_ca` and `local_latents` this
        returns formatted samples.

        Note: This calls the decoder from the autoencoder, since it needs to go from
        local latent variables to the actual coordinates and sequence.

        Note: The self.autoencoder.decode function (used here) returns a dictoinary like
        {
            "coors_nm": [b, n, 37, 3], already masked
            "residue_type": [b, n], already masked, careful with 0s
            "residue_mask": [b, n]
            "atom_mask": [b, n, 37]
        }

        Args:
            x: sample.
            extra_info: a dict with additional things, including:
                - "mask", whose value is boolean of shape [nsamples, n]
                - ...
            ret_mode: target format, for now only supports atom37.

        Returns:
            Sample x in the requested format.
        """
        output_decoder = self.autoencoder.decode(
            z_latent=x["local_latents"], ca_coors_nm=x["bb_ca"], mask=mask
        )

        if ret_mode == "samples":
            return x

        elif ret_mode == "coors37_n_aatype":
            return {
                "coors": nm_to_ang(output_decoder["coors_nm"]),  # [b, n, 37, 3]
                "residue_type": output_decoder["residue_type"],  # [b, n]
                "mask": output_decoder["residue_mask"],  # [b, n]
            }

        elif ret_mode == "pdb_string":
            pdb_strings = []

            coors_atom_37 = (
                nm_to_ang(output_decoder["coors_nm"]).float().detach().cpu().numpy(),
            )  # [b, n, 37, 3]
            residue_type = output_decoder["residue_type"]  # [b, n]
            atom_mask = output_decoder["atom_mask"]  # [b, n, 37]
            n = coors_atom_37.shape[-3]

            for i in range(atom_mask.shape[0]):
                prot = create_full_prot(
                    atom37=coors_atom_37[i, ...],
                    atom37_mask=atom_mask[i, ...],
                    aatype=residue_type[i, ...],
                )
                pdb_string = to_pdb(prot=prot)
                pdb_strings.append(
                    {
                        "pdb_string": pdb_string,
                        "nres": n,
                    }
                )
            return pdb_strings

        else:
            raise NotImplementedError(
                f"{ret_mode} format for data modes `[bb_ca, latent_locals]` not implemented"
            )

