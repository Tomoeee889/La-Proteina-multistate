# mlp_trainer.py — версия с Flow Matching + Fix v_norm + Dual Path Mixing + Расширенные метрики

import torch
import torch.nn.functional as F
import lightning as L
from torch import Tensor
from typing import Dict, Optional
from mlp_mixer import MLP_Mixer

NON_CA = [0, 2, 3, 4] + list(range(5, 37))


def build_backbone_frames(coors_nm: Tensor):
    N_a  = coors_nm[:, :, 0, :]
    CA_a = coors_nm[:, :, 1, :]
    C_a  = coors_nm[:, :, 2, :]

    v1 = F.normalize(N_a - CA_a, dim=-1)
    v2 = C_a - CA_a
    v2 = v2 - (v2 * v1).sum(-1, keepdim=True) * v1
    v2 = F.normalize(v2, dim=-1)
    v3 = torch.cross(v1, v2, dim=-1)

    R = torch.stack([v1, v2, v3], dim=-1)
    t = CA_a
    return R, t


def fafe_loss(pred_coors_nm, true_coors_nm, mask_res, atom_mask, eps=1e-7):
    R_pred, t_pred = build_backbone_frames(pred_coors_nm)
    R_gt, t_gt = build_backbone_frames(true_coors_nm)

    R_diff = R_pred.transpose(-1, -2) @ R_gt
    trace = R_diff.diagonal(dim1=-2, dim2=-1).sum(-1)
    cos_a = torch.clamp((trace - 1) / 2, -1 + eps, 1 - eps)
    so3_dist = torch.acos(cos_a)

    r3_dist = torch.norm(t_pred - t_gt, dim=-1) / 1.0
    se3_dist = torch.sqrt(so3_dist ** 2 + r3_dist ** 2 + eps)

    bb_valid = (atom_mask[:, :, 0] * atom_mask[:, :, 1] * atom_mask[:, :, 2]).float() * mask_res.float()
    loss = (se3_dist * bb_valid).sum() / (bb_valid.sum() + 1e-8)
    return loss

class MLP_Trainer(L.LightningModule):
    def __init__(self, autoencoder, flow_matcher, latent_dim=8, dual_path_alpha=0.9):
        super().__init__()
        self.autoencoder = autoencoder
        self.flow_matcher = flow_matcher
        self.latent_dim = latent_dim
        self.dual_path_alpha = dual_path_alpha
        self.mlp_mixer = MLP_Mixer(latent_dim=latent_dim)
        self.register_buffer("blosum62", self._create_blosum62_matrix())

        for p in self.autoencoder.parameters():
            p.requires_grad = False
        for p in self.flow_matcher.parameters():
            p.requires_grad = False

        freqs = torch.tensor([0.0723, 0.0517, 0.0407, 0.0601, 0.0247, 0.0430, 0.0761, 0.0785, 0.0284, 0.0474,
                              0.0770, 0.0689, 0.0240, 0.0353, 0.0464, 0.0715, 0.0545, 0.0109, 0.0298, 0.0589])
        self.class_weights = (1.0 / (freqs + 1e-6))
        self.class_weights = self.class_weights / self.class_weights.mean()
        
        # 🔹 BLOSUM62 матрица (упрощённая версия)
        self.blosum62 = self._create_blosum62_matrix().to(self.device)

    def _create_blosum62_matrix(self) -> Tensor:
        """
        Создаёт упрощённую BLOSUM62 матрицу для 20 аминокислот.
        Возвращает Tensor [20, 20] с нормализованными значениями.
        """
        # Инициализация нулевой матрицы
        blosum = torch.zeros(20, 20)
        
        # Диагональ (точные совпадения) = 1.0
        for i in range(20):
            blosum[i, i] = 1.0
        
        # Похожие аминокислоты (группы по химическим свойствам)
        # Индексы: A:0, R:1, N:2, D:3, C:4, Q:5, E:6, G:7, H:8, I:9,
        #          L:10, K:11, M:12, F:13, P:14, S:15, T:16, W:17, Y:18, V:19
        
        # Гидрофобные: I, L, V, M, F
        hydrophobic = [9, 10, 19, 12, 13]
        for i in hydrophobic:
            for j in hydrophobic:
                if i != j:
                    blosum[i, j] = 0.3
        
        # Полярные положительно заряженные: R, K, H
        positive = [1, 11, 8]
        for i in positive:
            for j in positive:
                if i != j:
                    blosum[i, j] = 0.4
        
        # Полярные отрицательно заряженные: D, E
        negative = [3, 6]
        for i in negative:
            for j in negative:
                if i != j:
                    blosum[i, j] = 0.5
        
        # Маленькие: G, A, S
        small = [7, 0, 15]
        for i in small:
            for j in small:
                if i != j:
                    blosum[i, j] = 0.2
        
        # Ароматические: F, Y, W
        aromatic = [13, 18, 17]
        for i in aromatic:
            for j in aromatic:
                if i != j:
                    blosum[i, j] = 0.3
        
        # Similar pairs по реальной BLOSUM62
        similar_pairs = [
            (2, 5),   # N-Q
            (15, 16), # S-T
            (9, 10),  # I-L
            (4, 12),  # C-M
        ]
        for a, b in similar_pairs:
            blosum[a, b] = blosum[b, a] = 0.5
        
        return blosum

    def configure_optimizers(self):
        return torch.optim.AdamW(self.mlp_mixer.parameters(), lr=1e-4, weight_decay=1e-5)

    def on_fit_start(self):
        if self.trainer.train_dataloader is not None:
            steps_per_epoch = len(self.trainer.train_dataloader)
            total_steps = self.trainer.max_epochs * steps_per_epoch
            self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[
                    torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=0.1, total_iters=total_steps//10),
                    torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=total_steps - total_steps//10, eta_min=1e-6)
                ],
                milestones=[total_steps//10]
            )

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if hasattr(self, 'scheduler') and self.scheduler is not None:
            self.scheduler.step()
            self.log("train/lr", self.scheduler.get_last_lr()[0], on_step=True, on_epoch=False, sync_dist=True)

    def predict_for_sampling(self, batch: Dict, mode: str = "full", **kwargs):
        x_t = batch["x_t"]
        t_tensor = list(batch["t"].values())[0]
        t = t_tensor.view(-1, 1, 1) if t_tensor.dim() == 1 else t_tensor
        
        x1_latent = x_t["local_latents"].clone()
        x1_coords = x_t.get("bb_ca", torch.zeros_like(x1_latent[..., :3])).clone()
        
        eps = 1e-4
        v_latent = (x1_latent - x_t["local_latents"]) / (1.0 - t + eps)
        v_coords = (x1_coords - x_t.get("bb_ca", x1_coords)) / (1.0 - t + eps)
        
        return {
            "local_latents": {"v": v_latent, "x_1": x1_latent},
            "bb_ca": {"v": v_coords, "x_1": x1_coords}
        }

    def _get_fm_perturbed_with_velocity(self, z_clean: Tensor, coords: Tensor, mask: Tensor, t: Tensor, noise: Optional[Tensor] = None):
        if noise is None:
            noise = torch.randn_like(z_clean)
        
        x_0 = {
            "local_latents": noise,
            "bb_ca": coords + torch.randn_like(coords) * 0.05
        }
        x_1 = {
            "local_latents": z_clean,
            "bb_ca": coords
        }
        
        t_scalar = t.squeeze(-1).squeeze(-1)
        t_dict = {"local_latents": t_scalar, "bb_ca": t_scalar}
        x_t = self.flow_matcher.interpolate(x_0=x_0, x_1=x_1, t=t_dict, mask=mask)
        
        def predict_fn(batch, mode="full", **kwargs):
            x_t_batch = batch["x_t"]
            t_tensor = list(batch["t"].values())[0]
            t_exp = t_tensor.view(-1, 1, 1) if t_tensor.dim() == 1 else t_tensor
            
            x1_latent_true = z_clean.clone()
            x1_coords_true = coords.clone()
            
            eps = 1e-4
            v_latent = (x1_latent_true - x_t_batch["local_latents"]) / (1.0 - t_exp + eps)
            v_coords = (x1_coords_true - x_t_batch.get("bb_ca", x1_coords_true)) / (1.0 - t_exp + eps)
            
            return {
                "local_latents": {"v": v_latent, "x_1": x1_latent_true},
                "bb_ca": {"v": v_coords, "x_1": x1_coords_true}
            }
        
        batch_fm = {
            "x_t": x_t,
            "t": t_dict,
            "mask": mask,
        }
        
        with torch.no_grad():
            nn_out = self.flow_matcher.get_clean_pred_n_guided_vector(
                batch=batch_fm,
                predict_for_sampling=predict_fn,
                guidance_w=1.0,
                ag_ratio=0.0
            )
        
        v = nn_out["local_latents"]["v"]
        z_t = x_t["local_latents"]
        
        return z_t, v

    def _shared_forward(self, batch, log_prefix: str = "train"):
        # 1. Забираем чистые данные (Ground Truth)
        z1_clean = batch["z1"].to(self.device)
        z2_clean = batch["z2"].to(self.device)
        coords1 = batch["coords1"].to(self.device)
        coords2 = batch["coords2"].to(self.device)
        mask1 = batch["mask1"].to(self.device)
        mask2 = batch["mask2"].to(self.device)
        mask = mask1 & mask2

        if coords1.abs().max() > 50.0:
            coords1 = coords1 / 10.0
            coords2 = coords2 / 10.0

        bs, seq_len = z1_clean.shape[0], z1_clean.shape[1]

        # 2. СТАРТ ИЗ ШУМА
        z1_curr = torch.randn_like(z1_clean)
        z2_curr = torch.randn_like(z2_clean)

        # 3. НАСТРОЙКИ ИТЕРАТИВНОГО УТОЧНЕНИЯ
        n_refine_steps = 5
        step_t_val = 1.0 / n_refine_steps
        
        t_tensor = torch.full((bs,), step_t_val, device=self.device)
        t_dict = {"local_latents": t_tensor, "bb_ca": t_tensor}

        # 4. ЦИКЛ УТОЧНЕНИЯ
        transition_losses = []  # 🔹 Для отслеживания изменений между шагами
        
        for step in range(n_refine_steps):
            z1_prev = z1_curr.clone()
            z2_prev = z2_curr.clone()
            
            x_0_A = {"local_latents": z1_curr, "bb_ca": coords1}
            x_1_A = {"local_latents": z1_clean, "bb_ca": coords1}
            z1_next = self.flow_matcher.interpolate(x_0=x_0_A, x_1=x_1_A, t=t_dict, mask=mask1)["local_latents"]
            
            x_0_B = {"local_latents": z2_curr, "bb_ca": coords2}
            x_1_B = {"local_latents": z2_clean, "bb_ca": coords2}
            z2_next = self.flow_matcher.interpolate(x_0=x_0_B, x_1=x_1_B, t=t_dict, mask=mask2)["local_latents"]

            alpha = self.dual_path_alpha
            z1_mix = alpha * z1_next + (1.0 - alpha) * z2_next
            z2_mix = alpha * z2_next + (1.0 - alpha) * z1_next

            diff_mix = z1_mix - z2_mix
            z_cons = self.mlp_mixer(diff_mix, z1_mix, z2_mix)

            z1_curr = z_cons
            z2_curr = z_cons
            
            # 🔹 Transition loss: насколько изменился латент за шаг
            trans_loss = F.mse_loss(z_cons, z1_prev)
            transition_losses.append(trans_loss)

        # 5. ФИНАЛЬНОЕ СОСТОЯНИЕ
        z_cons_pred = z1_curr
        
        # 6. Логика последовательности
        # ==========================================
        # 6. ЛОГИКА ПОСЛЕДОВАТЕЛЬНОСТИ (ОБА ПУТИ)
        # ==========================================
        true_labels_1 = batch["seq1"].to(self.device)
        true_labels_2 = batch["seq2"].to(self.device)
        mask1_flat = mask1.view(-1)
        mask2_flat = mask2.view(-1)

        # --- Декодирование для пути 1 ---
        pred1 = self.autoencoder.decode(z_latent=z_cons_pred, ca_coors_nm=coords1, mask=mask1)
        logits1 = pred1["seq_logits"]  # [B, N, 20]

        loss_seq1 = F.cross_entropy(
            logits1.view(-1, 20), 
            true_labels_1.view(-1), 
            reduction='none', 
            label_smoothing=0.2
        )
        loss_seq1 = (loss_seq1 * mask1_flat).sum() / (mask1_flat.sum() + 1e-8)

        # --- Декодирование для пути 2 ---
        pred2 = self.autoencoder.decode(z_latent=z_cons_pred, ca_coors_nm=coords2, mask=mask2)
        logits2 = pred2["seq_logits"]  # [B, N, 20]

        loss_seq2 = F.cross_entropy(
            logits2.view(-1, 20), 
            true_labels_2.view(-1), 
            reduction='none', 
            label_smoothing=0.2
        )
        loss_seq2 = (loss_seq2 * mask2_flat).sum() / (mask2_flat.sum() + 1e-8)

        # Усреднённый лосс
        loss_seq = (loss_seq1 + loss_seq2) * 0.5

        # --- Accuracy (Top-1) для обоих путей ---
        seq_acc_1 = (logits1.argmax(-1) == true_labels_1).float()
        seq_acc_1 = (seq_acc_1 * mask1).sum() / (mask1.sum() + 1e-8)
        
        seq_acc_2 = (logits2.argmax(-1) == true_labels_2).float()
        seq_acc_2 = (seq_acc_2 * mask2).sum() / (mask2.sum() + 1e-8)
        
        seq_acc = (seq_acc_1 + seq_acc_2) * 0.5  # Финальная метрика

        # ==========================================
        # 🔹 РАСШИРЕННЫЕ МЕТРИКИ (ОБА ПУТИ)
        # ==========================================
        with torch.no_grad():
            # --- Путь 1 ---
            probs1 = F.softmax(logits1, dim=-1)           # [B, N, 20]
            probs1_flat = probs1.view(-1, 20)             # [B*N, 20]
            true_labels_1_flat = true_labels_1.view(-1)   # [B*N]
            
            # Top-3 Accuracy (путь 1)
            _, top3_pred_1 = logits1.view(-1, 20).topk(3, dim=-1)
            correct_top3_1 = (top3_pred_1 == true_labels_1_flat.unsqueeze(-1)).any(dim=-1).float()
            acc_top3_1 = (correct_top3_1 * mask1_flat).sum() / (mask1_flat.sum() + 1e-8)
            
            # Top-5 Accuracy (путь 1)
            _, top5_pred_1 = logits1.view(-1, 20).topk(5, dim=-1)
            correct_top5_1 = (top5_pred_1 == true_labels_1_flat.unsqueeze(-1)).any(dim=-1).float()
            acc_top5_1 = (correct_top5_1 * mask1_flat).sum() / (mask1_flat.sum() + 1e-8)
            
            # Entropy (путь 1)
            entropy1 = -torch.sum(probs1_flat * torch.log(probs1_flat + 1e-8), dim=-1)
            avg_entropy_1 = (entropy1 * mask1_flat).sum() / (mask1_flat.sum() + 1e-8)
            
            # Confidence (путь 1)
            confidence1 = probs1_flat.max(dim=-1).values
            avg_confidence_1 = (confidence1 * mask1_flat).sum() / (mask1_flat.sum() + 1e-8)
            
            # BLOSUM62 (путь 1)
            true_onehot_1 = F.one_hot(true_labels_1_flat, num_classes=20).float()
            blosum_row_1 = true_onehot_1 @ self.blosum62
            blosum_score_1 = torch.sum(probs1_flat * blosum_row_1, dim=1)
            avg_blosum_1 = (blosum_score_1 * mask1_flat).sum() / (mask1_flat.sum() + 1e-8)

            # --- Путь 2 (аналогично) ---
            probs2 = F.softmax(logits2, dim=-1)
            probs2_flat = probs2.view(-1, 20)
            true_labels_2_flat = true_labels_2.view(-1)
            
            _, top3_pred_2 = logits2.view(-1, 20).topk(3, dim=-1)
            correct_top3_2 = (top3_pred_2 == true_labels_2_flat.unsqueeze(-1)).any(dim=-1).float()
            acc_top3_2 = (correct_top3_2 * mask2_flat).sum() / (mask2_flat.sum() + 1e-8)
            
            _, top5_pred_2 = logits2.view(-1, 20).topk(5, dim=-1)
            correct_top5_2 = (top5_pred_2 == true_labels_2_flat.unsqueeze(-1)).any(dim=-1).float()
            acc_top5_2 = (correct_top5_2 * mask2_flat).sum() / (mask2_flat.sum() + 1e-8)
            
            entropy2 = -torch.sum(probs2_flat * torch.log(probs2_flat + 1e-8), dim=-1)
            avg_entropy_2 = (entropy2 * mask2_flat).sum() / (mask2_flat.sum() + 1e-8)
            
            confidence2 = probs2_flat.max(dim=-1).values
            avg_confidence_2 = (confidence2 * mask2_flat).sum() / (mask2_flat.sum() + 1e-8)
            
            true_onehot_2 = F.one_hot(true_labels_2_flat, num_classes=20).float()
            blosum_row_2 = true_onehot_2 @ self.blosum62
            blosum_score_2 = torch.sum(probs2_flat * blosum_row_2, dim=1)
            avg_blosum_2 = (blosum_score_2 * mask2_flat).sum() / (mask2_flat.sum() + 1e-8)

            # --- Усредняем метрики по двум путям ---
            acc_top3 = (acc_top3_1 + acc_top3_2) * 0.5
            acc_top5 = (acc_top5_1 + acc_top5_2) * 0.5
            avg_entropy = (avg_entropy_1 + avg_entropy_2) * 0.5
            avg_confidence = (avg_confidence_1 + avg_confidence_2) * 0.5
            avg_blosum = (avg_blosum_1 + avg_blosum_2) * 0.5
        
        # 7. Логирование геометрии (FAFE)
        loss_fafe = torch.tensor(0.0, device=self.device)
        if batch.get("atom_mask1") is not None and batch["atom_mask1"].sum() > 0:
            pred_full_A = self.autoencoder.decode(z_latent=z_cons_pred, ca_coors_nm=coords1, mask=mask1)["coors_nm"]
            pred_full_B = self.autoencoder.decode(z_latent=z_cons_pred, ca_coors_nm=coords2, mask=mask2)["coors_nm"]
            
            gt_A = batch["coords1_full"].to(self.device) / 10.0
            gt_B = batch["coords2_full"].to(self.device) / 10.0
            mask_atom_A = batch["atom_mask1"].to(self.device)
            mask_atom_B = batch["atom_mask2"].to(self.device)

            loss_fafe = 0.5 * (
                fafe_loss(pred_full_A, gt_A, mask1, mask_atom_A) +
                fafe_loss(pred_full_B, gt_B, mask2, mask_atom_B)
            )
            with torch.no_grad():
                R_pred, t_pred = build_backbone_frames(pred_full_A)
                R_gt, t_gt = build_backbone_frames(gt_A)
                R_diff = R_pred.transpose(-1, -2) @ R_gt
                trace = R_diff.diagonal(dim1=-2, dim2=-1).sum(-1)
                cos_a = torch.clamp((trace - 1) / 2, -1 + 1e-7, 1 - 1e-7)
                angle_err_deg = torch.acos(cos_a) * (180 / torch.pi) * mask1
                n_valid = mask1.sum().clamp(min=1)
                self.log(f"{log_prefix}/geom_angle_deg", angle_err_deg.sum() / n_valid, 
                         on_step=(log_prefix=="train"), on_epoch=True, sync_dist=True)

        # 8. Веса и итоговый лосс
        w_seq = 1.0
        w_fafe = 15.0
        
        mean_z_target = (z1_clean + z2_clean) * 0.5
        latent_reg = F.mse_loss(z_cons_pred, mean_z_target)

        total_loss = w_seq * loss_seq + w_fafe * loss_fafe + 0.05 * latent_reg
        
        # 9. Логирование всех метрик
        is_train = (log_prefix == "train")
        self.log(f"{log_prefix}/total_loss", total_loss, on_step=is_train, on_epoch=True, prog_bar=is_train, sync_dist=True)
        self.log(f"{log_prefix}/loss_seq", loss_seq, on_step=is_train, on_epoch=True, sync_dist=True)
        self.log(f"{log_prefix}/loss_fafe", loss_fafe, on_step=is_train, on_epoch=True, sync_dist=True)
        self.log(f"{log_prefix}/seq_top1_acc", seq_acc, on_step=is_train, on_epoch=True, prog_bar=is_train, sync_dist=True)
        
        # 🔹 Новые метрики
        self.log(f"{log_prefix}/seq_top3_acc", acc_top3, on_step=is_train, on_epoch=True, sync_dist=True)
        self.log(f"{log_prefix}/seq_top5_acc", acc_top5, on_step=is_train, on_epoch=True, sync_dist=True)
        self.log(f"{log_prefix}/pred_entropy", avg_entropy, on_step=is_train, on_epoch=True, sync_dist=True)
        self.log(f"{log_prefix}/avg_confidence", avg_confidence, on_step=is_train, on_epoch=True, sync_dist=True)
        self.log(f"{log_prefix}/blosum62_score", avg_blosum, on_step=is_train, on_epoch=True, sync_dist=True)
        self.log(f"{log_prefix}/conf_acc_gap", avg_confidence - seq_acc, on_step=False, on_epoch=True, sync_dist=True)
        
        # Transition loss (средний за все шаги refine)
        if len(transition_losses) > 0:
            avg_trans_loss = sum(transition_losses) / len(transition_losses)
            self.log(f"{log_prefix}/transition_loss", avg_trans_loss, on_step=is_train, on_epoch=True, sync_dist=True)
        
        if loss_fafe > 0:
            with torch.no_grad():
                self.log(f"{log_prefix}/geom_angle_deg", angle_err_deg.sum() / n_valid, 
                         on_step=is_train, on_epoch=True, sync_dist=True)
        
        return total_loss, loss_seq, loss_fafe, seq_acc

    def training_step(self, batch, batch_idx):
        total, seq, fafe, acc = self._shared_forward(batch, log_prefix="train")
        return total

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            total, seq, fafe, acc = self._shared_forward(batch, log_prefix="val")
        return total

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            total, seq, fafe, acc = self._shared_forward(batch, log_prefix="test")
        return total