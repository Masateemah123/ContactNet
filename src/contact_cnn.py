from typing import Optional, Dict, Any, Tuple
import torch
from torchmetrics.classification import MulticlassF1Score
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


# --------------------------------------------
#  Helpers
# --------------------------------------------

def decimal2binary(x: torch.Tensor) -> torch.Tensor:
    """Map class index [0..15] -> 4-bit contacts; returns uint8 (N,4)."""
    x = x.to(torch.long)
    mask = (2 ** torch.arange(3, -1, -1, device=x.device))
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).to(torch.uint8)


class _FocalLoss(nn.Module):
    """
    Standard Focal Loss with optional upweighting of rare slip classes.
    """
    def __init__(self, gamma: float = 2.0, slip_weight_factor: float = 1.0):
        super().__init__()
        self.gamma = float(gamma)
        self.slip_weight_factor = float(slip_weight_factor)
        # critical classes used for up-weighting (example set)
        self.critical_classes = torch.tensor([1, 2, 4, 8, 7, 11, 13, 14], dtype=torch.long)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (B, C), targets: (B,)
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)                # p_t
        loss = (1.0 - pt) ** self.gamma * ce

        if self.slip_weight_factor > 1.0:
            crit = self.critical_classes.to(targets.device)
            # iscrit: (B,) boolean indicating target is in critical set
            iscrit = (targets.unsqueeze(-1) == crit).any(dim=-1)
            loss = loss * torch.where(iscrit, self.slip_weight_factor, 1.0)

        return loss.mean()


# --------------------------------------------
#  Model Backbone
# --------------------------------------------

class contact_cnn_gru_progressive(nn.Module):
    def __init__(self, gru_hidden_size: int = 128, num_classes: int = 16, input_channels: int = 48):
        super().__init__()

        assert gru_hidden_size % 2 == 0, "GRU hidden size must be even."

        # ---------------- Convolution ----------------
        self.block1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool1d(2)
        )

        self.block2 = nn.Sequential(
            nn.Conv1d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool1d(2)
        )

        self.block3 = nn.Sequential(
            nn.Conv1d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.MaxPool1d(2)
        )

        # ---------------- GRUs ----------------
        self.gru_early = nn.GRU(
            input_size=128,
            hidden_size=gru_hidden_size // 2,
            num_layers=1,
            batch_first=True
        )

        self.gru_late = nn.GRU(
            input_size=256,
            hidden_size=gru_hidden_size // 2,
            num_layers=1,
            batch_first=True
        )

        # ---------------- Classifiers ----------------
        self.main_classifier = nn.Sequential(
            nn.Linear(gru_hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

        # Auxiliary classifier operates on the sequence outputs of the late GRU
        self.auxi_classifier = nn.Sequential(
            nn.Linear(gru_hidden_size // 2, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor):
        # Accept either (B, T, C) or (B, C, T)
        if x.dim() != 3:
            raise ValueError(f"Expected 3D tensor, got {tuple(x.shape)}")

        # Heuristic to detect layout: if last dim resembles channels (<=128) and time > channels, assume (B,T,C)
        # Otherwise assume (B,C,T)
        if x.shape[-1] <= 128 and x.shape[-1] <= x.shape[1]:
            # x is (B, T, C) -> convert to (B, C, T) for Conv1d
            x = x.permute(0, 2, 1)

        # ---------- CNN ----------
        x = self.block1(x)   # -> (B, 64, T/2)
        x = self.block2(x)   # -> (B, 128, T/4)

        # Prepare early GRU input (B, T2, 128)
        x_early = x.permute(0, 2, 1)
        gru_early_out, _ = self.gru_early(x_early)

        # Continue CNN
        x = self.block3(x)   # -> (B, 256, T/8)

        x_late = x.permute(0, 2, 1)
        gru_late_out, _ = self.gru_late(x_late)

        # Main logits from final timestep of each GRU
        early_final = gru_early_out[:, -1, :]   # (B, hidden//2)
        late_final = gru_late_out[:, -1, :]     # (B, hidden//2)
        fused = torch.cat([early_final, late_final], dim=1)  # (B, hidden)

        main_logits = self.main_classifier(fused)            # (B, num_classes)

        # Auxiliary logits: per-timestep predictions from late GRU (B, T3, num_classes)
        aux_logits = self.auxi_classifier(gru_late_out)

        return main_logits, aux_logits


# --------------------------------------------
#  Lightning Module
# --------------------------------------------

class ContactLitModule(pl.LightningModule):
    def __init__(
        self,
        num_classes: int = 16,
        input_channels: int = 48,
        loss: str = "focal",
        focal_gamma: float = 2.0,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        pretrained_path: Optional[str] = None,
        log_per_leg: bool = True,
        scheduler: Optional[Dict[str, Any]] = None,
        label_smoothing: float = 0.0,
        slip_weight_factor: float = 5.0,
        class_weights: Optional[torch.Tensor] = None,
        use_auxiliary_loss: bool = False,
        auxiliary_loss_weight: float = 0.3
    ):
        super().__init__()
        self.save_hyperparameters()

        # Metrics
        self.train_f1 = MulticlassF1Score(num_classes=num_classes, average='macro')
        self.val_f1   = MulticlassF1Score(num_classes=num_classes, average='macro')
        self.test_f1  = MulticlassF1Score(num_classes=num_classes, average='macro')

        # instantiate model backbone
        self.model = contact_cnn_gru_progressive(
            gru_hidden_size=128,
            num_classes=num_classes,
            input_channels=input_channels
        )

        # Loss selection
        if isinstance(class_weights, (list, tuple, torch.Tensor)):
            class_weights = torch.tensor(class_weights, dtype=torch.float32)

        if loss.lower() in ("ce", "cross_entropy", "crossentropy"):
            self.loss_fn = nn.CrossEntropyLoss(
                weight=class_weights,
                label_smoothing=float(label_smoothing)
            )
        else:
            # focal loss
            self.loss_fn = _FocalLoss(gamma=float(focal_gamma),
                                      slip_weight_factor=float(slip_weight_factor))

        # Optional pretrained - robust loader
        if pretrained_path:
            try:
                ckpt = torch.load(pretrained_path, map_location="cpu")
                # ckpt may be a dict with 'state_dict' or a plain state_dict
                if isinstance(ckpt, dict) and ("state_dict" in ckpt or "model_state_dict" in ckpt):
                    state_key = "state_dict" if "state_dict" in ckpt else "model_state_dict"
                    sd = ckpt[state_key]
                else:
                    sd = ckpt
                # normalize keys: accept both "model.*" or raw keys
                new_sd = {}
                for k, v in sd.items():
                    if k.startswith("model."):
                        new_sd[k.replace("model.", "", 1)] = v
                    else:
                        new_sd[k] = v
                # load leniently
                self.model.load_state_dict(new_sd, strict=False)
                print(f"[ContactLitModule] Loaded pretrained: {pretrained_path}")
            except Exception as e:
                print(f"[ContactLitModule] Warning: failed to load pretrained '{pretrained_path}': {e}")

    # ---------------- Loss ----------------

    def _compute_loss(self, main_logits: torch.Tensor, aux_logits: torch.Tensor,
                      targets: torch.Tensor, seq_targets: Optional[torch.Tensor] = None):
        """
        Returns (final_loss, main_loss, aux_loss)
        - main_loss: loss on final logits
        - aux_loss: per-timestep seq loss (if enabled)
        """
        main_loss = self.loss_fn(main_logits, targets)

        if not self.hparams.use_auxiliary_loss:
            return main_loss, main_loss, torch.tensor(0.0, device=self.device)

        # aux_logits: (B, T, C)
        B, T, C = aux_logits.shape
        aux_logits_flat = aux_logits.reshape(B * T, C)

        if seq_targets is not None:
            seq_targets_flat = seq_targets.reshape(B * T)
        else:
            # expand class target to per-timestep pseudo target if seq not provided
            seq_targets_flat = targets.unsqueeze(1).expand(-1, T).reshape(B * T)

        aux_loss = self.loss_fn(aux_logits_flat, seq_targets_flat)

        final = main_loss + float(self.hparams.auxiliary_loss_weight) * aux_loss
        return final, main_loss, aux_loss

    # ---------------- Metrics ----------------

    def _metrics(self, logits: torch.Tensor, targets: torch.Tensor):
        pred = logits.argmax(dim=1)
        class_acc = (pred == targets).float().mean()

        pb, gb = decimal2binary(pred), decimal2binary(targets)
        per_leg_acc = (pb == gb).float().mean(dim=0)

        return {"class_acc": class_acc, "per_leg_acc": per_leg_acc}

    # ---------------- Training ----------------

    def training_step(self, batch, batch_idx):
        # expect batch to be (x, y, seq_y) or dict-like -> handled by data loader
        if isinstance(batch, (list, tuple)):
            x, y, seq_y = batch
        elif isinstance(batch, dict):
            x = batch.get("data") or batch.get("x")
            y = batch.get("label") or batch.get("y")
            seq_y = batch.get("seq_label") or batch.get("seq_y")
        else:
            raise RuntimeError("Unexpected batch format in training_step")

        y = y.long()
        if seq_y is not None:
            seq_y = seq_y.long()

        main_logits, aux_logits = self.model(x)
        total, main_loss, aux_loss = self._compute_loss(main_logits, aux_logits, y, seq_y)

        m = self._metrics(main_logits, y)
        # update metrics with predicted logits and true labels
        self.train_f1.update(main_logits.softmax(dim=1).argmax(dim=1), y)

        # logging
        self.log("train/loss", total, on_epoch=True, prog_bar=True, logger=True)
        self.log("train/f1_macro", self.train_f1, on_epoch=True, prog_bar=True, logger=True)
        self.log("train/acc_class", m["class_acc"], on_epoch=True, prog_bar=False, logger=True)
        self.log("train/acc_leg_mean", m["per_leg_acc"].mean(), on_epoch=True, prog_bar=False, logger=True)

        return total

    def validation_step(self, batch, idx):
        if isinstance(batch, (list, tuple)):
            x, y, seq_y = batch
        elif isinstance(batch, dict):
            x = batch.get("data") or batch.get("x")
            y = batch.get("label") or batch.get("y")
            seq_y = batch.get("seq_label") or batch.get("seq_y")
        else:
            raise RuntimeError("Unexpected batch format in validation_step")

        y = y.long()
        if seq_y is not None:
            seq_y = seq_y.long()

        main_logits, aux_logits = self.model(x)
        total, main_loss, aux_loss = self._compute_loss(main_logits, aux_logits, y, seq_y)

        m = self._metrics(main_logits, y)
        self.val_f1.update(main_logits.softmax(dim=1).argmax(dim=1), y)

        self.log("val/loss", total, on_epoch=True, prog_bar=True, logger=True)
        self.log("val/f1_macro", self.val_f1, on_epoch=True, prog_bar=True, logger=True)
        self.log("val/acc_class", m["class_acc"], on_epoch=True, prog_bar=False, logger=True)

        return total

    # ---------------- Test ----------------

    def test_step(self, batch, batch_idx):
        if isinstance(batch, (list, tuple)):
            x, y, seq_y = batch
        elif isinstance(batch, dict):
            x = batch.get("data") or batch.get("x")
            y = batch.get("label") or batch.get("y")
            seq_y = batch.get("seq_label") or batch.get("seq_y")
        else:
            raise RuntimeError("Unexpected batch format in test_step")

        y = y.long()
        if seq_y is not None:
            seq_y = seq_y.long()

        main_logits, aux_logits = self.model(x)
        total, main_loss, aux_loss = self._compute_loss(main_logits, aux_logits, y, seq_y)

        self.test_f1.update(main_logits.softmax(dim=1).argmax(dim=1), y)
        self.log("test/f1_macro", self.test_f1, on_epoch=True, prog_bar=True, logger=True)

        return total

    # ---------------- Optimizer ----------------

    def configure_optimizers(self):
        opt = torch.optim.Adam(
            self.parameters(),
            lr=float(self.hparams.lr),
            weight_decay=float(self.hparams.weight_decay)
        )

        scfg = self.hparams.scheduler
        if scfg is None:
            return opt

        name = scfg.get("name", "").lower()

        if name == "reduce_on_plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt,
                mode="min",
                factor=float(scfg.get("factor", 0.5)),
                patience=int(scfg.get("patience", 2)),
                min_lr=float(scfg.get("min_lr", 0.0))
            )
            return {
                "optimizer": opt,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": scfg.get("monitor", "val/loss")
                }
            }

        elif name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt,
                T_max=int(scfg.get("t_max", 30)),
                eta_min=float(scfg.get("eta_min", 0.0))
            )
            return {"optimizer": opt, "lr_scheduler": scheduler}

        return opt
