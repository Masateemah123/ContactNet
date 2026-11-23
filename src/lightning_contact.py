# src/lightning_contact.py
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

# ----------------- helpers -----------------
def decimal2binary(x: torch.Tensor) -> torch.Tensor:
    """Map class index [0..15] -> 4-bit contacts; returns uint8 (N,4)."""
    x = x.to(torch.long)
    mask = (2 ** torch.arange(3, -1, -1, device=x.device, dtype=torch.long))
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).to(torch.uint8)

class _FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = float(gamma)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        return ((1.0 - pt) ** self.gamma * ce).mean()
    
    def _weighted_loss(self, logits, targets):
        loss = self.loss_fn(logits, targets)
        weights = torch.ones_like(loss)
        critical = torch.tensor([1, 2, 4, 8, 7, 11, 13, 14], device=targets.device)
        is_critical = torch.isin(targets, critical)
        weights[is_critical] = self.slip_weight_factor
        return (loss * weights).mean()

# ----------------- backbone -----------------

class contact_cnn_gru_progressive(nn.Module):
    def __init__(self, gru_hidden_size=128, num_classes=16):
        super().__init__()
        # Old widths: 64, 128, 256
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels=48, out_channels=64,kernel_size=3, stride=1, padding=1), 
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.MaxPool1d(kernel_size=2, 
                         stride=2),
        )
      
        self.block2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        self.block3 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.MaxPool1d(kernel_size=2, 
                         stride=2)
        )

        # Each GRU gets half of gru_hidden_size -> 64 if gru_hidden_size=128
        self.gru_early = nn.GRU(input_size=128,
                                hidden_size=gru_hidden_size // 2,
                                num_layers=1, 
                                batch_first=True)
        
        self.gru_late  = nn.GRU(input_size=256, 
                                hidden_size=gru_hidden_size // 2,
                                num_layers=1, 
                                batch_first=True)

        # Classifier: 128 -> 512 -> 256 -> 16 (to match ckpt shapes)
        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden_size, 512), 
            nn.ReLU(), 
            nn.Dropout(0.5),
            nn.Linear(512, 256),            
            nn.ReLU(), 
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 48)
        x = x.permute(0, 2, 1)          #(B, 48, T)
        x = self.block1(x)              #(B, 64, T1)
        x = self.block2(x)              #(B, 128, T2)

        # Early GRU on mid-level features
        x_early = x.permute(0, 2, 1)    #(B, T2, 128)
        
        # Progressive feature extraction
        gru_early_out, _ = self.gru_early(x_early)
        early_features = gru_early_out[:, -1, :]  #(B, 64)

        # Continue CNN
        x = self.block3(x)              #(B, 256, T3)

        # Late GRU on high-level features
        x_late = x.permute(0, 2, 1)     #(B, T3, 256)
        gru_late_out, _ = self.gru_late(x_late)
        late_features = gru_late_out[:, -1, :]    #(B, 64)

        # Combines multi-scale temporal features
        combined_features = torch.cat([early_features, late_features], dim=1)  # (B, 128)
        logits = self.classifier(combined_features)                    # (B, 16)
        return logits
  
# ----------------- Lightning Module -----------------
class ContactLitModule(pl.LightningModule):
    """
    Wraps model + train/val/test loops.  
    Label smoothing works by softening the hard one-hot encoded labels, 
    distributing a small amount of probability mass to other classes. 
    Focal loss reduces the relative loss for well-classified examples, 
    allowing the model to focus on hard-to-classify examples.
    Batch:
      - data:  (B, T, 48)
      - label: (B,) in [0..15]
    """
    def __init__(self,
                 num_classes: int = 16,
                 loss: str = "focal",           # "ce" or "focal"
                 focal_gamma: float = 2.0,
                 lr: float = 1e-4,
                 weight_decay: float = 1e-5,
                 pretrained_path: Optional[str] = None,
                 log_per_leg: bool = True,
                 scheduler: Optional[dict] = None,
                 label_smoothing: float = 0.0,
                 slip_weight_factor: float = 3.0):
        super().__init__()
        self.save_hyperparameters()

        self.model = contact_cnn_gru_progressive(num_classes=num_classes)
        self.init_lr = lr
        self.weight_decay = weight_decay
        self.scheduler_cfg = scheduler
        self.log_per_leg = log_per_leg
        self.slip_weight_factor = slip_weight_factor
        self.critical_classes = torch.tensor([1, 2, 4, 8, 7, 11, 13, 14])

        loss_name = (loss or "ce").lower()
        if loss_name == "ce":
            self.loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        else:
            self.loss_fn = _FocalLoss(focal_gamma)

        if pretrained_path:
            ckpt = torch.load(pretrained_path, map_location="cpu")
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                sd = {k.replace("model.", "", 1): v
                      for k, v in ckpt["state_dict"].items()
                      if k.startswith("model.")}
            else:
                sd = ckpt.get("model_state_dict", ckpt)
            self.model.load_state_dict(sd, strict=False)
            print(f"[ContactLitModule] Loaded pretrained from {pretrained_path}")

    # -------- metrics --------
    def _metrics(self, logits, targets) -> Dict[str, Any]:
        with torch.no_grad():
            pred = logits.argmax(dim=1)
            class_acc = (pred == targets).float().mean()
            pb, gb = decimal2binary(pred), decimal2binary(targets)
            per_leg_acc = (pb == gb).float().mean(dim=0)
        return {"class_acc": class_acc, "per_leg_acc": per_leg_acc}

    # -------- steps --------
    def training_step(self, batch, batch_idx):
        x, y = batch["data"], batch["label"].long()
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        m = self._metrics(logits, y)
        self.log("train/loss", loss, on_epoch=True, prog_bar=True)
        self.log("train/acc_class", m["class_acc"], on_epoch=True, prog_bar=True)
        self.log("train/acc_leg_mean", m["per_leg_acc"].mean(), on_epoch=True, prog_bar=True)
        if self.log_per_leg:
            for i in range(4):
                self.log(f"train/acc_leg{i}", m["per_leg_acc"][i], on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["data"], batch["label"].long()
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        m = self._metrics(logits, y)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        self.log("val/acc_class", m["class_acc"], on_epoch=True, prog_bar=True)
        self.log("val/acc_leg_mean", m["per_leg_acc"].mean(), on_epoch=True, prog_bar=True)
        if self.log_per_leg:
            for i in range(4):
                self.log(f"val/acc_leg{i}", m["per_leg_acc"][i], on_epoch=True)
        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        x, y = batch["data"], batch["label"].long()
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        m = self._metrics(logits, y)
        self.log("test/loss", loss)
        self.log("test/acc_class", m["class_acc"])
        self.log("test/acc_leg_mean", m["per_leg_acc"].mean())
        if self.log_per_leg:
            for i in range(4):
                self.log(f"test/acc_leg{i}", m["per_leg_acc"][i])
        return {"test_loss": loss}

    # -------- optimizer --------
    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.init_lr, weight_decay=self.weight_decay)
        sch_cfg = self.scheduler_cfg

        if not sch_cfg:
            return opt

        name = str(sch_cfg.get("name", "none")).lower()

        if name == "reduce_on_plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt,
                mode=sch_cfg.get("mode", "min"),
                factor=sch_cfg.get("factor", 0.5),
                patience=sch_cfg.get("patience", 3),
                min_lr=sch_cfg.get("min_lr", 0.0),
            )
            return {
                "optimizer": opt,
                "lr_scheduler": {"scheduler": scheduler, "monitor": sch_cfg.get("monitor", "val/loss")},
            }

        elif name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt,
                T_max=sch_cfg.get("t_max", getattr(self.trainer, "max_epochs", 30)),
                eta_min=sch_cfg.get("eta_min", 0.0),
            )
            return {"optimizer": opt, "lr_scheduler": scheduler}

        elif name == "onecycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                opt,
                max_lr=sch_cfg.get("max_lr", self.init_lr),
                epochs=sch_cfg.get("epochs", getattr(self.trainer, "max_epochs", 30)),
                steps_per_epoch=sch_cfg["steps_per_epoch"],
            )
            return {"optimizer": opt, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

        elif name == "warmup_const":
            warmup_epochs = sch_cfg.get("warmup_epochs", 5)
            def lr_lambda(epoch):
                return (epoch + 1) / warmup_epochs if epoch < warmup_epochs else 1.0
            scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
            return {"optimizer": opt, "lr_scheduler": scheduler}

        return opt