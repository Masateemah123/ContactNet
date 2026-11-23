# src/train_lightning.py
import os
import yaml
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from src.contact_cnn import ContactLitModule
from src.dataset import contact_dataset


def main(cfg_path: str):
    # ---- load config ----
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg.get("seed", 42))
    pl.seed_everything(seed, workers=True)

    data_folder = cfg["data_folder"]
    batch_size = int(cfg.get("batch_size", 30))
    num_workers = int(cfg.get("num_workers", 4))
    shuffle = bool(cfg.get("shuffle", True))

    print(f"[BOOT] __file__={__file__}", flush=True)
    print(f"[BOOT] cwd={os.getcwd()}", flush=True)
    print(f"[BOOT] cfg_path={cfg_path}", flush=True)

    # ---------------------------------- window size derived from seconds ------------------------------------------
    sr = int(cfg.get("sample_rate_hz", 1000))
    window_size = int(round(float(cfg.get("window_sec", 0.15)) * sr))
    cfg["window_size"] = window_size
    print(f"[CFG] sample_rate_hz={sr}, window_size(samples)={window_size} (~{window_size/sr:.3f}s)", flush=True)

    # ------------------------------------- loaders -----------------------------------------------------------------
    train_ds = contact_dataset(
        data_path=os.path.join(data_folder, "train.npy"),
        label_path=os.path.join(data_folder, "train_label.npy"),
        window_size=window_size,
        device="cpu",
    )
    val_ds = contact_dataset(
        data_path=os.path.join(data_folder, "val.npy"),
        label_path=os.path.join(data_folder, "val_label.npy"),
        window_size=window_size,
        device="cpu",
    )

    train_ld = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    val_ld = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    print("Train dataset:", type(train_ld.dataset).__name__)
    print("Val   dataset:", type(val_ld.dataset).__name__)

    # --------------------------------- model  --------------------------------------
    
    lit = ContactLitModule(
        lr=float(cfg.get("init_lr", 1e-4)),
        weight_decay=float(cfg.get("weight_decay", 1e-5)),
        loss=str(cfg.get("loss_type", "focal")),
        focal_gamma=float(cfg.get("focal_gamma", 2.0)),
        label_smoothing=float(cfg.get("label_smoothing", 0.0)),
        slip_weight_factor=float(cfg.get("slip_weight_factor", 3.0))
)

    # ---------------------------------- logging / checkpoints ---------------------------------------
    #
    model_save_base = cfg["model_save_path"]  
    ckpt_dir = os.path.dirname(model_save_base) or "."
    log_dir = cfg.get("log_writer_path", "./logs")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # persist run config alongside checkpoints
    with open(os.path.join(ckpt_dir, "train_config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f)

    logger = TensorBoardLogger(save_dir=log_dir, name="", version=None, default_hp_metric=False)

    # ContactLitModule logs "val/acc_class" — monitor and maximize that.
    ckpt_cb = pl.callbacks.ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=os.path.basename(model_save_base) + "-{epoch:02d}-{val_acc_class:.4f}",
        monitor="val/acc_class",
        mode="max",
        save_top_k=int(cfg.get("save_top_k", 3)),
        save_last=True,
        auto_insert_metric_name=False,
    )
    lr_cb = pl.callbacks.LearningRateMonitor(logging_interval="epoch")

    callbacks = [ckpt_cb, lr_cb]
    es_patience = int(cfg.get("early_stop_patience", 0))
    if es_patience > 0:
        callbacks.append(pl.callbacks.EarlyStopping(monitor="val/acc_class", mode="max", patience=es_patience))

    # ---- trainer ----
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    trainer = pl.Trainer(
        max_epochs=int(cfg.get("num_epoch", 30)),
        accelerator="auto",
        devices=1,
        precision=int(cfg.get("precision", 32)),  # 16 or 32
        log_every_n_steps=int(cfg.get("log_every_n_steps", 50)),
        callbacks=callbacks,
        logger=logger,
        default_root_dir=ckpt_dir,
        gradient_clip_val=float(cfg.get("grad_clip", 0.25)),
        deterministic=bool(cfg.get("deterministic", True)),
        accumulate_grad_batches=int(cfg.get("accumulate_grad_batches", 1)),
    )

    # ---- train ----
    trainer.fit(lit, train_ld, val_ld)

    # ---- report best ----
    print(f"[BEST] path={ckpt_cb.best_model_path} score={ckpt_cb.best_model_score}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()
    main(args.config)
