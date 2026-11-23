# src/inference_one_seq.py  (locked permutation = 2,0,3,1)

import os, sys, glob, re, argparse
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import yaml
from tqdm import tqdm
import scipy.io as sio
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from src.contact_cnn import ContactLitModule
from src.contact_cnn import contact_cnn_gru_progressive
from src.dataset import contact_dataset

# ---------- constants ----------
FIXED_PERM = (2, 0, 3, 1)  # <— locked permutation (“2031”)

# ---------- helpers ----------
def bits2decimal(bits_u8: torch.Tensor) -> torch.Tensor:
    w = torch.tensor([8, 4, 2, 1], device=bits_u8.device, dtype=torch.int64)
    return (bits_u8.to(torch.int64) * w).sum(dim=-1)

def apply_perm(bits_u8: torch.Tensor, perm: tuple[int,int,int,int]) -> torch.Tensor:
    idx = torch.tensor(perm, device=bits_u8.device)
    return bits_u8.index_select(dim=1, index=idx)

def compute_f1_and_confusion(bin_pred: torch.Tensor, bin_gt: torch.Tensor, save_path=None):
    pred_np = bin_pred.cpu().numpy()
    gt_np = bin_gt.cpu().numpy()

    # Calculate L1 Error (New, less stringent metric)
    l1_errors = torch.abs(bin_pred.to(torch.float32) - bin_gt.to(torch.float32)).sum(dim=1)
    mean_l1_error = l1_errors.mean().item()
    
    print("\n" + "="*60)
    print("PER-LEG BINARY CLASSIFICATION METRICS")
    print("="*60)

    per_leg_f1, per_leg_conf = [], []
    for leg in range(4):
        pr, gt = pred_np[:, leg], gt_np[:, leg]
        f1 = f1_score(gt, pr, average='binary', zero_division=0)
        per_leg_f1.append(f1)
        cm = confusion_matrix(gt, pr, labels=[0, 1])
        per_leg_conf.append(cm)

        tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
        prec = tp/(tp+fp) if (tp+fp)>0 else 0
        rec  = tp/(tp+fn) if (tp+fn)>0 else 0
        spec = tn/(tn+fp) if (tn+fp)>0 else 0

        print(f"\nLeg {leg}:")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  Confusion Matrix:")
        print(f"    TN={tn:<6} FP={fp:<6}")
        print(f"    FN={fn:<6} TP={tp:<6}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  Specificity:{spec:.4f}")

    avg_f1 = float(np.mean(per_leg_f1))
    print(f"\nAverage F1 Score (across legs): {avg_f1:.4f}")
    print(f"Per-leg F1 scores: {per_leg_f1}")
    print(f"Mean L1 Error (Avg. Incorrect Legs): {mean_l1_error:.4f} legs") # New Metric

    print("\n" + "="*60)
    print("16-CLASS CONTACT PATTERN METRICS")
    print("="*60)

    pred_16 = bits2decimal(bin_pred).cpu().numpy()
    gt_16   = bits2decimal(bin_gt).cpu().numpy()

    class_acc = (pred_16 == gt_16).mean().item() # Explicitly calculating 16-class accuracy again
    f1_macro = f1_score(gt_16, pred_16, average='macro', zero_division=0)
    f1_micro = f1_score(gt_16, pred_16, average='micro', zero_division=0)
    f1_w     = f1_score(gt_16, pred_16, average='weighted', zero_division=0)
    
    print(f"\n16-Class Accuracy: {class_acc:.4f}")
    print(f"16-Class F1 Scores:\n  Macro F1: {f1_macro:.4f}\n  Micro F1: {f1_micro:.4f}\n  Weighted F1: {f1_w:.4f}")

    cm16 = confusion_matrix(gt_16, pred_16, labels=list(range(16)))
    print("\nPer-Class Report:")
    print(classification_report(
        gt_16, pred_16, labels=list(range(16)),
        target_names=[f"Class_{i:02d}" for i in range(16)],
        zero_division=0
    ))

    if save_path:
        print(f"\nSaving confusion matrix plots to: {save_path}")
        fig = plt.figure(figsize=(20, 10))
        for leg in range(4):
            ax = plt.subplot(2, 4, leg + 1)
            sns.heatmap(per_leg_conf[leg], annot=True, fmt='d', cmap='Blues',
                        xticklabels=['No Contact','Contact'],
                        yticklabels=['No Contact','Contact'],
                        cbar=False, ax=ax)
            ax.set_title(f'Leg {leg} - F1: {per_leg_f1[leg]:.3f}')
            ax.set_ylabel('True'); ax.set_xlabel('Pred')

        ax = plt.subplot(2, 2, 3)
        sns.heatmap(cm16, annot=True, fmt='d', cmap='YlOrRd', ax=ax,
                    xticklabels=range(16), yticklabels=range(16))
        ax.set_title(f'16-Class Confusion Matrix\nMacro F1: {f1_macro:.3f}')
        ax.set_ylabel('True'); ax.set_xlabel('Pred')

        ax = plt.subplot(2, 2, 4)
        cmn = cm16.astype('float') / np.maximum(cm16.sum(axis=1, keepdims=True), 1)
        sns.heatmap(np.nan_to_num(cmn), annot=True, fmt='.2f', cmap='YlOrRd', ax=ax,
                    xticklabels=range(16), yticklabels=range(16))
        ax.set_title(f'16-Class Confusion (Normalized)\nWeighted F1: {f1_w:.3f}')
        ax.set_ylabel('True'); ax.set_xlabel('Pred')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrices saved to {save_path}")
        plt.close()

    return {
        'per_leg': {'f1_scores': per_leg_f1, 'avg_f1': avg_f1, 'confusion_matrices': per_leg_conf},
        '16_class': {'f1_macro': f1_macro, 'f1_micro': f1_micro, 'f1_weighted': f1_w, 'confusion_matrix': cm16},
        'l1_error': mean_l1_error # Store new metric
    }

# --- add near the top, alongside other constants ---
def build_class_patterns(device):
    patterns = []
    for i in range(16):
        bits = [(i >> j) & 1 for j in range(3, -1, -1)]
        patterns.append(bits)
    return torch.tensor(patterns, dtype=torch.float32, device=device)  # (16,4)

def resolve_ckpt(path: str) -> str:
    if os.path.isfile(path):
        return path
    if os.path.isdir(path):
        cands = glob.glob(os.path.join(path, "*.ckpt"))
        if not cands:
            raise FileNotFoundError(f"No .ckpt files in: {path}")
        def epoch_num(p):
            m = re.search(r"-(\d+)-", os.path.basename(p))
            return int(m.group(1)) if m else -1
        best = max(cands, key=lambda p: (epoch_num(p), os.path.getmtime(p)))
        print(f"[resolve_ckpt] Using: {best}")
        return best
    raise FileNotFoundError(f"Checkpoint not found: {path}")

def decimal2binary(x: torch.Tensor) -> torch.Tensor:
    x = x.to(torch.long)
    mask = (2 ** torch.arange(3, -1, -1, device=x.device, dtype=x.dtype))
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).to(torch.uint8)

def load_model_for_inference(model_path: str, device: torch.device, num_classes: int = 16) -> torch.nn.Module:
    ckpt = torch.load(model_path, map_location="cpu")

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        hparams = ckpt.get("hyper_parameters", {}) or {}
        lit = ContactLitModule(
            num_classes=hparams.get('num_classes', num_classes),
            loss=hparams.get('loss', 'focal'),
            focal_gamma=hparams.get('focal_gamma', 2.0),
            lr=hparams.get('lr', 1e-4),
            weight_decay=hparams.get('weight_decay', 1e-5),
            label_smoothing=hparams.get('label_smoothing', 0.0),
            slip_weight_factor=hparams.get('slip_weight_factor', 3.0),
        )
        lit.load_state_dict(ckpt["state_dict"], strict=True)
        lit.to(device).eval()
        return lit.model
    

    state = ckpt.get("model_state_dict", ckpt)
    model = contact_cnn_gru_progressive(num_classes=num_classes)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model


#def load_model_for_inference(model_path: str, device: torch.device, num_classes: int = 16) -> torch.nn.Module:
    #ckpt = torch.load(model_path, map_location="cpu")

    #if isinstance(ckpt, dict) and "state_dict" in ckpt:
        #hparams = ckpt.get("hyper_parameters", {}) or {}
        # Ensure we pass the required ContactLitModule parameters when loading from checkpoint
        #lit = ContactLitModule(
        #    lr=hparams.get('lr', 1e-4),
        #    weight_decay=hparams.get('weight_decay', 1e-5),
        #   loss_type_contact=hparams.get('loss_type_contact', "focal"),
        #    focal_gamma_contact=hparams.get('focal_gamma_contact', 2.0),
        #    terrain_num_classes=hparams.get('terrain_num_classes', 8),
        #    lambda_contact=hparams.get('lambda_contact', 1.0),
        #    lambda_l1=hparams.get('lambda_l1', 0.0),
        #    label_smoothing=hparams.get('label_smoothing', 0.0),
        #    slip_weight_factor=hparams.get('slip_weight_factor', 1.0) # Default to 1.0 if not in checkpoint
        #)
        #lit.load_state_dict(ckpt["state_dict"], strict=True)
        #lit.to(device).eval()
        # Return only the core model for inference
        #return lit.model

    #state = ckpt.get("model_state_dict", ckpt)
    model = contact_cnn_gru_progressive(contact_num_classes=num_classes)
    #model.load_state_dict(state, strict=True)
    #model.to(device).eval()
    #return model

@torch.no_grad()
def inference(dataloader, model, device, permutation=None):
    out_bits = torch.empty(0, 4, dtype=torch.uint8, device=device)
    model.eval()
    for sample in tqdm(dataloader, desc="Running inference"):
        x = sample['data'].to(device, non_blocking=True)
        out = model(x)
        logits = out[0] if isinstance(out, tuple) else out
        pred = logits.argmax(1)
        bits = decimal2binary(pred)
        if permutation is not None:
            bits = apply_perm(bits, permutation)
        out_bits = torch.cat((out_bits, bits), dim=0)
    return out_bits

def save2mat(pred_bin: torch.Tensor, config: dict, permutation_used=None, metrics=None, probs=None):
    data = np.load(config['data_path'])
    labels_dec = np.load(config['label_path'])
    labels_bin = decimal2binary(torch.from_numpy(labels_dec)).cpu().numpy().reshape(-1, 4)

    ws = int(config['window_size'])
    contacts_gt  = labels_bin[ws - 1:, :]
    contacts_est = pred_bin.detach().cpu().numpy() if torch.is_tensor(pred_bin) else pred_bin

    N = min(contacts_est.shape[0], contacts_gt.shape[0], data.shape[0] - (ws - 1))
    contacts_est = contacts_est[:N]
    contacts_gt  = contacts_gt[:N]

    d  = data[ws - 1: ws - 1 + N]
    q  = d[:,  0:12]; qd = d[:, 12:24]; p = d[:, 24:36]; v = d[:, 36:48]

    out = {'contacts_est': contacts_est, 'contacts_gt': contacts_gt, 'q': q, 'qd': qd, 'p': p, 'v': v}
    if permutation_used is not None:
        out['permutation_applied'] = np.array(permutation_used)
    if metrics is not None:
        out['f1_per_leg']          = np.array(metrics['per_leg']['f1_scores'])
        out['f1_avg']              = metrics['per_leg']['avg_f1']
        out['f1_macro_16class']    = metrics['16_class']['f1_macro']
        out['f1_weighted_16class'] = metrics['16_class']['f1_weighted']
        out['mean_l1_error']       = metrics['l1_error'] # Include new metric
    
    if probs is not None:
        out['contacts_prob'] = probs.detach().cpu().numpy()[:N]   # NEW: soft b(k), aligned to same N
    sio.savemat(config['mat_save_path'], out)
    print(f"Saved .mat to: {config['mat_save_path']}")


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description='Inference (locked permutation)')
    ap.add_argument('--config_name', type=str,
                    default=os.path.dirname(os.path.abspath(__file__)) + '/../config/inference.yaml')
    ap.add_argument('--save_confusion_plots', type=str, default=None)
    args = ap.parse_args()

    with open(args.config_name, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using", device)
    class_patterns = build_class_patterns(device)

    dataset = contact_dataset(
        data_path=config['data_path'],
        label_path=config['label_path'],
        window_size=config['window_size'],
        device="cpu"
    )
    dataloader = DataLoader(dataset=dataset, batch_size=config['batch_size'])
    
    ckpt_path = resolve_ckpt(config["model_load_path"])
    model = load_model_for_inference(ckpt_path, device=device, num_classes=16)
    
    # ----- Collect predictions & GT (no-perm + fixed perm) -----
    bin_pred_all = torch.empty(0, 4, dtype=torch.uint8, device=device)
    bin_gt_all   = torch.empty(0, 4, dtype=torch.uint8, device=device)

    print("\n" + "="*60)
    print("ACCURACY EVALUATION MODE")
    print("="*60)

    print("\nCollecting predictions...")
    for sample in tqdm(dataloader, desc="Inference"):
        x = (sample['data'] if isinstance(sample, dict) else sample[0]).to(device, non_blocking=True)
        y = (sample['label'] if isinstance(sample, dict) else sample[1]).to(device, non_blocking=True).long()
        out = model(x)
        logits = out[0] if isinstance(out, tuple) else out
        pred = logits.argmax(1)
        bin_pred_all = torch.cat((bin_pred_all, decimal2binary(pred)), dim=0)
        bin_gt_all   = torch.cat((bin_gt_all,   decimal2binary(y)),   dim=0)

    # Calculate L1 error for base prediction
    l1_errors_base = torch.abs(bin_pred_all.to(torch.float32) - bin_gt_all.to(torch.float32)).sum(dim=1)
    base_l1_error = l1_errors_base.mean().item()

    # Baseline (no permutation)
    base_leg_acc = (bin_pred_all == bin_gt_all).float().mean(dim=0).cpu().numpy()
    base_leg_mean = float(base_leg_acc.mean())
    base_cls_acc = (bits2decimal(bin_pred_all) == bits2decimal(bin_gt_all)).float().mean().item()
    print(f"\n[BASELINE - No Permutation]")
    print(f"  Class accuracy: {base_cls_acc:.4f}")
    print(f"  Mean L1 Error (Avg. Incorrect Legs): {base_l1_error:.4f} legs") # Highlight L1 error here
    print(f"  Per-leg accuracy: {base_leg_acc}")
    print(f"  Mean leg accuracy: {base_leg_mean:.4f}")

    print("\n[BASELINE METRICS]")
    _ = compute_f1_and_confusion(
        bin_pred_all, bin_gt_all,
        save_path=None if args.save_confusion_plots else None  # only save for fixed perm below
    )

    # ----- Fixed permutation ONLY -----
    print(f"\nApplying FIXED permutation: {FIXED_PERM}")
    pred_fixed = apply_perm(bin_pred_all, FIXED_PERM)
    
    # Calculate L1 error for fixed permutation prediction
    l1_errors_fixed = torch.abs(pred_fixed.to(torch.float32) - bin_gt_all.to(torch.float32)).sum(dim=1)
    fixed_l1_error = l1_errors_fixed.mean().item()

    fixed_leg_acc  = (pred_fixed == bin_gt_all).float().mean(dim=0).cpu().numpy()
    fixed_leg_mean = float(fixed_leg_acc.mean())
    fixed_cls_acc  = (bits2decimal(pred_fixed) == bits2decimal(bin_gt_all)).float().mean().item()

    print(f"\n[FIXED PERMUTATION {FIXED_PERM}]")
    print(f"  Class accuracy: {fixed_cls_acc:.4f}")
    print(f"  Mean L1 Error (Avg. Incorrect Legs): {fixed_l1_error:.4f} legs") # Highlight L1 error here
    print(f"  Per-leg accuracy: {fixed_leg_acc}")
    print(f"  Mean leg accuracy: {fixed_leg_mean:.4f}")

    print("\n[FIXED PERMUTATION METRICS]")
    fixed_metrics = compute_f1_and_confusion(
        pred_fixed, bin_gt_all,
        save_path=args.save_confusion_plots
    )

    # Save .mat if requested
    if bool(config.get('save_mat', False)):
        if not config.get('mat_save_path'):
            raise ValueError("save_mat=True but 'mat_save_path' missing in config.")
        print("\n" + "="*60); print("SAVING RESULTS"); print("="*60)
        save2mat(pred_fixed, config, permutation_used=FIXED_PERM, metrics=fixed_metrics)
        print("✓ Results saved successfully")

if __name__ == '__main__':
    # Due to the complex nature of the setup involving external imports (like `ContactLitModule`),
    # ensure that the definition of ContactLitModule is available at the top level for the
    # load_model_for_inference function to work correctly.
    # The original file structure is assumed to load ContactLitModule from src.lightning_contact.
    # We must ensure the necessary imports are present for the script to run standalone in its environment.
    main()

    

# import os, sys, glob, re, argparse
# ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# if ROOT not in sys.path:
#     sys.path.insert(0, ROOT)

# import numpy as np
# import yaml
# from tqdm import tqdm
# import scipy.io as sio
# import torch
# from torch.utils.data import DataLoader

# from sklearn.metrics import f1_score, confusion_matrix, classification_report
# import matplotlib.pyplot as plt
# import seaborn as sns
# from itertools import permutations

# ==== imports updated: use the multi-head model inside lightning_contact ====
# from src.contact_cnn import ContactLitModule, contact_cnn_gru_progressive
# from src.dataset import contact_dataset

# # ---------------- helpers ----------------
# def bits2decimal(bits_u8: torch.Tensor) -> torch.Tensor:
#     w = torch.tensor([8, 4, 2, 1], device=bits_u8.device, dtype=torch.int64)
#     return (bits_u8.to(torch.int64) * w).sum(dim=-1)

# def apply_perm(bits_u8: torch.Tensor, perm: tuple[int,int,int,int]) -> torch.Tensor:
#     idx = torch.tensor(perm, device=bits_u8.device)
#     return bits_u8.index_select(dim=1, index=idx)

# def compute_f1_and_confusion(bin_pred: torch.Tensor, bin_gt: torch.Tensor, save_path=None):
#     pred_np = bin_pred.cpu().numpy()
#     gt_np = bin_gt.cpu().numpy()

#     print("\n" + "="*60)
#     print("PER-LEG BINARY CLASSIFICATION METRICS")
#     print("="*60)

#     per_leg_f1, per_leg_confusion = [], []
#     for leg in range(4):
#         pred_leg = pred_np[:, leg]; gt_leg = gt_np[:, leg]
#         f1 = f1_score(gt_leg, pred_leg, average='binary', zero_division=0)
#         per_leg_f1.append(f1)
#         cm = confusion_matrix(gt_leg, pred_leg, labels=[0, 1])
#         per_leg_confusion.append(cm)
#         tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
#         prec = tp/(tp+fp) if (tp+fp)>0 else 0
#         rec  = tp/(tp+fn) if (tp+fn)>0 else 0
#         spec = tn/(tn+fp) if (tn+fp)>0 else 0
#         print(f"\nLeg {leg}: F1={f1:.4f}")
#         print(f"  TN={tn:<6} FP={fp:<6}\n  FN={fn:<6} TP={tp:<6}")
#         print(f"  Precision={prec:.4f}  Recall={rec:.4f}  Specificity={spec:.4f}")

#     avg_f1_per_leg = np.mean(per_leg_f1)
#     print(f"\nAverage F1 Score (across legs): {avg_f1_per_leg:.4f}")
#     print(f"Per-leg F1 scores: {per_leg_f1}")

#     print("\n" + "="*60)
#     print("16-CLASS CONTACT PATTERN METRICS")
#     print("="*60)

#     pred_16class = bits2decimal(bin_pred).cpu().numpy()
#     gt_16class = bits2decimal(bin_gt).cpu().numpy()

#     f1_macro = f1_score(gt_16class, pred_16class, average='macro', zero_division=0)
#     f1_micro = f1_score(gt_16class, pred_16class, average='micro', zero_division=0)
#     f1_weighted = f1_score(gt_16class, pred_16class, average='weighted', zero_division=0)
#     print(f"\n16-Class F1: Macro={f1_macro:.4f}  Micro={f1_micro:.4f}  Weighted={f1_weighted:.4f}")

#     cm_16class = confusion_matrix(gt_16class, pred_16class, labels=list(range(16)))
#     print("\nPer-Class Report:")
#     print(classification_report(
#         gt_16class, pred_16class, labels=list(range(16)),
#         target_names=[f"Class_{i:02d}" for i in range(16)], zero_division=0
#     ))

#     if save_path:
#         print(f"\nSaving confusion matrix plots to: {save_path}")
#         fig = plt.figure(figsize=(20, 10))
#         for leg in range(4):
#             ax = plt.subplot(2, 4, leg + 1)
#             sns.heatmap(per_leg_confusion[leg], annot=True, fmt='d', cmap='Blues',
#                         xticklabels=['No Contact','Contact'], yticklabels=['No Contact','Contact'],
#                         cbar=False, ax=ax)
#             ax.set_title(f'Leg {leg} - F1: {per_leg_f1[leg]:.3f}')
#             ax.set_ylabel('True'); ax.set_xlabel('Pred')
#         ax = plt.subplot(2, 2, 3)
#         sns.heatmap(cm_16class, annot=True, fmt='d', cmap='YlOrRd', ax=ax,
#                     xticklabels=range(16), yticklabels=range(16))
#         ax.set_title(f'16-Class Confusion (Macro F1 {f1_macro:.3f})')
#         ax.set_ylabel('True'); ax.set_xlabel('Pred')
#         ax = plt.subplot(2, 2, 4)
#         cm_norm = cm_16class.astype('float')/np.maximum(cm_16class.sum(axis=1, keepdims=True),1)
#         sns.heatmap(np.nan_to_num(cm_norm), annot=True, fmt='.2f', cmap='YlOrRd', ax=ax,
#                     xticklabels=range(16), yticklabels=range(16))
#         ax.set_title(f'16-Class Confusion (Norm, Weighted {f1_weighted:.3f})')
#         ax.set_ylabel('True'); ax.set_xlabel('Pred')
#         plt.tight_layout(); plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close()
#         print(f"✓ Confusion matrices saved to {save_path}")

#     return {
#         'per_leg': {'f1_scores': per_leg_f1, 'avg_f1': avg_f1_per_leg, 'confusion_matrices': per_leg_confusion},
#         '16_class': {'f1_macro': f1_macro, 'f1_micro': f1_micro, 'f1_weighted': f1_weighted, 'confusion_matrix': cm_16class}
#     }

# def compute_terrain_metrics(pred_classes: np.ndarray, gt_classes: np.ndarray, num_classes: int, save_path: str | None):
#     """Multi-class metrics + confusion for terrain."""
#     print("\n" + "="*60)
#     print("TERRAIN CLASSIFICATION METRICS")
#     print("="*60)

#     acc = (pred_classes == gt_classes).mean()
#     f1_macro = f1_score(gt_classes, pred_classes, average='macro', zero_division=0)
#     f1_micro = f1_score(gt_classes, pred_classes, average='micro', zero_division=0)
#     f1_weighted = f1_score(gt_classes, pred_classes, average='weighted', zero_division=0)
#     print(f"Accuracy={acc:.4f} | F1 Macro={f1_macro:.4f}  Micro={f1_micro:.4f}  Weighted={f1_weighted:.4f}")
#     cm = confusion_matrix(gt_classes, pred_classes, labels=list(range(num_classes)))
#     print("\nPer-Class Report:")
#     print(classification_report(gt_classes, pred_classes,
#                                 labels=list(range(num_classes)),
#                                 target_names=[f"terrain_{i}" for i in range(num_classes)],
#                                 zero_division=0))
#     if save_path:
#         plt.figure(figsize=(6,5))
#         sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
#                     xticklabels=[f"{i}" for i in range(num_classes)],
#                     yticklabels=[f"{i}" for i in range(num_classes)])
#         plt.title('Terrain Confusion'); plt.ylabel('True'); plt.xlabel('Pred')
#         base, ext = os.path.splitext(save_path)
#         out_path = f"{base}_terrain{ext or '.png'}"
#         plt.tight_layout(); plt.savefig(out_path, dpi=300, bbox_inches='tight'); plt.close()
#         print(f"✓ Terrain confusion saved to {out_path}")

#     return {
#         'acc': float(acc),
#         'f1_macro': float(f1_macro),
#         'f1_micro': float(f1_micro),
#         'f1_weighted': float(f1_weighted),
#         'confusion_matrix': cm
#     }

# def resolve_ckpt(path: str) -> str:
#     if os.path.isfile(path): return path
#     if os.path.isdir(path):
#         candidates = glob.glob(os.path.join(path, "*.ckpt"))
#         if not candidates:
#             raise FileNotFoundError(f"No .ckpt files in directory: {path}")
#         def epoch_num(p):
#             m = re.search(r"-(\d+)-", os.path.basename(p))
#             return int(m.group(1)) if m else -1
#         best = max(candidates, key=lambda p: (epoch_num(p), os.path.getmtime(p)))
#         print(f"[resolve_ckpt] Using: {best}")
#         return best
#     raise FileNotFoundError(f"Checkpoint path not found: {path}")

# def decimal2binary(x: torch.Tensor) -> torch.Tensor:
#     x = x.to(torch.long)
#     mask = (2 ** torch.arange(3, -1, -1, device=x.device, dtype=x.dtype))
#     return x.unsqueeze(-1).bitwise_and(mask).ne(0).to(torch.uint8)

# def load_model_for_inference(model_path: str, device: torch.device, num_classes: int = 16, terrain_num_classes: int | None = None) -> torch.nn.Module:
#     """
#     Loads Lightning .ckpt (preferred) or classic .pt. Returns a plain nn.Module (the multi-head model).
#     """
#     ckpt = torch.load(model_path, map_location="cpu")
#     if isinstance(ckpt, dict) and "state_dict" in ckpt:
#         lit = ContactLitModule.load_from_checkpoint(model_path, map_location="cpu")
#         model = lit.model
#         model.to(device).eval()
#         return model
#     # classic .pt
#     state = ckpt.get("model_state_dict", ckpt)
#     # terrain_num_classes fallback
#     tnc = 8 if terrain_num_classes is None else int(terrain_num_classes)
#     model = contact_cnn_gru_progressive(contact_num_classes=num_classes, terrain_num_classes=tnc)
#     model.load_state_dict(state, strict=False)
#     model.to(device).eval()
#     return model

# @torch.no_grad()
# def inference(dataloader, model, device, permutation=None, terrain_num_classes: int | None = None):
#     """
#     Returns:
#       contacts_bits: (N,4) uint8
#       terrain_pred:  (N,) int64 OR None (if model/batch has no terrain)
#       terrain_logits: (N, K) float32 OR None
#     """
#     contacts_bits_all = torch.empty(0, 4, dtype=torch.uint8, device=device)
#     terrain_pred_all  = []
#     terrain_logits_all = []

#     for sample in tqdm(dataloader, desc="Running inference"):
#         x = sample['data'].to(device, non_blocking=True)

#         out = model(x)
#         # support old contact-only models
#         if isinstance(out, tuple) and len(out) == 2:
#             contact_logits, terrain_logits = out
#         else:
#             contact_logits, terrain_logits = out, None

#         pred_contact = contact_logits.argmax(1)
#         bin_pred = decimal2binary(pred_contact)
#         if permutation is not None:
#             bin_pred = apply_perm(bin_pred, permutation)
#         contacts_bits_all = torch.cat([contacts_bits_all, bin_pred], dim=0)

#         if terrain_logits is not None:
#             t_pred = terrain_logits.argmax(1).detach().cpu().numpy()
#             terrain_pred_all.append(t_pred)
#             terrain_logits_all.append(terrain_logits.detach().cpu().numpy())

#     terrain_pred_all = None if len(terrain_pred_all)==0 else np.concatenate(terrain_pred_all, axis=0)
#     terrain_logits_all = None if len(terrain_logits_all)==0 else np.concatenate(terrain_logits_all, axis=0)

#     return contacts_bits_all, terrain_pred_all, terrain_logits_all

# def save2mat(pred_bin: torch.Tensor,
#              config: dict,
#              permutation_used=None,
#              metrics=None,
#              terrain_pred: np.ndarray | None = None,
#              terrain_logits: np.ndarray | None = None):
#     """
#     Export to .mat:
#       - contacts_est/gt (N,4) uint8
#       - terrain_est (N,) int (if available)
#       - terrain_logits (N,K) float (if available)
#       - terrain_gt (N,) int (if available in labels)
#       - q, qd, p, v slices
#       - permutation, metrics (contact + terrain)
#     """
#     data = np.load(config['data_path'])         # (T, 54)
#     labels_dec = np.load(config['label_path'])  # (T,)
#     labels_bin = decimal2binary(torch.from_numpy(labels_dec)).cpu().numpy().reshape(-1, 4)

#     # optional terrain gt path
#     terrain_gt = None
#     if 'terrain_label_path' in config and config['terrain_label_path']:
#         terrain_gt = np.load(config['terrain_label_path'])  # shape (T,)

#     ws = int(config['window_size'])
#     contacts_gt = labels_bin[ws - 1:, :]
#     contacts_est = pred_bin.detach().cpu().numpy() if torch.is_tensor(pred_bin) else pred_bin

#     N = min(contacts_est.shape[0], contacts_gt.shape[0], data.shape[0] - (ws - 1))
#     contacts_est = contacts_est[:N]
#     contacts_gt  = contacts_gt[:N]

#     d = data[ws - 1: ws - 1 + N]
#     q, qd, p, v = d[:, 0:12], d[:, 12:24], d[:, 24:36], d[:, 36:48]

#     out = {'contacts_est': contacts_est, 'contacts_gt': contacts_gt, 'q': q, 'qd': qd, 'p': p, 'v': v}

#     if permutation_used is not None:
#         out['permutation_applied'] = np.array(permutation_used)

#     # terrain alignment to same N (dataset should align terrain labels to window end as well)
#     if terrain_pred is not None:
#         out['terrain_est'] = terrain_pred[:N]
#     if terrain_logits is not None:
#         out['terrain_logits'] = terrain_logits[:N]
#     if terrain_gt is not None:
#         out['terrain_gt'] = terrain_gt[ws - 1: ws - 1 + N]

#     if metrics is not None:
#         # contact metrics
#         if 'per_leg' in metrics and '16_class' in metrics:
#             out['f1_per_leg'] = np.array(metrics['per_leg']['f1_scores'])
#             out['f1_avg'] = metrics['per_leg']['avg_f1']
#             out['f1_macro_16class'] = metrics['16_class']['f1_macro']
#             out['f1_weighted_16class'] = metrics['16_class']['f1_weighted']
#         # terrain metrics
#         if 'terrain' in metrics:
#             t = metrics['terrain']
#             out['terrain_acc'] = np.array(t['acc'])
#             out['terrain_f1_macro'] = np.array(t['f1_macro'])
#             out['terrain_f1_weighted'] = np.array(t['f1_weighted'])
#             out['terrain_confusion'] = t['confusion_matrix']

#     sio.savemat(config['mat_save_path'], out)
#     print(f"Saved .mat to: {config['mat_save_path']}")

# # ---------------- main ----------------
# def main():
#     parser = argparse.ArgumentParser(description='Inference for contact+terrain network')
#     parser.add_argument('--config_name', type=str,
#                         default=os.path.dirname(os.path.abspath(__file__)) + '/../config/inference.yaml')
#     parser.add_argument('--use_best_permutation', action='store_true')
#     parser.add_argument('--fixed_permutation', type=str, default=None,
#                         help='Comma-separated permutation to apply, e.g., "2,0,3,1"')
#     parser.add_argument('--save_confusion_plots', type=str, default=None,
#                         help='Base path to save contact confusion matrices (adds _terrain for terrain)')
#     args = parser.parse_args()

#     with open(args.config_name, 'r') as f:
#         config = yaml.safe_load(f)

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print("Using", device)

#     dataset = contact_dataset(
#         data_path=config['data_path'],
#         label_path=config['label_path'],
#         window_size=config['window_size'],
#         device="cpu"
#     )
#     dataloader = DataLoader(dataset=dataset, batch_size=config['batch_size'])

#     terrain_num_classes = int(config.get('terrain_num_classes', 0) or 0)

#     ckpt_path = resolve_ckpt(config["model_load_path"])
#     model = load_model_for_inference(ckpt_path, device=device, num_classes=16,
#                                      terrain_num_classes=terrain_num_classes if terrain_num_classes>0 else None)

#     fixed_perm = None
#     if args.fixed_permutation:
#         fixed_perm = tuple(map(int, args.fixed_permutation.split(',')))
#         print(f"Using fixed permutation: {fixed_perm}")

#     pred_bin = None
#     best_perm = None
#     metrics = {}

#     if bool(config.get('calculate_accuracy', True)):
#         print("\n" + "="*60)
#         print("ACCURACY EVALUATION MODE")
#         print("="*60)

#         bin_pred_all = torch.empty(0, 4, dtype=torch.uint8, device=device)
#         bin_gt_all   = torch.empty(0, 4, dtype=torch.uint8, device=device)

#         # Terrain collectors
#         terrain_pred_all = []
#         terrain_logits_all = []
#         terrain_gt_all = []

#         print("\nCollecting predictions...")
#         for sample in tqdm(dataloader, desc="Inference"):
#             x = sample['data'].to(device, non_blocking=True)
#             y = sample['label'].to(device, non_blocking=True).long()

#             out = model(x)
#             if isinstance(out, tuple) and len(out)==2:
#                 contact_logits, terrain_logits = out
#             else:
#                 contact_logits, terrain_logits = out, None

#             pred_contact = contact_logits.argmax(1)
#             bin_pred = decimal2binary(pred_contact)
#             bin_gt   = decimal2binary(y)

#             bin_pred_all = torch.cat((bin_pred_all, bin_pred), dim=0)
#             bin_gt_all   = torch.cat((bin_gt_all,   bin_gt),   dim=0)

#             # terrain (only if head exists and dataset provides labels)
#             if (terrain_logits is not None) and ('terrain' in sample):
#                 t_gt = sample['terrain'].cpu().numpy().astype(np.int64)
#                 t_pred = terrain_logits.argmax(1).cpu().numpy().astype(np.int64)
#                 terrain_pred_all.append(t_pred)
#                 terrain_logits_all.append(terrain_logits.detach().cpu().numpy())
#                 terrain_gt_all.append(t_gt)

#         # CONTACT: baseline metrics
#         base_leg_acc = (bin_pred_all == bin_gt_all).float().mean(dim=0).cpu().numpy()
#         base_leg_mean = float(base_leg_acc.mean())
#         base_cls_acc = (bits2decimal(bin_pred_all) == bits2decimal(bin_gt_all)).float().mean().item()
#         print(f"\n[BASELINE - No Permutation]")
#         print(f"  Class accuracy: {base_cls_acc:.4f}")
#         print(f"  Per-leg accuracy: {base_leg_acc}")
#         print(f"  Mean leg accuracy: {base_leg_mean:.4f}")

#         print("\n[BASELINE METRICS]")
#         baseline_metrics = compute_f1_and_confusion(
#             bin_pred_all, bin_gt_all,
#             save_path=args.save_confusion_plots if not (args.use_best_permutation or fixed_perm) else None
#         )

#         # CONTACT: fixed perm
#         if fixed_perm:
#             pred_fixed = apply_perm(bin_pred_all, fixed_perm)
#             fixed_metrics = compute_f1_and_confusion(
#                 pred_fixed, bin_gt_all,
#                 save_path=args.save_confusion_plots if not args.use_best_permutation else None
#             )

#         # CONTACT: sweep
#         print("\nRunning permutation sweep...")
#         best = sweep_permutations(bin_pred_all, bin_gt_all)
#         best_perm = best["perm"]
#         print(f"\n[BEST PERMUTATION FROM SWEEP: {best_perm}]")
#         print(f"  Class accuracy: {best['acc_cls']:.4f}")
#         print(f"  Per-leg accuracy: {best['acc_leg']}")
#         print(f"  Mean leg accuracy: {best['acc_leg_mean']:.4f}")
#         print(f"  Per-leg F1 scores: {best['f1_per_leg']}")
#         print(f"  Average F1: {best['f1_avg']:.4f}")
#         print(f"  16-class Macro F1: {best['f1_macro_16class']:.4f}")

#         # choose
#         if args.use_best_permutation:
#             print(f"\n✓ Using best permutation {best_perm} for outputs")
#             pred_bin = apply_perm(bin_pred_all, best_perm)
#             contact_metrics = compute_f1_and_confusion(pred_bin, bin_gt_all, save_path=args.save_confusion_plots)
#         elif fixed_perm:
#             print(f"\n✓ Using fixed permutation {fixed_perm} for outputs")
#             pred_bin = apply_perm(bin_pred_all, fixed_perm)
#             contact_metrics = fixed_metrics
#             best_perm = fixed_perm
#         else:
#             print(f"\n✓ Using raw predictions (no permutation) for outputs")
#             pred_bin = bin_pred_all
#             contact_metrics = baseline_metrics

#         metrics.update(contact_metrics)

#         # ===== TERRAIN: metrics if collected =====
#         if len(terrain_pred_all) > 0 and len(terrain_gt_all) > 0:
#             t_pred = np.concatenate(terrain_pred_all, axis=0)
#             t_gt   = np.concatenate(terrain_gt_all, axis=0)
#             t_metrics = compute_terrain_metrics(
#                 pred_classes=t_pred,
#                 gt_classes=t_gt,
#                 num_classes=int(config.get('terrain_num_classes', max(int(t_pred.max())+1, 1))),
#                 save_path=args.save_confusion_plots
#             )
#             metrics['terrain'] = t_metrics
#             terrain_pred_concat = t_pred
#             terrain_logits_concat = np.concatenate(terrain_logits_all, axis=0) if len(terrain_logits_all)>0 else None
#         else:
#             terrain_pred_concat = None
#             terrain_logits_concat = None

#     else:
#         print("\n" + "="*60)
#         print("INFERENCE-ONLY MODE (no accuracy calculation)")
#         print("="*60)
#         perm_to_use = fixed_perm if fixed_perm else None
#         pred_bin, terrain_pred_concat, terrain_logits_concat = inference(
#             dataloader, model, device, permutation=perm_to_use,
#             terrain_num_classes=int(config.get('terrain_num_classes', 0) or 0)
#         )
#         best_perm = perm_to_use

#     # ----- save .mat -----
#     if bool(config.get('save_mat', False)):
#         if not config.get('mat_save_path'):
#             raise ValueError("save_mat=True but 'mat_save_path' not provided in config.")
#         print("\n" + "="*60); print("SAVING RESULTS"); print("="*60)
#         save2mat(pred_bin, config,
#                  permutation_used=best_perm,
#                  metrics=metrics,
#                  terrain_pred=terrain_pred_concat,
#                  terrain_logits=terrain_logits_concat)
#         print("✓ Results saved successfully")

# if __name__ == '__main__':
#     main()
