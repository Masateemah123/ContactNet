# src/test_lightning.py
import os, argparse, yaml
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import precision_score, jaccard_score, confusion_matrix

# package-absolute imports
from src.contact_cnn import ContactLitModule
from src.contact_cnn import contact_cnn_gru_progressive
from src.dataset import contact_dataset


# ----------------- helpers -----------------

def decimal2binary(x: torch.Tensor) -> torch.Tensor:
    """Map class index [0..15] -> 4-bit contacts; returns uint8 (N,4)."""
    x = x.to(torch.long)
    mask = (2 ** torch.arange(3, -1, -1, device=x.device, dtype=x.dtype))
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).to(torch.uint8)


def load_model_for_inference(model_load_path: str, device: torch.device, num_classes: int = 16) -> nn.Module:
    """
    Load either a Lightning .ckpt (with 'state_dict') or a classic .pt (with 'model_state_dict').
    Returns a plain nn.Module ready for forward().
    """
    ckpt = torch.load(model_load_path, map_location="cpu")

    # Case A: Lightning checkpoint
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        lit = ContactLitModule.load_from_checkpoint(model_load_path, map_location="cpu")
        model = lit.model
        model.to(device).eval()
        return model

    # Case B: Classic .pt (earlier trainer)
    state = ckpt.get("model_state_dict", None)
    if state is None and isinstance(ckpt, dict):
        # maybe the dict itself IS the state_dict
        state = ckpt
    if state is None:
        raise ValueError(f"Unrecognized checkpoint format: {model_load_path}")

    model = contact_cnn_gru_progressive(num_classes=num_classes)
    incompatible = model.load_state_dict(state, strict=False)  # tolerate minor key diffs
    missing = getattr(incompatible, "missing_keys", [])
    unexpected = getattr(incompatible, "unexpected_keys", [])
    if missing or unexpected:
        print("[load_model_for_inference] Missing keys:", missing)
        print("[load_model_for_inference] Unexpected keys:", unexpected)
    model.to(device).eval()
    return model


def compute_confusion_mat(bin_contact_pred_arr: np.ndarray, bin_contact_gt_arr: np.ndarray):
    cm = {}
    cm['leg_rf'] = confusion_matrix(bin_contact_gt_arr[:, 0], bin_contact_pred_arr[:, 0], labels=[0, 1])
    cm['leg_lf'] = confusion_matrix(bin_contact_gt_arr[:, 1], bin_contact_pred_arr[:, 1], labels=[0, 1])
    cm['leg_rh'] = confusion_matrix(bin_contact_gt_arr[:, 2], bin_contact_pred_arr[:, 2], labels=[0, 1])
    cm['leg_lh'] = confusion_matrix(bin_contact_gt_arr[:, 3], bin_contact_pred_arr[:, 3], labels=[0, 1])
    cm['total']  = cm['leg_rf'] + cm['leg_lf'] + cm['leg_rh'] + cm['leg_lh']
    cm['total_ratio'] = cm['total'] / max(1, np.sum(cm['total']))

    def rates(mat):
        TN, FP, FN, TP = mat[0,0], mat[0,1], mat[1,0], mat[1,1]
        fn = FN / (FN + TP) if (FN + TP) > 0 else np.nan
        fp = FP / (FP + TN) if (FP + TN) > 0 else np.nan
        return fn, fp

    fn_rate, fp_rate = {}, {}
    for k in ['leg_rf', 'leg_lf', 'leg_rh', 'leg_lh']:
        fn_rate[k], fp_rate[k] = rates(cm[k])
    fn_rate['total'], fp_rate['total'] = rates(cm['total'])
    return cm, fn_rate, fp_rate


def compute_precision(bin_pred_arr, bin_gt_arr, pred_arr, gt_arr):
    precision_of_class    = precision_score(gt_arr, pred_arr, average='weighted', zero_division=0)
    precision_of_all_legs = precision_score(bin_gt_arr.flatten(), bin_pred_arr.flatten(), zero_division=0)
    precision_of_legs     = [precision_score(bin_gt_arr[:, i], bin_pred_arr[:, i], zero_division=0) for i in range(4)]
    return precision_of_class, precision_of_legs, precision_of_all_legs


def compute_jaccard(bin_pred_arr, bin_gt_arr, pred_arr, gt_arr):
    jaccard_of_class    = jaccard_score(gt_arr, pred_arr, average='weighted', zero_division=0)
    jaccard_of_all_legs = jaccard_score(bin_gt_arr.flatten(), bin_pred_arr.flatten(), zero_division=0)
    jaccard_of_legs     = [jaccard_score(bin_gt_arr[:, i], bin_pred_arr[:, i], zero_division=0) for i in range(4)]
    return jaccard_of_class, jaccard_of_legs, jaccard_of_all_legs


@torch.no_grad()
def compute_accuracy(dataloader, model: nn.Module, device: torch.device):
    num_correct = 0
    num_data = 0
    correct_per_leg = np.zeros(4, dtype=np.float64)

    bin_pred_arr = np.zeros((0, 4), dtype=np.uint8)
    bin_gt_arr   = np.zeros((0, 4), dtype=np.uint8)
    pred_arr     = np.zeros((0,), dtype=np.int64)
    gt_arr       = np.zeros((0,), dtype=np.int64)

    model.eval()
    for sample in tqdm(dataloader):
        x = (sample['data'] if isinstance(sample, dict) else sample[0]).to(device, non_blocking=True)
        y = (sample['label'] if isinstance(sample, dict) else sample[1]).to(device, non_blocking=True).long()

        out = model(x)
        logits = out[0] if isinstance(out, tuple) else out
        pred   = logits.argmax(dim=1)

        bin_pred = decimal2binary(pred)
        bin_gt   = decimal2binary(y)

        bin_pred_arr = np.vstack((bin_pred_arr, bin_pred.cpu().numpy()))
        bin_gt_arr   = np.vstack((bin_gt_arr,   bin_gt.cpu().numpy()))
        pred_arr     = np.hstack((pred_arr, pred.cpu().numpy()))
        gt_arr       = np.hstack((gt_arr,   y.cpu().numpy()))

        correct_per_leg += (bin_pred == bin_gt).sum(dim=0).cpu().numpy()
        num_data += x.size(0)
        num_correct += (pred == y).sum().item()

    return num_correct / max(1, num_data), correct_per_leg / max(1, num_data), bin_pred_arr, bin_gt_arr, pred_arr, gt_arr


# ----------------- main -----------------

def main():
    parser = argparse.ArgumentParser(description='Test the contact network (Lightning-compatible)')
    parser.add_argument('--config_name', type=str,
                        default=os.path.dirname(os.path.abspath(__file__)) + '/../config/test_params.yaml')
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config_name))

    device = torch.device('cuda' if (torch.cuda.is_available() and cfg.get('use_gpu', True)) else 'cpu')
    print('Using', device)

    # dataset
    test_ds = contact_dataset(
        data_path  = os.path.join(cfg['data_folder'], "test.npy"),
        label_path = os.path.join(cfg['data_folder'], "test_label.npy"),
        window_size= cfg['window_size'],
        device     = "cpu"   # keep CPU; we move to device in the loop
    )
    test_ld = DataLoader(
        dataset=test_ds,
        batch_size=cfg['batch_size'],
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 4)),
        pin_memory=True
    )

    # model (Lightning .ckpt or classic .pt)
    model = load_model_for_inference(cfg['model_load_path'], device=device, num_classes=16)

    # evaluate
    test_acc, acc_per_leg, bin_pred_arr, bin_gt_arr, pred_arr, gt_arr = compute_accuracy(test_ld, model, device)
    precision_of_class, precision_of_legs, precision_of_all_legs = compute_precision(bin_pred_arr, bin_gt_arr, pred_arr, gt_arr)
    jaccard_of_class, jaccard_of_legs, jaccard_of_all_legs       = compute_jaccard(bin_pred_arr, bin_gt_arr, pred_arr, gt_arr)
    confusion_mat, fn_rate, fp_rate                               = compute_confusion_mat(bin_pred_arr, bin_gt_arr)

    # report
    print(f"Test accuracy (class): {test_acc:.4f}")
    for i, name in enumerate(["leg 0 (rf)", "leg 1 (lf)", "leg 2 (rh)", "leg 3 (lh)"]):
        print(f"Accuracy of {name}: {acc_per_leg[i]:.4f}")
    print(f"Accuracy (per-leg mean): {float(np.mean(acc_per_leg)):.4f}")
    print("---------------")
    print(f"Precision (class): {precision_of_class:.4f}")
    for i, p in enumerate(precision_of_legs):
        print(f"Precision leg {i}: {p:.4f}")
    print(f"Precision (all legs, flattened): {precision_of_all_legs:.4f}")
    print("---------------")
    print(f"Jaccard (class): {jaccard_of_class:.4f}")
    for i, j in enumerate(jaccard_of_legs):
        print(f"Jaccard leg {i}: {j:.4f}")
    print(f"Jaccard (all legs, flattened): {jaccard_of_all_legs:.4f}")
    print("---------------")
    print("Confusion matrices per leg:")
    for k in ['leg_rf','leg_lf','leg_rh','leg_lh','total']:
        print(k, "=\n", confusion_mat[k])
    print("Total ratio =\n", confusion_mat['total_ratio'])
    print("---------------")
    for k in ['leg_rf','leg_lf','leg_rh','leg_lh','total']:
        print(f"FN rate {k}: {fn_rate[k]:.4f}")
    print("---------------")
    for k in ['leg_rf','leg_lf','leg_rh','leg_lh','total']:
        print(f"FP rate {k}: {fp_rate[k]:.4f}")


if __name__ == '__main__':
    main()
