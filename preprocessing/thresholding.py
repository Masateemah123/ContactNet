# -*- coding: utf-8 -*-
"""
Ground-Truth Foot-Contact Conditioning Pipeline
Otsu-Guided Threshold Analysis + Dual-Threshold Hysteresis + Minimum-Duration Filtering
"""

import argparse
import bagpy
from bagpy import bagreader
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def otsu_threshold(data: np.ndarray) -> float:
    """Compute optimal bimodal split threshold via Otsu's between-class variance."""
    hist, bin_edges = np.histogram(data, bins=256)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    hist = hist.astype(float) / hist.sum()

    max_variance = 0.0
    best_threshold = float(bin_centers[0])

    for i in range(1, len(hist) - 1):
        w0 = hist[:i].sum()
        w1 = hist[i:].sum()
        if w0 == 0 or w1 == 0:
            continue

        mu0 = (hist[:i] * bin_centers[:i]).sum() / w0
        mu1 = (hist[i:] * bin_centers[i:]).sum() / w1
        variance = w0 * w1 * (mu0 - mu1) ** 2

        if variance > max_variance:
            max_variance = variance
            best_threshold = float(bin_centers[i])

    return best_threshold


def analyze_contact_thresholds(data: pd.Series, col_name: str) -> dict:
    """Compute statistical and Otsu priors for per-leg raw contact sensor calibration."""
    mean_val = float(np.mean(data))
    std_val = float(np.std(data))
    median_val = float(np.median(data))
    otsu_thresh = otsu_threshold(data.to_numpy())

    print(f"\n=== Threshold Analysis for {col_name} ===")
    print(f"Mean: {mean_val:.1f} | Median: {median_val:.1f} | Std: {std_val:.1f}")
    print(f"Mean + 0.5*Std: {mean_val + 0.5 * std_val:.1f} | Mean + 1*Std: {mean_val + std_val:.1f}")
    print(f"Otsu threshold: {otsu_thresh:.1f} | Range: [{np.min(data):.1f}, {np.max(data):.1f}]")

    return {
        "mean": mean_val,
        "median": median_val,
        "std": std_val,
        "mean_plus_half_std": mean_val + 0.5 * std_val,
        "mean_plus_std": mean_val + std_val,
        "otsu": otsu_thresh,
    }


def detect_contacts_threshold(
    data: np.ndarray,
    threshold: float,
    hysteresis_factor: float = 0.9,
    min_contact_duration: float = 0.05,
    sampling_rate_hz: int = 500,
) -> np.ndarray:
    """
    Apply dual-threshold hysteresis and minimum-duration filtering to suppress
    pneumatic foot-sensor vibration chatter and transient swing dropouts.
    """
    upper_thresh = threshold
    lower_thresh = threshold * hysteresis_factor

    contacts = np.zeros(len(data), dtype=bool)
    current_state = False

    for i in range(len(data)):
        if not current_state and data[i] > upper_thresh:
            current_state = True
        elif current_state and data[i] < lower_thresh:
            current_state = False
        contacts[i] = current_state

    min_samples = int(min_contact_duration * sampling_rate_hz)
    contact_diff = np.diff(contacts.astype(int))
    contact_starts = np.where(contact_diff == 1)[0] + 1
    contact_ends = np.where(contact_diff == -1)[0] + 1

    if contacts[0]:
        contact_starts = np.insert(contact_starts, 0, 0)
    if contacts[-1]:
        contact_ends = np.append(contact_ends, len(contacts))

    ls, le = None, None
    for start, end in zip(contact_starts, contact_ends):
        if ls is not None and le is not None:
            if start - le < min_samples:
                contacts[le:start] = True
        if end - start < min_samples:
            ls, le = start, end

    return contacts


def main():
    parser = argparse.ArgumentParser(description="Condition raw Unitree foot-contact signals from ROS bag.")
    parser.add_argument("--bag_file", type=str, default="data/raw_bags/trial.bag", help="Path to input ROS1 .bag file")
    parser.add_argument("--topic", type=str, default="/unitree/raw_contact", help="Raw contact ROS topic")
    parser.add_argument("--output_csv", type=str, default="improved_contact_labels.csv", help="Output CSV filename")
    parser.add_argument("--save_plot", type=str, default="figures/conditioned_contact_gait_cycles.png", help="Output diagnostic plot path")
    parser.add_argument("--hysteresis", type=float, default=0.9, help="Lower threshold multiplier (default: 0.9)")
    parser.add_argument("--min_duration", type=float, default=0.05, help="Minimum stance/swing duration in seconds")
    parser.add_argument("--fs", type=int, default=500, help="Raw contact topic sampling rate in Hz")
    args = parser.parse_args()

    bag = bagreader(args.bag_file)
    csvfile_raw = bag.message_by_topic(args.topic)
    df_raw = pd.read_csv(csvfile_raw)

    contact_cols = ["raw_contacts_0", "raw_contacts_1", "raw_contacts_2", "raw_contacts_3"]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    for col in contact_cols:
        analyze_contact_thresholds(df_raw[col], col)

    # Calibrated per-leg thresholds derived from statistical & Otsu prior analysis
    default_thresholds = {
        "raw_contacts_0": 200.0,
        "raw_contacts_1": 350.0,
        "raw_contacts_2": 150.0,
        "raw_contacts_3": 350.0,
    }

    detected_contacts = {}
    for col in contact_cols:
        thresh = default_thresholds[col]
        detected_contacts[col] = detect_contacts_threshold(
            df_raw[col].to_numpy(),
            threshold=thresh,
            hysteresis_factor=args.hysteresis,
            min_contact_duration=args.min_duration,
            sampling_rate_hz=args.fs,
        )
        pct = np.mean(detected_contacts[col]) * 100.0
        print(f"{col}: Upper={thresh:.1f}, Lower={thresh * args.hysteresis:.1f}, Stance={pct:.1f}%")

    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True, dpi=200)
    for i, col in enumerate(contact_cols):
        axes[i].plot(df_raw["Time"], df_raw[col], color=colors[i], alpha=0.75, linewidth=1.0, label="Raw Signal")
        axes[i].axhline(y=default_thresholds[col], color="red", linestyle="--", alpha=0.85, label=f"Upper Threshold ({default_thresholds[col]:.0f})")
        axes[i].axhline(y=default_thresholds[col] * args.hysteresis, color="darkred", linestyle=":", alpha=0.85, label=f"Lower Hysteresis ({default_thresholds[col] * args.hysteresis:.0f})")

        ax2 = axes[i].twinx()
        ax2.fill_between(df_raw["Time"], 0, detected_contacts[col].astype(int), alpha=0.22, color="green", label="Conditioned Contact")
        ax2.set_ylim(-0.1, 1.3)
        ax2.set_ylabel("Contact State", fontsize=10)

        axes[i].set_ylabel("Raw Value [a.u.]", fontsize=10)
        axes[i].set_title(f"{col} — Hysteresis & Duration-Conditioned Contact Detection", fontsize=11, fontweight="bold")
        axes[i].grid(True, alpha=0.3)

        lines1, labels1 = axes[i].get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        axes[i].legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Time [s]", fontsize=11)
    plt.tight_layout()
    plt.savefig(args.save_plot, bbox_inches="tight")
    print(f"Saved diagnostic plot to {args.save_plot}")

    contact_df = df_raw[["Time"] + contact_cols].copy()
    for col in contact_cols:
        contact_df[f"{col}_contact_threshold"] = detected_contacts[col].astype(int)

    contact_df.to_csv(args.output_csv, index=False)
    print(f"Saved conditioned contact labels to {args.output_csv} (shape={contact_df.shape})")


if __name__ == "__main__":
    main()
