"""
Train and evaluate a relative-XYZ regressor from raw 10-channel waveforms.
"""

from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from relative_xyz_common import (
    DATASET_DIR,
    MODEL_DIR,
    RelativeXYZNet,
    baseline_from_anchor,
    metric_dict,
    save_json,
    set_torch_determinism,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default=DATASET_DIR)
    parser.add_argument("--out_dir", default=MODEL_DIR)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None,
                        help="cpu | mps | cuda. Default = auto.")
    return parser.parse_args()


class RelativeXYZDataset(Dataset):
    def __init__(self, dataset_dir: str, indices: np.ndarray, waveform_scale: float):
        self.waveforms = np.load(os.path.join(dataset_dir, "waveforms.npy"), mmap_mode="r")
        self.anchors = np.load(os.path.join(dataset_dir, "anchors.npy"), mmap_mode="r")
        self.targets_relative = np.load(os.path.join(dataset_dir, "targets_relative.npy"), mmap_mode="r")
        self.targets_global = np.load(os.path.join(dataset_dir, "targets_global.npy"), mmap_mode="r")
        self.indices = np.asarray(indices, dtype=np.int64)
        self.waveform_scale = float(waveform_scale)

    def __len__(self):
        return self.indices.size

    def __getitem__(self, item):
        idx = int(self.indices[item])
        waveform = (np.asarray(self.waveforms[idx], dtype=np.float32) / self.waveform_scale).copy()
        target_rel = np.asarray(self.targets_relative[idx], dtype=np.float32).copy()
        anchor = np.asarray(self.anchors[idx], dtype=np.float32).copy()
        target_global = np.asarray(self.targets_global[idx], dtype=np.float32).copy()
        return (
            torch.from_numpy(waveform),
            torch.from_numpy(target_rel),
            torch.from_numpy(anchor),
            torch.from_numpy(target_global),
        )


def load_manifest(dataset_dir: str):
    with open(os.path.join(dataset_dir, "manifest.json"), "r") as handle:
        return json.load(handle)


def split_indices(dataset_dir: str):
    split_ids = np.load(os.path.join(dataset_dir, "split.npy"))
    return {
        "train": np.flatnonzero(split_ids == 0).astype(np.int64),
        "val": np.flatnonzero(split_ids == 1).astype(np.int64),
        "test": np.flatnonzero(split_ids == 2).astype(np.int64),
    }


def evaluate_model(model, loader, device):
    model.eval()
    sumsq_axis = np.zeros(3, dtype=np.float64)
    sumsq_3d = 0.0
    count = 0

    with torch.no_grad():
        for waveforms, _, anchors, target_global in loader:
            waveforms = waveforms.to(device)
            pred_rel = model(waveforms).cpu()
            pred_global = pred_rel + anchors
            error = pred_global - target_global
            err_np = error.numpy().astype(np.float64, copy=False)
            sumsq_axis += np.square(err_np).sum(axis=0)
            sumsq_3d += np.square(err_np).sum()
            count += err_np.shape[0]

    axis_rmse = np.sqrt(sumsq_axis / max(count, 1))
    return {
        "rmse_x": float(axis_rmse[0]),
        "rmse_y": float(axis_rmse[1]),
        "rmse_z": float(axis_rmse[2]),
        "rmse_xyz": float(np.sqrt(sumsq_3d / max(count, 1))),
    }


def evaluate_anchor_baseline(dataset_dir: str, indices: np.ndarray):
    anchors = np.load(os.path.join(dataset_dir, "anchors.npy"), mmap_mode="r")[indices]
    targets_global = np.load(os.path.join(dataset_dir, "targets_global.npy"), mmap_mode="r")[indices]
    return baseline_from_anchor(np.asarray(anchors), np.asarray(targets_global))


def write_history_csv(path: str, rows):
    fieldnames = ["epoch", "train_loss", "val_rmse_x", "val_rmse_y", "val_rmse_z", "val_rmse_xyz"]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_metrics_csv(path: str, payload):
    fieldnames = ["model", "split", "rmse_x", "rmse_y", "rmse_z", "rmse_xyz"]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for model_name, split_metrics in payload.items():
            for split_name, metrics in split_metrics.items():
                row = {"model": model_name, "split": split_name}
                row.update(metrics)
                writer.writerow(row)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    set_torch_determinism(args.seed)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))

    manifest = load_manifest(args.dataset_dir)
    split_idx = split_indices(args.dataset_dir)
    waveform_scale = float(manifest["waveform_scale"])

    datasets = {
        split_name: RelativeXYZDataset(args.dataset_dir, idx, waveform_scale)
        for split_name, idx in split_idx.items()
    }
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        ),
    }
    eval_loaders = {
        split_name: DataLoader(
            datasets[split_name],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        for split_name in ["train", "val", "test"]
    }

    if args.device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    model = RelativeXYZNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()

    history_rows = []
    best_val_rmse = float("inf")
    epochs_without_improvement = 0
    checkpoint_path = os.path.join(args.out_dir, "best_model.pt")

    print(f"Training on {device} with waveform scale {waveform_scale:.6f}")
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for waveforms, target_rel, _, _ in loaders["train"]:
            waveforms = waveforms.to(device)
            target_rel = target_rel.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred_rel = model(waveforms)
            loss = loss_fn(pred_rel, target_rel)
            loss.backward()
            optimizer.step()

            batch_size = waveforms.shape[0]
            train_loss_sum += float(loss.item()) * batch_size
            train_count += batch_size

        train_loss = train_loss_sum / max(train_count, 1)
        val_metrics = evaluate_model(model, loaders["val"], device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_rmse_x": val_metrics["rmse_x"],
            "val_rmse_y": val_metrics["rmse_y"],
            "val_rmse_z": val_metrics["rmse_z"],
            "val_rmse_xyz": val_metrics["rmse_xyz"],
        }
        history_rows.append(row)

        print(
            f"epoch {epoch:02d}  train_loss={train_loss:.6f}  "
            f"val_rmse_xyz={val_metrics['rmse_xyz']:.4f}  "
            f"(x={val_metrics['rmse_x']:.4f}, y={val_metrics['rmse_y']:.4f}, z={val_metrics['rmse_z']:.4f})"
        )

        if val_metrics["rmse_xyz"] < best_val_rmse - 1e-6:
            best_val_rmse = val_metrics["rmse_xyz"]
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "waveform_scale": waveform_scale,
                    "config": vars(args),
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping after epoch {epoch}")
                break

    if not os.path.exists(checkpoint_path):
        raise RuntimeError("Training finished without saving a checkpoint")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    nn_metrics = {
        split_name: evaluate_model(model, eval_loaders[split_name], device)
        for split_name in ["train", "val", "test"]
    }
    baseline_metrics = {
        split_name: evaluate_anchor_baseline(args.dataset_dir, split_idx[split_name])
        for split_name in ["train", "val", "test"]
    }

    payload = {
        "config": vars(args),
        "dataset_manifest": manifest,
        "best_epoch": int(checkpoint["epoch"]),
        "best_val_metrics": checkpoint["val_metrics"],
        "nn_metrics": nn_metrics,
        "anchor_baseline_metrics": baseline_metrics,
    }

    save_json(os.path.join(args.out_dir, "metrics.json"), payload)
    write_history_csv(os.path.join(args.out_dir, "history.csv"), history_rows)
    write_metrics_csv(
        os.path.join(args.out_dir, "metrics.csv"),
        {"nn": nn_metrics, "anchor_baseline": baseline_metrics},
    )

    print("Final NN metrics:")
    for split_name, metrics in nn_metrics.items():
        print(f"  {split_name}: {metrics}")
    print("Anchor baseline:")
    for split_name, metrics in baseline_metrics.items():
        print(f"  {split_name}: {metrics}")


if __name__ == "__main__":
    main()
