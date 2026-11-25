#!/usr/bin/env python3
"""
训练 Vernier 超轻量装甲板数字分类 MLP（28x20 -> 64 -> 32 -> 9），
并导出为 Vernier 二进制权重格式（VRMLP001，按 [in_dim][out_dim] 行主序存放以适配 CUDA 合并访存）。

数据集目录结构：
    dataset/
      0/  (负样本 / 非装甲板)
      1/  (1 号英雄)
      2/  (2 号工程)
      3/  (3 号步兵)
      4/  (4 号步兵)
      5/  (5 号步兵)
      6/  (前哨站)
      7/  (哨兵)
      8/  (基地)
"""

import argparse
import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

INPUT_W = 20
INPUT_H = 28
INPUT_DIM = INPUT_W * INPUT_H  # 560
HIDDEN1 = 64
HIDDEN2 = 32
NUM_CLASSES = 9


class VernierMlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(INPUT_DIM, HIDDEN1)
        self.fc2 = nn.Linear(HIDDEN1, HIDDEN2)
        self.fc3 = nn.Linear(HIDDEN2, NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def export_vrmlp(model: VernierMlp, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        # 文件头：magic[8], input_w, input_h, hidden1, hidden2, num_classes, reserved
        f.write(
            struct.pack(
                "<8s6I", b"VRMLP001", INPUT_W, INPUT_H, HIDDEN1, HIDDEN2, NUM_CLASSES, 0
            )
        )
        # PyTorch Linear 权重形状为 [out_dim, in_dim]；转置为 [in_dim, out_dim]，
        # 使 CUDA 线程（tid -> out_dim）对全局显存达到 100% 合并访存
        for layer in (model.fc1, model.fc2, model.fc3):
            w_t = layer.weight.detach().cpu().numpy().T.astype(np.float32, copy=False)
            b = layer.bias.detach().cpu().numpy().astype(np.float32, copy=False)
            f.write(np.ascontiguousarray(w_t).tobytes())
            f.write(np.ascontiguousarray(b).tobytes())
    print(f"Exported Vernier MLP weights to {out_path} ({out_path.stat().st_size} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and export Vernier 28x20 MLP")
    parser.add_argument("--data", type=Path, required=True, help="ImageFolder dataset root")
    parser.add_argument("--out", type=Path, default=Path("models/armor_mlp.bin"))
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    tfm = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((INPUT_H, INPUT_W)),
            transforms.ToTensor(),
        ]
    )

    ds = datasets.ImageFolder(root=str(args.data), transform=tfm)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VernierMlp().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            opt.step()
            total_loss += loss.item() * imgs.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += imgs.size(0)
        print(f"Epoch {epoch:02d}/{args.epochs} | loss={total_loss/total:.4f} | acc={correct/total:.4f}")

    export_vrmlp(model, args.out)


if __name__ == "__main__":
    main()
