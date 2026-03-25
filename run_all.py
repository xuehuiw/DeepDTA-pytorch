import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import json
from tqdm import tqdm

from source.utils.data_loader import DeepDTADataset
from source.models.deepdta_model import DeepDTA

# ===================== 论文完整配置 =====================
MAX_SMI_LEN = 100
MAX_PRO_LEN = 1000
EMBED_DIM = 128
NUM_FILTERS = 32
BATCH_SIZE = 256
EPOCHS = 100
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_cindex(y, y_pred):
    y = y.flatten()
    y_pred = y_pred.flatten()
    pairs = 0
    correct = 0
    for i in range(len(y)):
        for j in range(i + 1, len(y)):
            if y[i] > y[j]:
                pairs += 1
                if y_pred[i] > y_pred[j]:
                    correct += 1
            elif y[i] < y[j]:
                pairs += 1
                if y_pred[i] < y_pred[j]:
                    correct += 1
    return correct / pairs if pairs > 0 else 0

def save_log(log_data, dataset):
    with open(f"train_log_{dataset}.txt", "w") as f:
        f.write("epoch,train_loss,test_loss,ci\n")
        for line in log_data:
            f.write(f"{line['epoch']},{line['train_loss']},{line['test_loss']},{line['ci']}\n")

def run_full_experiment(DATASET):
    print(f"\n========================================")
    print(f"🚀 开始完整实验：{DATASET} 数据集")
    print(f"========================================")

    dataset = DeepDTADataset(f"./data/{DATASET}")
    train_folds = json.load(open(f"./data/{DATASET}/folds/train_fold_setting1.txt"))
    test_fold = json.load(open(f"./data/{DATASET}/folds/test_fold_setting1.txt"))

    train_loader = DataLoader(Subset(dataset, train_folds[0]), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(Subset(dataset, test_fold), batch_size=BATCH_SIZE, shuffle=False)

    model = DeepDTA().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_ci = 0
    log_list = []

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for smi, pro, y in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            smi, pro, y = smi.to(DEVICE), pro.to(DEVICE), y.to(DEVICE)
            pred = model(smi, pro)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for smi, pro, y in test_loader:
                smi, pro, y = smi.to(DEVICE), pro.to(DEVICE), y.to(DEVICE)
                pred = model(smi, pro)
                y_true.append(y.cpu().numpy())
                y_pred.append(pred.cpu().numpy())

        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        test_loss = np.mean((y_true - y_pred)**2)
        ci = get_cindex(y_true, y_pred)

        log_list.append({"epoch": epoch+1, "train_loss": train_loss, "test_loss": test_loss, "ci": ci})
        print(f"Epoch {epoch+1:2d} | Train {train_loss:.4f} | Test {test_loss:.4f} | CI {ci:.4f}")

        if ci > best_ci:
            best_ci = ci
            torch.save(model.state_dict(), f"deepdta_best_{DATASET}.pth")

    save_log(log_list, DATASET)
    print(f"\n🏆 {DATASET} 实验完成！最佳 CI: {best_ci:.4f}")

# ==================== 核心：自动跑两个数据集 ====================
if __name__ == "__main__":
    print("[INFO] 服务器全自动双数据集实验开始")
    run_full_experiment("davis")
    run_full_experiment("kiba")
    print("\n🎉🎉🎉 全部实验完成！")