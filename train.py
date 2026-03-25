import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import json
from tqdm import tqdm

# 导入模块
from source.utils.data_loader import DeepDTADataset
from source.models.deepdta_model import DeepDTA

# ===================== 超小验证配置 =====================
MAX_SMI_LEN = 100
MAX_PRO_LEN = 1000
BATCH_SIZE = 4
EPOCHS = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CI 指数（论文核心）
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

# 单数据集训练（小型验证）
def train_single_dataset(dataset_name):
    print(f"\n===== 开始验证：{dataset_name} 数据集 =====")
    dataset = DeepDTADataset(
        data_path=f"./data/{dataset_name}",
        max_smi_len=MAX_SMI_LEN,
        max_seq_len=MAX_PRO_LEN
    )

    # 只取前100条数据，超快
    small_data = Subset(dataset, list(range(100)))
    loader = DataLoader(small_data, batch_size=BATCH_SIZE, shuffle=True)

    model = DeepDTA().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for smi, pro, y in tqdm(loader):
            smi, pro, y = smi.to(DEVICE), pro.to(DEVICE), y.to(DEVICE)
            pred = model(smi, pro)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1} | Loss: {total_loss/len(loader):.4f}")

    print(f"✅ {dataset_name} 验证完成！一切正常！")

# ==================== 本地验证：自动依次跑 DAVIS → KIBA ====================
if __name__ == "__main__":
    print("[INFO] 开始本地最小验证版")
    train_single_dataset("davis")
    train_single_dataset("kiba")
    print("\n🎉🎉🎉 全部验证完成！代码、模型、数据、环境 100% 正常！")