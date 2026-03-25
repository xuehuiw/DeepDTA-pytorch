import json
import pickle
import numpy as np
from collections import OrderedDict
import torch
from torch.utils.data import Dataset

# 字符编码表（和原版论文完全一致）
CHARPROTSET = {
    "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
    "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
    "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
    "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25
}
CHARPROTLEN = 25

CHARISOSMISET = {
    "#": 1, "%": 2, ")": 3, "(": 4, "+": 5, "-": 6,
    ".": 7, "1": 8, "0": 9, "3": 10, "2": 11, "5": 12,
    "4": 13, "7": 14, "6": 15, "9": 16, "8": 17, "=": 18,
    "A": 19, "C": 20, "B": 21, "E": 22, "D": 23, "G": 24,
    "F": 25, "I": 26, "H": 27, "K": 28, "M": 29, "L": 30,
    "O": 31, "N": 32, "P": 33, "S": 34, "R": 35, "U": 36,
    "T": 37, "W": 38, "V": 39, "Y": 40, "[": 41, "Z": 42,
    "]": 43, "_": 44, "a": 45, "c": 46, "b": 47, "e": 48,
    "d": 49, "g": 50, "f": 51, "i": 52, "h": 53, "m": 54,
    "l": 55, "o": 56, "n": 57, "s": 58, "r": 59, "u": 60,
    "t": 61, "y": 62
}
CHARISOSMILEN = 62

class DeepDTADataset(Dataset):
    def __init__(self, data_path, max_smi_len=100, max_seq_len=1000):
        self.data_path = data_path
        self.max_smi_len = max_smi_len
        self.max_seq_len = max_seq_len
        self.load_dataset()

    def load_dataset(self):
        # 加载药物、蛋白、标签数据
        self.ligands = json.load(open(f"{self.data_path}/ligands_can.txt"), object_pairs_hook=OrderedDict)
        self.proteins = json.load(open(f"{self.data_path}/proteins.txt"), object_pairs_hook=OrderedDict)
        self.Y = pickle.load(open(f"{self.data_path}/Y", "rb"), encoding='latin1')

        # 编码
        self.X_smi = self.encode_smiles()
        self.X_pro = self.encode_proteins()

        # 获取有效索引
        self.row, self.col = np.where(np.isnan(self.Y) == False)

    def encode_smiles(self):
        encoded = []
        for smi in self.ligands.values():
            code = [CHARISOSMISET[c] if c in CHARISOSMISET else 0 for c in smi[:self.max_smi_len]]
            code += [0] * (self.max_smi_len - len(code))
            encoded.append(code)
        return np.array(encoded, dtype=np.int64)

    def encode_proteins(self):
        encoded = []
        for seq in self.proteins.values():
            code = [CHARPROTSET[c] if c in CHARPROTSET else 0 for c in seq[:self.max_seq_len]]
            code += [0] * (self.max_seq_len - len(code))
            encoded.append(code)
        return np.array(encoded, dtype=np.int64)

    def __len__(self):
        return len(self.row)

    def __getitem__(self, idx):
        i = self.row[idx]
        j = self.col[idx]
        smi = torch.LongTensor(self.X_smi[i])
        pro = torch.LongTensor(self.X_pro[j])
        y = torch.FloatTensor([self.Y[i, j]])
        return smi, pro, y