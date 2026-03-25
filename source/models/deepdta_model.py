import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepDTA(nn.Module):
    def __init__(
        self,
        vocab_smi=64,    # SMILES字符数
        vocab_pro=26,    # 蛋白质字符数
        embed_dim=128,   # 嵌入维度
        num_filters=32,  # 卷积核数量
        max_smi_len=100, # 药物最大长度
        max_pro_len=1000 # 蛋白最大长度
    ):
        super(DeepDTA, self).__init__()
        
        # ===================== 药物分支 CNN =====================
        self.embed_smi = nn.Embedding(vocab_smi, embed_dim)
        
        self.smi_conv1 = nn.Conv1d(embed_dim, num_filters, 4)
        self.smi_conv2 = nn.Conv1d(num_filters, num_filters*2, 8)
        self.smi_conv3 = nn.Conv1d(num_filters*2, num_filters*3, 12)
        
        # ===================== 蛋白分支 CNN =====================
        self.embed_pro = nn.Embedding(vocab_pro, embed_dim)
        
        self.pro_conv1 = nn.Conv1d(embed_dim, num_filters, 8)
        self.pro_conv2 = nn.Conv1d(num_filters, num_filters*2, 12)
        self.pro_conv3 = nn.Conv1d(num_filters*2, num_filters*3, 16)
        
        # ===================== 全连接层 =====================
        self.fc1 = nn.Linear(num_filters*3 * 2, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, 1)
        
        self.dropout = nn.Dropout(0.1)

    def forward(self, smi, pro):
        # ------------------- 药物分支 -------------------
        x_smi = self.embed_smi(smi).permute(0, 2, 1)
        x_smi = F.relu(self.smi_conv1(x_smi))
        x_smi = F.relu(self.smi_conv2(x_smi))
        x_smi = F.relu(self.smi_conv3(x_smi))
        x_smi = torch.max(x_smi, dim=2)[0]
        
        # ------------------- 蛋白分支 -------------------
        x_pro = self.embed_pro(pro).permute(0, 2, 1)
        x_pro = F.relu(self.pro_conv1(x_pro))
        x_pro = F.relu(self.pro_conv2(x_pro))
        x_pro = F.relu(self.pro_conv3(x_pro))
        x_pro = torch.max(x_pro, dim=2)[0]
        
        # ------------------- 融合 -------------------
        x = torch.cat([x_smi, x_pro], dim=1)
        
        # ------------------- 全连接 -------------------
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.out(x)
        
        return x