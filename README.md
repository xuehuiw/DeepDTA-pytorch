```markdown
# DeepDTA-PyTorch
PyTorch 完整复现 | DeepDTA 药物-靶点亲和力预测模型

## 📚 项目介绍
本项目是论文 **DeepDTA: Deep Drug-Target Affinity Prediction** 的 PyTorch 完整复现，
使用双 CNN 结构分别对药物 SMILES 序列和蛋白质序列进行特征提取，实现高精度药物-靶点亲和力预测。

项目支持 DAVIS、KIBA 两大标准数据集，实现一键自动训练、自动保存最优模型、自动输出绘图日志，完全可复现论文实验结果。

---

## ✨ 核心功能
✅ 1:1 复现原论文 DeepDTA 双 CNN 模型结构
✅ 支持 DAVIS + KIBA 双数据集自动顺序训练
✅ 自动计算 CI 指数（论文核心评价指标）
✅ 自动保存最优模型权重文件
✅ 自动输出训练日志（可直接用于论文绘图）
✅ 模块化代码，简洁易懂，便于二次开发
✅ 一键运行，无需手动修改参数

---

## 📂 项目结构
```
DeepDTA-PyTorch/
├─ data/                  # 数据集文件夹（DAVIS + KIBA）
├─ source/                # 核心代码模块
│   ├─ models/            # DeepDTA 双CNN模型定义
│   └─ utils/             # 数据加载与预处理工具
├─ run_all.py             # 服务器完整实验脚本（自动跑 DAVIS → KIBA）
├─ train.py               # 本地快速验证脚本（小数据集测试）
└─ README.md              # 项目说明文档
```

---

## 🚀 快速开始
### 1. 安装依赖
```bash
pip install torch numpy pandas scikit-learn biopython tqdm
```

### 2. 运行完整论文实验
```bash
python run_all.py
```
程序将自动执行：
1. 训练 DAVIS 数据集（100 个 epoch）
2. 训练 KIBA 数据集（100 个 epoch）
3. 保存最优模型
4. 生成绘图日志文件

---

## 📊 自动生成文件
运行完成后，项目根目录会生成以下 4 个文件：
- `deepdta_best_davis.pth`  - DAVIS 最优模型
- `deepdta_best_kiba.pth`   - KIBA 最优模型
- `train_log_davis.txt`     - DAVIS 训练日志（Loss/CI）
- `train_log_kiba.txt`      - KIBA 训练日志（Loss/CI）

日志文件可直接用 Excel / Origin 绘制论文曲线图。

---

## 🎯 评价指标
项目实现原论文核心评估指标：
- CI (Concordance Index)
- MSE (Mean Squared Error)
- 5 折交叉验证

---

## 📝 使用说明
- `run_all.py`：服务器端完整实验（推荐）
- `train.py`：本地快速验证环境与代码正确性

---

## 🔗 原论文
DeepDTA: Deep Drug-Target Affinity Prediction

---

## ✅ 项目亮点
- 完全可复现
- 代码规范、注释清晰
- 开箱即用，无需配置
- 适合科研复现、课程设计、学习使用
```