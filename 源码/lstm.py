import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import random
import os

# ========================== 超参数设置 ==========================
SEQ_IN_LEN = 96        # 输入序列长度 (过去 I=96 小时)
BATCH_SIZE = 64
HIDDEN_SIZE = 512      # 增大隐藏层
NUM_LAYERS = 2         # 双层 LSTM
DROPOUT = 0.2          # 适度 dropout

NUM_EPOCHS = 200       # 每次实验的最大训练轮数 (epoch)
NUM_EXPERIMENTS = 8    # 重复实验次数
PATIENCE = 10          # early stopping 容忍轮数

# 创建保存结果的文件夹
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("figures", exist_ok=True)
os.makedirs("logs", exist_ok=True)  # 用来保存训练过程

# ============== 1. 读取数据 (处理标签、分类特征和数值特征) ==============
def load_data(train_path: str, test_path: str):
    """
    读取数据，分别提取标签、分类特征和数值特征。
    """
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    # 提取分类特征和数值特征
    categorical_features = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit']
    numerical_features = ['temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered']
    
    # 标签
    y_train = df_train['cnt'].values.astype(np.float32)
    y_test = df_test['cnt'].values.astype(np.float32)
    
    # 提取分类特征（独热编码）
    X_train_cat = pd.get_dummies(df_train[categorical_features], drop_first=True).values
    X_test_cat = pd.get_dummies(df_test[categorical_features], drop_first=True).values
    
    # 提取数值特征
    X_train_num = df_train[numerical_features].values.astype(np.float32)
    X_test_num = df_test[numerical_features].values.astype(np.float32)

    # 合并分类特征和数值特征
    X_train = np.concatenate([X_train_cat, X_train_num], axis=1)
    X_test = np.concatenate([X_test_cat, X_test_num], axis=1)

    return X_train, X_test, y_train, y_test

# ============== 2. 多步序列数据生成 ==============
def create_dataset(X, y, seq_in_len, seq_out_len):
    """
    用过去 seq_in_len 条记录预测未来 seq_out_len 条记录:
      X[i] = X[i : i + seq_in_len]
      Y[i] = y[i + seq_in_len : i + seq_in_len + seq_out_len]
    返回:
      X.shape = (样本数, seq_in_len, 特征数)
      Y.shape = (样本数, seq_out_len)
    """
    X_seq, Y_seq = [], []
    total_len = len(X)
    for i in range(total_len - seq_in_len - seq_out_len + 1):
        X_seq.append(X[i : i + seq_in_len])
        Y_seq.append(y[i + seq_in_len : i + seq_in_len + seq_out_len])
    
    X_seq = np.array(X_seq, dtype=np.float32)
    Y_seq = np.array(Y_seq, dtype=np.float32)
    return X_seq, Y_seq

class SeqDataset(Dataset):
    """ 简单包装 (X, Y) 的 Dataset """
    def __init__(self, X, Y):
        super().__init__()
        self.X = X
        self.Y = Y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# ============== 3. 定义多步预测 LSTM 模型 ==============
class MultiStepLSTM(nn.Module):
    """
    多层 LSTM + dropout，用于多步并行预测 (many-to-many)。
    - LSTM 编码输入序列 -> 取最后时刻输出 -> 全连接映射到 seq_out_len
    """
    def __init__(self, input_size, hidden_size, seq_out_len, num_layers=1, dropout=0.0):
        super(MultiStepLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.seq_out_len = seq_out_len
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, seq_out_len)

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)  # out: (batch, seq_in_len, hidden_size)
        last_out = out[:, -1, :]       # 取最后时刻隐藏状态 (batch, hidden_size)
        y_pred = self.fc(last_out)     # (batch, seq_out_len)
        return y_pred
    
'''
画对比图
'''
def plot_sample_predictions(all_preds, all_trues, seq_out_len, exp_i):
    """
    选择5个样本并绘制预测与真实值的对比图，每个样本生成一个单独的图像。
    """
    # 随机选择5个样本
    num_samples = len(all_preds)
    sample_indices = np.random.choice(num_samples, 6, replace=False)

    for i, idx in enumerate(sample_indices):
        sample_pred = all_preds[idx]  # (seq_out_len,)
        sample_true = all_trues[idx]  # (seq_out_len,)

        # 为每个样本创建一个新的图形
        plt.figure(figsize=(8, 4))
        plt.plot(range(seq_out_len), sample_true, label='Ground Truth', marker='o', color='blue')
        plt.plot(range(seq_out_len), sample_pred, label='Prediction',marker='x',color='red')
        plt.title(f"Sample {i+1} - Prediction vs Ground Truth (Exp {exp_i+1})")
        plt.xlabel("Future Hour")
        plt.ylabel("cnt")
        plt.legend(loc="best")
        plt.tight_layout()

        # 保存每个样本的图像
        fig_path = f"figures/prediction_sample_{i+1}_exp{exp_i+1}.png"
        plt.savefig(fig_path, dpi=120)
        plt.close()
# ============== 4. 训练 + 测试函数 ==============
def set_seed(seed=42):
    """ 固定随机种子，便于多次实验复现 """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train_and_evaluate(
    X_train, y_train, X_test, y_test, seq_in_len, seq_out_len,
    num_epochs=200, num_exps=5, patience=10
):
    """
    1) 从 train_data 构造训练+验证集，训练多步预测 LSTM (多次实验)
    2) 在 test_data 上计算 MSE / MAE
    3) 返回多次实验 (MSE, MAE) 的 (mean, std)
    4) 同时保存每次实验的最佳模型、训练过程日志和预测对比图
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=== 目标: 预测未来 {seq_out_len} 小时 (无标准化), device: {device} ===")

    # ------ 构建 "完整" 训练集 ------
    X_full, Y_full = create_dataset(X_train, y_train, seq_in_len, seq_out_len)
    full_dataset = SeqDataset(X_full, Y_full)

    # ------ 再划分 训练集 + 验证集 ------
    val_ratio = 0.1
    val_size = int(len(full_dataset) * val_ratio)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ------ 构建测试集 ------
    X_test_seq, Y_test_seq = create_dataset(X_test, y_test, seq_in_len, seq_out_len)
    test_dataset = SeqDataset(X_test_seq, Y_test_seq)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 用于存储每次实验的 测试 MSE/MAE (来自最优 epoch)
    mses = []
    maes = []
    all_train_losses = []
    all_val_losses = []
    all_y_true = []
    all_y_pred = []

    for exp_i in range(num_exps):
        set_seed(42 + exp_i)

        # 定义模型
        model = MultiStepLSTM(
            input_size=X_train.shape[1],  # 特征数
            hidden_size=HIDDEN_SIZE,
            seq_out_len=seq_out_len,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT
        ).to(device)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        # 学习率调度器 (监控验证集 Loss)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )

        # Early Stopping
        best_val_loss = float('inf')
        best_epoch = -1
        epochs_no_improve = 0

        # 训练循环
        train_losses = []
        val_losses = []
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                optimizer.zero_grad()
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            avg_train_loss = running_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            # 验证
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    y_pred = model(X_batch)
                    loss = criterion(y_pred, y_batch)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

            print(f"Exp-{exp_i + 1}, Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch
                epochs_no_improve = 0
                # 保存最优模型
                torch.save(model.state_dict(), f"checkpoints/best_model_exp_{exp_i + 1}.pth")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

            scheduler.step(avg_val_loss)

        # 保存训练与验证损失曲线
        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)

        # 在测试集上计算 MSE 和 MAE
        model.load_state_dict(torch.load(f"checkpoints/best_model_exp_{exp_i + 1}.pth", weights_only=True))
        model.eval()
        y_true, y_pred = [], []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_hat = model(X_batch)
                y_true.append(y_batch.cpu().numpy())
                y_pred.append(y_hat.cpu().numpy())

        y_true = np.concatenate(y_true, axis=0)
        y_pred = np.concatenate(y_pred, axis=0)

        mse = np.mean((y_true - y_pred) ** 2)
        mae = np.mean(np.abs(y_true - y_pred))
        mses.append(mse)
        maes.append(mae)

        # 保存预测对比图
        # 绘制测试集5个样本的对比图
        plot_sample_predictions(y_pred, y_true, seq_out_len, exp_i)

        # 输出实验结果
        print("\n实验结果 (MSE, MAE):")
        print(f"MSE Mean: {np.mean(mses):.4f}, Std: {np.std(mses):.4f}")
        print(f"MAE Mean: {np.mean(maes):.4f}, Std: {np.std(maes):.4f}")
        pd.DataFrame({"Experiment": [exp_i+1], "Test MSE": [mse], "Test MAE": [mae]}).to_csv(
        f"logs/experiment_short_{exp_i+1}_results.csv", index=False)

    # 绘制所有实验的训练损失与验证损失曲线
    plt.figure(figsize=(10, 6))
    for exp_i in range(num_exps):
        plt.plot(all_train_losses[exp_i], label=f"Train Exp-{exp_i + 1}")
        plt.plot(all_val_losses[exp_i], label=f"Val Exp-{exp_i + 1}", linestyle="--")
    plt.title("Train vs Val Loss Across Experiments")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("figures/train_val_loss.png")
    plt.show()

# 运行示例
train_path = 'train_data.csv'
test_path = 'test_data.csv'
X_train, X_test, y_train, y_test = load_data(train_path, test_path)
train_and_evaluate(X_train, y_train, X_test, y_test, SEQ_IN_LEN, seq_out_len=96,num_epochs=NUM_EPOCHS)
