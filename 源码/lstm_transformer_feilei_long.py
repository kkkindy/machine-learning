import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import datetime
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
# 设置常量
ENCODER = 2
DROPOUT = 0.1
NUM_EPOCHS = 200
BATCH_SIZE = 64
D_MODEL = 128
LEARNING_RATE = 1e-4
PATIENCE = 10
NUM_SAMPLES = 10
NUM_EXP = 8
LOSS_PIC_SAVE_PATH = './loss_pics'
SEQ_OUT_LEN = 240

class BikeDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 1. 读取与预处理
def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    numeric_features = [
        "temp", "atemp", "hum", "windspeed"
    ]
    categorical_features = [
        "season", "yr", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit"
    ]
    target_col = "cnt"
    
    return train_df, test_df, numeric_features, categorical_features, target_col


# 无归一化处理的特征预处理
def preprocess_features(train_df, test_df, numeric_features, categorical_features):
    # 直接使用原始数值特征
    X_train_numeric = train_df[numeric_features].values
    X_test_numeric = test_df[numeric_features].values
    
    # 类别特征独热编码
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_train_categorical = categorical_transformer.fit_transform(train_df[categorical_features])
    X_test_categorical = categorical_transformer.transform(test_df[categorical_features])
    
    # 合并数值特征和类别特征
    X_train = np.hstack([X_train_categorical,X_train_numeric])
    X_test = np.hstack([X_test_categorical,X_test_numeric])

    return X_train, X_test


# 创建时间序列数据的函数
def create_sequences_multi_feature(data, input_size=96, output_size=96, step_size=1, feature_dim=13, target_col="cnt"):
    X, y = [], []
    length = len(data)
    
    # 如果数据是 DataFrame，获取目标列的索引
    if isinstance(data, pd.DataFrame):
        target_index = data.columns.get_loc(target_col)
    else:
        # 如果是 NumPy 数组，假设目标列在最后
        target_index = feature_dim - 1
    
    for start_idx in range(0, length - input_size - output_size + 1, step_size):
        end_idx = start_idx + input_size
        out_end_idx = end_idx + output_size
        seq_x = data[start_idx:end_idx, :feature_dim]  # 过去96小时的数据
        seq_y = data[end_idx:out_end_idx, target_index]  # 未来96小时的cnt
        
        X.append(seq_x)
        y.append(seq_y)
    
    return np.array(X), np.array(y)


# =========================
# 定义 LSTM + Transformer 结合模型
# =========================
class LSTMTransformer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, lstm_layers, transformer_layers, d_model, nhead, dim_feedforward, dropout):
        super(LSTMTransformer, self).__init__()

        # LSTM Part
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=lstm_layers, batch_first=True)

        # Linear projection layer to match d_model
        self.lstm_projection = nn.Linear(hidden_size, d_model)  # Project to d_model

        # Transformer Part
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=transformer_layers,
            num_decoder_layers=transformer_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )

        # Output Layer
        self.fc_out = nn.Linear(d_model, output_size)

    def forward(self, x):
        # LSTM processing
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Project LSTM output to d_model
        lstm_out = self.lstm_projection(lstm_out)

        # Transformer processing
        transformer_out = self.transformer(lstm_out, lstm_out)

        # Output Layer
        output = self.fc_out(transformer_out)

        # Squeeze to remove the extra dimension
        output = output[:, -1, :]  # Get the last time step's output

        return output

# =========================
# 训练与评估流程
# =========================
def save_comparison_plot(y_true, y_pred, title, save_plots_path="./plots",exp_i=0):
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label="Ground Truth (cnt)", marker='o', linestyle='-', linewidth=2, color='blue')
    plt.plot(y_pred, label="Prediction (cnt)", marker='x', linestyle='--', linewidth=2, color='orange')
    plt.title(f"Test Prediction vs Ground Truth: {title}", fontsize=16)
    plt.xlabel("Time Step", fontsize=14)
    plt.ylabel("Bike Count (Original Scale)", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(visible=True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12, loc='upper left')
    
    out_dir = save_plots_path
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{exp_i}_exp_{title}_long_comparison_plot.png")
    plt.savefig(out_path, dpi=300)
    plt.close()

def evaluate_best_model(test_loader, best_checkpoint_path, model, device="cpu",num_samples=6,exp_i=0):
    """
    从检查点加载最佳模型并进行测试。
    """
    # 加载模型
    checkpoint = torch.load(best_checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    # 评估模型
    model.eval()
    criterion_mse = nn.MSELoss()
    mae_func = nn.L1Loss()

    mse_list = []
    mae_list = []

    all_preds = []
    all_trues = []

    with torch.no_grad():
        for X_test, y_test in test_loader:
            X_test = X_test.to(device)
            y_test = y_test.to(device)

            # 只需要输入 src 数据进行预测
            y_pred = model(X_test)

            mse_val = criterion_mse(y_pred, y_test).item()  # 计算 MSE
            mae_val = mae_func(y_pred, y_test).item()  # 计算 MAE

            mse_list.append(mse_val)
            mae_list.append(mae_val)

            all_preds.append(y_pred.cpu().numpy())  # 保存预测值
            all_trues.append(y_test.cpu().numpy())  # 保存真实值

    mean_mse = np.mean(mse_list)
    mean_mae = np.mean(mae_list)

    all_preds = np.concatenate(all_preds, axis=0)
    all_trues = np.concatenate(all_trues, axis=0)

    # 选择第一个样本进行对比图保存
    # 保存多个样本的对比图
    for i in range(min(num_samples, len(all_trues))):  # 选择前num_samples个样本进行保存
        save_comparison_plot(all_trues[i], all_preds[i], f"long_{exp_i}_exp_Sample_{i+1}", save_plots_path="./plots")

    return mean_mse, mean_mae, all_preds, all_trues


# 训练过程中的更新，写入损失到 CSV 文件
def save_loss_to_csv(exp_id, epoch, train_loss_avg, val_loss_avg, save_dir="./logs"):
    # 确保文件夹存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 准备存储的数据
    loss_data = {
        "Experiment": [exp_id + 1],
        "Epoch": [epoch + 1],
        "Train Loss": [train_loss_avg],
        "Val Loss": [val_loss_avg]
    }
    
    # 将数据转换为 DataFrame
    loss_df = pd.DataFrame(loss_data)
    
    # 定义 CSV 文件路径
    csv_file_path = os.path.join(save_dir, f"long_experiment_{exp_id + 1}_loss_results.csv")
    
    # 如果文件不存在，则创建文件并写入头部，如果存在则追加数据
    if not os.path.exists(csv_file_path):
        loss_df.to_csv(csv_file_path, index=False)
    else:
        loss_df.to_csv(csv_file_path, mode='a', header=False, index=False)


def train_lstm_transformer_model(train_loader, val_loader, model, num_epochs=10, lr=1e-3, device="cpu", checkpoint_dir="./checkpoints", patience=10,exp_i=0):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    os.makedirs(checkpoint_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model.to(device)
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0
    best_model_state = None

    best_checkpoint_path = None  # 记录最佳模型的路径

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [train]", leave=False)

        # 训练过程
        for X_batch, y_batch in pbar:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})

        train_loss_avg = train_loss / len(train_loader)
        train_losses.append(train_loss_avg)

        # 验证过程
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val = X_val.to(device)
                y_val = y_val.to(device)
                out_val = model(X_val)
                loss_val = criterion(out_val, y_val)
                val_loss += loss_val.item()

        val_loss_avg = val_loss / len(val_loader)
        val_losses.append(val_loss_avg)

        # 打印每个 epoch 的训练和验证损失
        print(f"Epoch [{epoch+1}/{num_epochs}] -> "
              f"Train Loss: {train_loss_avg:.4f}, "
              f"Val Loss: {val_loss_avg:.4f}")
        
        # 将每个 epoch 的损失写入 CSV 文件
        save_loss_to_csv(exp_i, epoch, train_loss_avg, val_loss_avg)

        # 保存当前 epoch 的检查点
        checkpoint_path = os.path.join(checkpoint_dir, f"long_lstm_transformer_{exp_i}_exp.pth")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss_avg,
            'val_loss': val_loss_avg
        }, checkpoint_path)

        # 检查是否是当前最佳模型（基于验证损失）
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            best_epoch = epoch + 1
            best_model_state = model.state_dict()
            patience_counter = 0  # 重置 patience 计数器
            best_checkpoint_path = os.path.join(checkpoint_dir, f"long_best_checkpoint_{exp_i}.pth")
            print(f"Best model so far at epoch {epoch+1} -> Saving the model :{best_checkpoint_path}")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss_avg,
                'val_loss': val_loss_avg
            }, best_checkpoint_path)
        else:
            patience_counter += 1

        # 如果 patience 次数超过阈值，则提前停止训练
        if patience_counter >= patience:
            print(f"Early stopping triggered. No improvement in validation loss for {patience} epochs.")
            break

    # 返回训练损失、验证损失和最佳模型的路径
    return train_losses, val_losses, best_checkpoint_path


def run_experiment(train_df, test_df, feature_cols, target_col="cnt", input_size=96, output_size=96, batch_size=32, num_epochs=10, device="cpu", exp_i=0):
    # 数值特征和类别特征
    numeric_features = ["temp", "atemp", "hum", "windspeed","cnt"]
    categorical_features = ["season", "yr", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit"]
    
    # 预处理特征数据
    X_train, X_test = preprocess_features(train_df, test_df, numeric_features, categorical_features)

    feature_dim = X_train.shape[1]  # 特征维度
    
    # 创建时间序列数据 (X, y) 
    X_train, y_train = create_sequences_multi_feature(X_train, input_size=input_size, output_size=output_size, step_size=1, feature_dim=feature_dim, target_col=target_col)
    X_test, y_test = create_sequences_multi_feature(X_test, input_size=input_size, output_size=output_size, step_size=1, feature_dim=feature_dim, target_col=target_col)

    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, shuffle=False)

    # 创建 Dataset 和 DataLoader
    train_dataset = BikeDataset(X_train, y_train)
    val_dataset = BikeDataset(X_val, y_val)
    test_dataset = BikeDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型
    model = LSTMTransformer(
        input_size=feature_dim,  # 输入维度
        hidden_size=512,  # LSTM隐藏维度
        output_size=output_size,  # 输出维度
        lstm_layers=2,  # LSTM层数
        transformer_layers=2,  # Transformer层数
        d_model=D_MODEL,  # Transformer的嵌入维度
        nhead=4,  # Transformer的头数
        dim_feedforward=128,  # 前馈网络的维度
        dropout=DROPOUT  # dropout比例
    )

    # 训练模型并返回训练损失、验证损失和最佳模型路径
    train_losses, val_losses, best_checkpoint_path = train_lstm_transformer_model(
        train_loader, val_loader, model, num_epochs=num_epochs, lr=LEARNING_RATE, device=device, exp_i=exp_i
    )

    print("Training completed.")
    
    # 加载最佳模型并在测试集上评估
    mse, mae, preds, trues = evaluate_best_model(
        test_loader=test_loader,  # 测试数据加载器
        best_checkpoint_path=best_checkpoint_path,
        model=model,
        device=device,
        num_samples=NUM_SAMPLES,
        exp_i=exp_i
    )

    print(f"Test MSE: {mse:.4f}, MAE: {mae:.4f}")

    # 保存预测与真实值的对比图
    save_comparison_plot(trues, preds, "Test", save_plots_path="./plots", exp_i=exp_i)
    
    return train_losses, val_losses, mse, mae

# 主函数
def main_train_and_test():
    # 数据加载和预处理
    train_path = 'train_data.csv'
    test_path = 'test_data.csv'
    train_df, test_df, numeric_features, categorical_features, target_col = load_data(train_path, test_path)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for exp_i in range(NUM_EXP):
        # 运行训练并返回训练损失、验证损失、测试结果
        train_losses, val_losses, mse, mae = run_experiment(
            train_df=train_df,
            test_df=test_df,
            feature_cols=numeric_features + categorical_features,
            target_col=target_col,
            input_size=96,
            output_size=SEQ_OUT_LEN,
            batch_size=32,
            num_epochs=NUM_EPOCHS,
            device=device,
            exp_i = exp_i
        )

        print(f"Training completed. Test MSE: {mse:.4f}, MAE: {mae:.4f}")
        pd.DataFrame({"Experiment": [exp_i+1], "Test MSE": [mse], "Test MAE": [mae]}).to_csv(
            f"logs/experiment_long_{exp_i+1}_results.csv", index=False)


# 运行实验
if __name__ == "__main__":
    main_train_and_test()
