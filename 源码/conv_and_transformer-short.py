import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import matplotlib.pyplot as plt
import glob
import datetime
import torch.nn.functional as F

# 设置常量
# ENCODER = 2
# DROPOUT = 0.2
NUM_EPOCHS = 300
BATCH_SIZE = 128
D_MODEL = 64
LEARNING_RATE = 1e-3
PATIENCE=10
LOSS_PIC_SAVE_PATH = './newfunc_short_loss_pics'

# 1. 读取与预处理
def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    feature_cols = [
        "season", "yr", "mnth", "hr", "holiday", "weekday", "workingday",
        "weathersit", "temp", "atemp", "hum", "windspeed"
    ]
    target_col = "cnt"
    
    return train_df, test_df, feature_cols, target_col

def load_test_data(test_path):
    test_df = pd.read_csv(test_path)
    feature_cols = [
        "season", "yr", "mnth", "hr", "holiday", "weekday", "workingday",
        "weathersit", "temp", "atemp", "hum", "windspeed"
    ]
    target_col = "cnt"
    
    return test_df, feature_cols, target_col


def standard_scale(train_df, test_df, feature_cols, target_col):
    scaler = StandardScaler()
    if train_df is not None:
        train_features_scaled = scaler.fit_transform(train_df[feature_cols])
        train_scaled = pd.DataFrame(train_features_scaled, columns=feature_cols)
        train_scaled[target_col] = train_df[target_col].values
    else:
        train_scaled = None
    test_features_scaled = scaler.fit_transform(test_df[feature_cols]) if train_df is None else scaler.transform(test_df[feature_cols])
    test_scaled = pd.DataFrame(test_features_scaled, columns=feature_cols)
    test_scaled[target_col] = test_df[target_col].values
    return train_scaled, test_scaled, scaler


def create_sequences_multi_feature(data, input_size=96, output_size=96, step_size=1, feature_dim=13, target_index=12):
    X, y = [], []
    length = len(data)
    for start_idx in range(0, length - input_size - output_size + 1, step_size):
        end_idx = start_idx + input_size
        out_end_idx = end_idx + output_size
        seq_x = data[start_idx:end_idx, :feature_dim]
        seq_y = data[end_idx:out_end_idx, target_index]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


class BikeDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# 2. 定义多尺度卷积网络 + Transformer模型


import torch
import torch.nn as nn
import torch.nn.functional as F

# 修改 MultiScaleConv 类
class MultiScaleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScaleConv, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(in_channels, out_channels, kernel_size=7, padding=3)
    
    def forward(self, x):
        out1 = F.relu(self.conv1(x))
        out2 = F.relu(self.conv2(x))
        out3 = F.relu(self.conv3(x))
        return out1 + out2 + out3  # 结果相加，可以改为其他融合方式

# 修改 TimeSeriesTransformerWithMultiscale 类
class TimeSeriesTransformerWithMultiscale(nn.Module):
    def __init__(self, feature_dim, d_model, input_size, num_heads, num_layers, num_classes):
        super(TimeSeriesTransformerWithMultiscale, self).__init__()
        
        self.feature_dim = feature_dim
        self.d_model = d_model
        self.input_size = input_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # 1. 多尺度卷积层
        self.multi_scale_conv = MultiScaleConv(in_channels=feature_dim, out_channels=d_model)
        
        # 2. 位置编码（这里直接将 pos_encoder_input 的维度改为 d_model）
        self.pos_encoder_input = nn.Parameter(torch.zeros(1, input_size, d_model))  # 修改为 d_model
        
        # 3. Transformer 编码器
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # 4. 分类器
        self.fc = nn.Linear(d_model, num_classes)


    def forward(self, x):
        # x 的形状是 (batch_size, feature_dim, input_size)
        #实际上：torch.Size([128, 96, 13])
        
        x = x.permute(0, 2, 1)  # 转换为 (batch_size, input_size, feature_dim)
        # 1. 通过卷积层获取特征
        conv_out = self.multi_scale_conv(x)  # 输出形状： (batch_size, d_model, input_size)= torch.Size([128, 64, 96])
        
        # 2. 添加位置编码
        # 位置编码的形状应该是 (batch_size, input_size, d_model)
        pos_enc = self.pos_encoder_input.expand(x.size(0), -1, -1)  # 扩展位置编码以适应 batch size |||pos_enc===torch.Size([128, 96, 64])
        pos_enc=pos_enc.permute(0,2,1)#pos_enc===torch.Size([128, 64, 96])
        x = conv_out + pos_enc  # 将卷积特征与位置编码相加
        
        # 3. 通过 Transformer 编码器
        # 注意 Transformer 的输入形状是 (seq_len, batch_size, d_model)，因此需要转置
        x = x.permute(2, 0, 1)  # (input_size, batch_size, d_model)
        
        transformer_out = self.transformer_encoder(x)  # 通过编码器，输出形状：(input_size, batch_size, d_model)
        
        # 4. 通过全连接层进行分类
        # 获取最后一个时间步的输出 (input_size, batch_size, d_model) -> (batch_size, d_model)
        last_step_out = transformer_out[-1, :, :]  # 取最后一个时间步的特征
        
        # 5. 输出分类结果
        logits = self.fc(last_step_out)  # (batch_size, num_classes)
        
        return logits


# 3. 训练与评估流程 new
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import torch
import datetime
from tqdm import tqdm
import torch.nn as nn

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler

def train_transformer_model(
    train_loader,
    val_loader,
    model,
    num_epochs=10,
    lr=1e-3,
    device="cpu",
    checkpoint_dir="./newfunc_short_checkpoints",
    patience=5
):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, verbose=True)
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model.to(device)
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [train]", leave=False)

        for X_batch, y_batch in pbar:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)  # Forward pass
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})

        train_loss_avg = train_loss / len(train_loader)
        train_losses.append(train_loss_avg)

        # ============================== 验证过程 ==============================
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

        print(f"Epoch [{epoch+1}/{num_epochs}] -> "
              f"Train Loss: {train_loss_avg:.4f}, "
              f"Val Loss: {val_loss_avg:.4f}")

        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            best_epoch = epoch + 1
            best_model_state = model.state_dict()
            patience_counter = 0
            best_checkpoint_path = os.path.join(checkpoint_dir, f"best_checkpoint_{timestamp}.pth")
            print(f"Saving best model at epoch {epoch+1}")
            torch.save(best_model_state, best_checkpoint_path)
        else:
            patience_counter += 1

        scheduler.step(val_loss_avg)

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1} due to no improvement")
            break

    return  train_losses, val_losses


def plot_loss(train_losses, save_path, name, exp_id):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label=name, color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f"exp{exp_id}_{name} Curve")
    plt.legend()
    plt.grid(True)
    save_path_and_filename = f"{save_path}/exp{exp_id}_{name}.png"
    plt.savefig(save_path_and_filename)
    print(f"{name} loss plot saved at {save_path_and_filename}")



def evaluate_model(test_loader, model, device="cpu"):
    """
    在测试集上预测并计算 MSE 和 MAE，针对 Encoder-only 模型。
    """
    model.eval()
    criterion_mse = nn.MSELoss()  # 均方误差
    mae_func = nn.L1Loss()  # 平均绝对误差

    mse_list = []
    mae_list = []

    all_preds = []
    all_trues = []

    with torch.no_grad():
        for X_test, y_test in test_loader:
            X_test = X_test.to(device)
            y_test = y_test.to(device)

            # 只需要输入 src 数据进行预测
            y_pred = model(X_test)  # (batch_size, output_size) 直接生成预测值

            mse_val = criterion_mse(y_pred, y_test).item()  # 计算 MSE
            mae_val = mae_func(y_pred, y_test).item()  # 计算 MAE

            mse_list.append(mse_val)
            mae_list.append(mae_val)

            all_preds.append(y_pred.cpu().numpy())  # 保存预测值
            all_trues.append(y_test.cpu().numpy())  # 保存真实值

    mean_mse = np.mean(mse_list)  # 计算平均 MSE
    mean_mae = np.mean(mae_list)  # 计算平均 MAE

    # 将所有的预测结果和真实结果拼接
    all_preds = np.concatenate(all_preds, axis=0)
    all_trues = np.concatenate(all_trues, axis=0)

    # 返回测试集的平均 MSE 和 MAE，以及所有的预测和真实值
    return mean_mse, mean_mae, all_preds, all_trues


def run_experiment(
    train_df,
    test_df,
    feature_cols,
    target_col,
    input_size=96,
    output_size=96,
    batch_size=32,
    num_epochs=10,
    device="cpu"
):
    """
    1) 只对 12 个特征做标准化，cnt 不做变换
    2) 构建 (X, y)，X含 13 列(12缩放特征 + 1原始cnt)，y为未来96小时的cnt
    3) 训练 Transformer
    4) 测试并返回 MSE, MAE, preds, trues
    5) 返回训练损失和验证损失，并绘制损失曲线
    """
    # ============ (1) 标准化(仅特征) ============

    # 使用 `standard_scale` 对特征进行标准化
    train_scaled, test_scaled, scaler = standard_scale(
        train_df, test_df, feature_cols=feature_cols, target_col=target_col
    )

    # 合并成 numpy，每行共 13 列 => 前12列是 scaled 特征，第13列是原始 cnt
    train_array = train_scaled[feature_cols + [target_col]].values  # shape = [N, 13]
    test_array  = test_scaled[feature_cols + [target_col]].values   # shape = [M, 13]
    
    feature_dim = 13  # 输入的特征维度（12 个特征 + 1 原始 cnt）
    target_index = 12  # `cnt` 为第 13 列

    # ============ 划分训练、验证集 ============

    train_len = int(len(train_array) * 0.9)  # 后10%做验证
    train_part = train_array[:train_len]
    val_part   = train_array[train_len:]

    X_train, y_train = create_sequences_multi_feature(
        train_part,
        input_size=input_size,
        output_size=output_size,
        step_size=1,
        feature_dim=feature_dim,
        target_index=target_index
    )
    X_val, y_val = create_sequences_multi_feature(
        val_part,
        input_size=input_size,
        output_size=output_size,
        step_size=1,
        feature_dim=feature_dim,
        target_index=target_index
    )
    X_test, y_test = create_sequences_multi_feature(
        test_array,
        input_size=input_size,
        output_size=output_size,
        step_size=1,
        feature_dim=feature_dim,
        target_index=target_index
    )

    # X_train = np.transpose(X_train, (2, 0, 1)) 
    # X_val = np.transpose(X_val, (2, 0, 1)) 
    # X_test = np.transpose(X_test, (2, 0, 1)) 
    # 创建 Dataset 和 DataLoader
    train_dataset = BikeDataset(X_train, y_train)
    val_dataset   = BikeDataset(X_val, y_val)
    test_dataset  = BikeDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # ============ (2) 定义模型 ============

    model = TimeSeriesTransformerWithMultiscale(
        feature_dim=13,  # 特征维度
        d_model=D_MODEL, 
        input_size=96,  # 输入序列长度
        num_heads=4,  # 可以去掉这个参数，移除对 `nhead` 的使用
        num_layers=2,  # 编码器层数
        num_classes=96  # 输出类别数
        )

    # ============ (3) 训练模型并记录损失 ============

    # 训练模型并获取训练和验证损失
    print("Training the model...")
    
    train_losses, val_losses = train_transformer_model(
        train_loader=train_loader, 
        val_loader=val_loader, 
        model=model,
        num_epochs=num_epochs,
        lr=LEARNING_RATE,
        device=device,
        checkpoint_dir="./newfunc_short_checkpoints",
        patience=PATIENCE  # 设置 patience 为 3
    )
    # ============ (4) 测试模型并获取评估指标 ============

    mse, mae, preds, trues = evaluate_model(test_loader, model, device=device)
    print(f"Test Set Evaluation -> Mean MSE: {mse:.4f}, Mean MAE: {mae:.4f}")

    return mse, mae, preds, trues,train_losses,val_losses



def myplot(trues, preds, exp_id):
    """
    绘制预测结果与真实值的对比图
    """
    max_idx = len(trues)
    for sample_idx in range(0, max_idx, 100):  # 对测试集前 5 个样本绘图,美观版
        plt.figure(figsize=(12, 6))  # 调整图的大小
        plt.plot(trues[sample_idx], label="Ground Truth (cnt)", marker='o', linestyle='-', linewidth=2, color='blue')
        plt.plot(preds[sample_idx], label="Prediction (cnt)", marker='x', linestyle='--', linewidth=2, color='orange')

        # 设置标题和轴标签的字体大小
        plt.title(f"Prediction for Sample {sample_idx} (Exp {exp_id})", fontsize=16)
        plt.xlabel("Time Step", fontsize=14)
        plt.ylabel("Bike Count (Original Scale)", fontsize=14)
        
        # 设置刻度字体大小
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        
        # 添加网格
        plt.grid(visible=True, linestyle='--', alpha=0.6)
        
        # 优化图例位置
        plt.legend(fontsize=12, loc='upper left')
        
        # 保存图片
        out_dir = "./newfunc_short_output"
        os.makedirs(out_dir, exist_ok=True)  # 确保保存路径存在
        out_path = os.path.join(out_dir, f"pred_vs_gt_sample{sample_idx}_exp{exp_id}.png")
        plt.savefig(out_path, dpi=300)  # 增加分辨率
        plt.close()

def main_train_and_test():
    """
    训练与测试模型的主函数
    """
    # ============ 文件路径(自行修改) ============ 
    train_path = "/home/jmyan/jqxx/data/train_data.csv"
    test_path = "/home/jmyan/jqxx/data/test_data.csv"

    # ============ 读取数据 ============ 
    train_df, test_df, feature_cols, target_col = load_data(train_path, test_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("===== 长期预测 (O=240) =====")
    num_experiments = 10
    longterm_mse_list = []
    longterm_mae_list = []

    for exp_id in range(num_experiments):
        print(f"\n--- 长期预测 实验 {exp_id+1} ---")
        mse, mae, preds, trues, train_losses, val_losses = run_experiment(
            train_df=train_df,
            test_df=test_df,
            feature_cols=feature_cols,
            target_col=target_col,
            input_size=96,
            output_size=96,
            batch_size=BATCH_SIZE,  # 32
            num_epochs=NUM_EPOCHS,  # 5  # 示例
            device=device
        )

        # 绘制训练和验证损失曲线
        plot_loss(train_losses, save_path=LOSS_PIC_SAVE_PATH, name="train_losses", exp_id=exp_id)
        plot_loss(val_losses, save_path=LOSS_PIC_SAVE_PATH, name="val_losses", exp_id=exp_id)

        # 记录 MSE 和 MAE
        longterm_mse_list.append(mse)
        longterm_mae_list.append(mae)

        print(f"[实验 {exp_id+1}] Test MSE: {mse:.4f}, MAE: {mae:.4f}")
        
        # 绘制测试集中特定样本的预测 vs. 真实值
        myplot(trues, preds, str(exp_id + 1))

    # 输出长期预测实验结果
    print(f"\n=== 长期预测 (O=240) {num_experiments} 次实验结果 ===")
    print(f"MSE mean: {np.mean(longterm_mse_list):.4f}, std: {np.std(longterm_mse_list):.4f}")
    print(f"MAE mean: {np.mean(longterm_mae_list):.4f}, std: {np.std(longterm_mae_list):.4f}")

# 判断是否为主程序入口
if __name__ == "__main__":
    main_train_and_test()
