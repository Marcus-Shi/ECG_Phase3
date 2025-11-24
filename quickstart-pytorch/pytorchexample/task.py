# marcus-shi/ecg_phase3/ECG_Phase3-ec1bb3f03abae1d74da6110dad4af62b3796a611/quickstart-pytorch/pytorchexample/task.py

import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset

# =============================================================================
# 1. 模型定义 (User Provided)
# =============================================================================

class Net(nn.Module):
    def __init__(self, in_channels=12, seq_len=640, num_classes=5, p_drop=0.2,
                 lstm_hidden=100, lstm_layers=3):

        super().__init__()
        # CNN part
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=54, kernel_size=15, stride=1, padding=7),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),    # L: 640 -> 320
            nn.Dropout(p_drop)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(54, 54, kernel_size=11, stride=1, padding=5),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3),    # L: 320 -> 106
            nn.Dropout(p_drop)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(54, 54, kernel_size=13, stride=1, padding=6),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3),    # L: 106 -> 35
            nn.Dropout(p_drop)
        )

        # LSTM part
        self.lstm = nn.LSTM(
            input_size=54,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=p_drop if lstm_layers > 1 else 0
        )

        # Classifier
        self.fc = nn.Linear(lstm_hidden, num_classes)

    def forward(self, x):
        # Expect 3D input: (Batch, Channels, Length) or (Batch, Length, Channels)
        # Assuming input X is typically (Batch, Length, Channels) from standard ECG formats,
        # but PyTorch Conv1d expects (Batch, Channels, Length).
        
        if x.dim() != 3:
            if x.dim() == 2:
                x = x.unsqueeze(1)
            else:
                raise ValueError(f"Expect 3D, got {x.shape}")

        # Check dimensions and swap if necessary to get (B, C, L)
        # 假设 seq_len=640, channels=12. 
        # 如果输入是 (B, 640, 12)，需要转置为 (B, 12, 640)
        if x.shape[1] != 12 and x.shape[2] == 12:
            x = x.permute(0, 2, 1).contiguous()

        # CNN feature extraction
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # Prepare for LSTM: (B, T, F) -> LSTM expects (Batch, Seq_Len, Features)
        # Conv1d output is (B, Features, Seq_Len), so we swap back
        x = x.permute(0, 2, 1).contiguous()

        # LSTM
        out, (h_n, c_n) = self.lstm(x)
        last = h_n[-1]
        logits = self.fc(last)
        return logits

# =============================================================================
# 2. 数据加载逻辑 (Custom for ECG & HFL)
# =============================================================================

# 请修改为您存放 .npy 文件的实际路径
DATA_ROOT = "./data"  

def load_data(partition_id: int, num_partitions: int, batch_size: int):
    """
    加载并划分数据用于分层联邦学习。
    
    Args:
        partition_id: 对应 Cluster Label (y_train 第8列)
    Returns:
        patient_dataloaders (dict): {patient_id: DataLoader}，用于模拟该簇内的多个病人
        val_loader (DataLoader): 该簇的验证集
    """
    
    # 1. 加载数据 (建议使用 mmap_mode='r' 以节省内存，如果内存足够可不加)
    X_train = np.load(os.path.join(DATA_ROOT, "X_train.npy"))
    y_train = np.load(os.path.join(DATA_ROOT, "y_train.npy"))
    
    # 2. 根据 Cluster Label (第8列) 筛选属于当前 ClientApp (Cluster Leader) 的数据
    # 注意：partition_id 在 Flower 中通常是 0 到 num_partitions-1
    # 我们假设 y_train 第8列的标签也是 0 到 4 (如果是 1-5，需要 -1)
    cluster_mask = (y_train[:, 8] == partition_id)
    
    X_cluster = X_train[cluster_mask]
    y_cluster = y_train[cluster_mask]
    
    if len(X_cluster) == 0:
        raise ValueError(f"Cluster {partition_id} has no data! Check partition config.")

    # 3. 在 Cluster 内部，按 Patient ID (第6列) 再次分组
    # 这对于 HFL 至关重要：Edge Server 需要迭代其辖下的不同病人
    patient_ids = np.unique(y_cluster[:, 5])
    patient_dataloaders = {}
    
    for pid in patient_ids:
        p_mask = (y_cluster[:, 5] == pid)
        X_p = X_cluster[p_mask]
        y_p = y_cluster[p_mask]
        
        # 处理标签：取前5列 (One-hot) 或转为 Class Index
        # CrossEntropyLoss 需要 Class Index (LongTensor)
        # 假设 y 是 one-hot，我们需要 argmax。如果是 index，直接用。
        # 你的描述：0-4列是one-hot。
        y_p_labels = np.argmax(y_p[:, 0:5], axis=1)
        
        # 转换为 Tensor
        tensor_x = torch.Tensor(X_p) # Float
        tensor_y = torch.LongTensor(y_p_labels)
        
        dataset = TensorDataset(tensor_x, tensor_y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        patient_dataloaders[pid] = loader

    # 4. 加载验证集 (可选：每个 Cluster 也有对应的验证集)
    # 简单起见，这里我们可能只加载全局验证集的一部分，或者该 Cluster 的验证集
    # 这里假设验证集逻辑类似
    X_val = np.load(os.path.join(DATA_ROOT, "X_val.npy"))
    y_val = np.load(os.path.join(DATA_ROOT, "y_val.npy"))
    
    # 同样只取当前 Cluster 的验证数据
    val_mask = (y_val[:, 8] == partition_id)
    if np.sum(val_mask) > 0:
        X_val_c = X_val[val_mask]
        y_val_c = y_val[val_mask]
        y_val_labels = np.argmax(y_val_c[:, 0:5], axis=1)
        
        val_dataset = TensorDataset(torch.Tensor(X_val_c), torch.LongTensor(y_val_labels))
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
    else:
        val_loader = None

    return patient_dataloaders, val_loader

def load_centralized_testset():
    """加载全局测试集用于 Server 端评估"""
    X_test = np.load(os.path.join(DATA_ROOT, "X_test.npy"))
    y_test = np.load(os.path.join(DATA_ROOT, "y_test.npy"))
    
    y_labels = np.argmax(y_test[:, 0:5], axis=1)
    dataset = TensorDataset(torch.Tensor(X_test), torch.LongTensor(y_labels))
    return DataLoader(dataset, batch_size=128)

# =============================================================================
# 3. 训练与测试辅助函数
# =============================================================================

def train_one_epoch(net, trainloader, device, lr):
    """在一个病人的数据上训练一个 epoch (Local Update)"""
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr) # 或 SGD
    net.train()
    
    total_loss = 0.0
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = net(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / len(trainloader)

def test(net, testloader, device):
    """评估模型"""
    if testloader is None:
        return 0.0, 0.0
        
    net.to(device)
    net.eval()
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = correct / total if total > 0 else 0
    avg_loss = loss / len(testloader)
    return avg_loss, accuracy