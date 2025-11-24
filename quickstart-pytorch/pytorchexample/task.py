"""pytorchexample: A Flower / PyTorch app."""

import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset
# 【关键修正】确保包含以下 sklearn 引用
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# =============================================================================
# 1. 模型定义
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
        if x.dim() != 3:
            if x.dim() == 2:
                x = x.unsqueeze(1)
            else:
                raise ValueError(f"Expect 3D, got {x.shape}")

        # 如果输入是 (B, 640, 12)，转置为 (B, 12, 640)
        if x.shape[1] != 12 and x.shape[2] == 12:
            x = x.permute(0, 2, 1).contiguous()

        # CNN feature extraction
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # Prepare for LSTM: (B, T, F)
        x = x.permute(0, 2, 1).contiguous()

        # LSTM
        out, (h_n, c_n) = self.lstm(x)
        last = h_n[-1]
        logits = self.fc(last)
        return logits

# =============================================================================
# 2. 数据加载逻辑
# =============================================================================

DATA_ROOT = "./data"

def load_data(partition_id: int, num_partitions: int, batch_size: int):
    """
    加载并划分数据用于分层联邦学习。
    partition_id: 对应 Cluster Label (y_train 第8列)
    """
    # 1. 加载数据
    X_train = np.load(os.path.join(DATA_ROOT, "X_train.npy"))
    y_train = np.load(os.path.join(DATA_ROOT, "y_train.npy"))
    
    # 2. 根据 Cluster Label (第8列) 筛选数据
    cluster_mask = (y_train[:, 8] == partition_id)
    X_cluster = X_train[cluster_mask]
    y_cluster = y_train[cluster_mask]
    
    if len(X_cluster) == 0:
        # 为了防止报错，如果某个 Cluster 没数据，返回空字典
        print(f"Warning: Cluster {partition_id} has no data.")
        return {}, None

    # 3. 按 Pseudo Patient ID (第5列) 分组
    patient_ids = np.unique(y_cluster[:, 5])
    patient_dataloaders = {}
    
    for pid in patient_ids:
        p_mask = (y_cluster[:, 5] == pid)
        X_p = X_cluster[p_mask]
        y_p = y_cluster[p_mask]
        
        # 处理标签：取前5列 (One-hot)
        y_p_labels = np.argmax(y_p[:, 0:5], axis=1)
        
        tensor_x = torch.Tensor(X_p)
        tensor_y = torch.LongTensor(y_p_labels)
        
        dataset = TensorDataset(tensor_x, tensor_y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        patient_dataloaders[pid] = loader

    # 4. 加载验证集
    X_val = np.load(os.path.join(DATA_ROOT, "X_val.npy"))
    y_val = np.load(os.path.join(DATA_ROOT, "y_val.npy"))
    
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
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    
    total_loss = 0.0
    if len(trainloader) == 0:
        return 0.0

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
    """
    评估模型，计算详细指标: Loss, Acc, Precision, Recall, F1 (Macro)
    """
    if testloader is None:
        return {
            "loss": 0.0, "accuracy": 0.0, 
            "precision": 0.0, "recall": 0.0, "f1": 0.0
        }
        
    net.to(device)
    net.eval()
    criterion = torch.nn.CrossEntropyLoss()
    
    total_loss = 0.0
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            
    avg_loss = total_loss / len(testloader) if len(testloader) > 0 else 0.0
    
    # 计算指标 (Macro Average)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    acc = accuracy_score(all_labels, all_preds)
    
    return {
        "loss": avg_loss,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }