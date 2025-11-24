# marcus-shi/ecg_phase3/ECG_Phase3-ec1bb3f03abae1d74da6110dad4af62b3796a611/quickstart-pytorch/pytorchexample/client_app.py

import torch
import copy
from collections import OrderedDict
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from pytorchexample.task import Net, load_data, train_one_epoch, test

app = ClientApp()

def aggregate_weights(weights_list, num_examples_list):
    """在 Edge 端聚合病人模型的辅助函数 (Weighted Average)"""
    total_examples = sum(num_examples_list)
    weighted_weights = [
        [layer * num for layer in weights] 
        for weights, num in zip(weights_list, num_examples_list)
    ]
    
    # Sum up weighted weights
    aggregated_weights = [
        sum(layers) / total_examples for layers in zip(*weighted_weights)
    ]
    return aggregated_weights

@app.train()
def train(msg: Message, context: Context):
    """
    分层联邦学习的 Edge Server 训练逻辑 (HierFAVG)。
    """
    # 1. 获取配置
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    
    # HFL Hyperparameters
    lr = msg.content["config"]["lr"]
    kappa_1 = int(context.run_config["kappa-1"]) # Local updates (epochs per patient)
    kappa_2 = int(context.run_config["kappa-2"]) # Edge aggregations
    batch_size = int(context.run_config["batch-size"])

    # 2. 加载数据 (返回该 Cluster 下所有病人的数据加载器)
    # 注意：在模拟中，每次 train 调用都会加载，建议添加缓存机制以提高效率
    patient_dataloaders, _ = load_data(partition_id, num_partitions, batch_size)
    
    # 3. 初始化 Edge Model (从 Global Server 接收)
    global_state_dict = msg.content["arrays"].to_torch_state_dict()
    edge_model = Net()
    edge_model.load_state_dict(global_state_dict)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    edge_model.to(device)

    # 4. 开始 Hierarchical Training Loops
    # 对应论文中的 Edge Aggregation 循环
    
    for edge_round in range(kappa_2):
        patient_weights_list = []
        patient_num_examples = []
        
        # 获取当前 Edge Model 的参数 (作为所有病人的初始参数)
        current_edge_weights = [val.cpu().numpy() for val in edge_model.state_dict().values()]
        
        # 遍历该 Cluster 下的每个病人 (模拟 Cross-Device 训练)
        for patient_id, loader in patient_dataloaders.items():
            # A. 病人下载 Edge 模型
            patient_model = Net()
            params_dict = zip(edge_model.state_dict().keys(), current_edge_weights)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            patient_model.load_state_dict(state_dict)
            patient_model.to(device)
            
            # B. 病人本地训练 (kappa_1 epochs)
            # 这里简单起见，我们假设 kappa_1 就是 epochs 的数量
            for _ in range(kappa_1):
                train_one_epoch(patient_model, loader, device, lr)
            
            # C. 收集病人更新后的权重
            p_weights = [val.cpu().numpy() for val in patient_model.state_dict().values()]
            patient_weights_list.append(p_weights)
            patient_num_examples.append(len(loader.dataset))
        
        # 5. Edge Aggregation (在 Cluster Leader 处聚合)
        if patient_weights_list:
            new_edge_weights = aggregate_weights(patient_weights_list, patient_num_examples)
            
            # 更新 Edge Model，准备进入下一个 kappa_2 轮次
            params_dict = zip(edge_model.state_dict().keys(), new_edge_weights)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            edge_model.load_state_dict(state_dict)

    # 6. 计算最终指标 (基于 Edge Model 在本地所有数据上的表现)
    # 为了统计，我们可以简单取一个病人的 loss 或者重新在所有病人数据上 eval
    # 这里简单返回 0.0，实际可加逻辑
    train_loss = 0.0 
    total_examples = sum([len(l.dataset) for l in patient_dataloaders.values()])

    # 7. 返回更新后的 Edge Model 给 Global Server (Cloud)
    final_edge_weights = edge_model.state_dict()
    model_record = ArrayRecord(final_edge_weights)
    
    metrics = {
        "train_loss": train_loss, 
        "num-examples": total_examples,
        "cluster_id": partition_id
    }
    
    return Message(
        content=RecordDict({"arrays": model_record, "metrics": MetricRecord(metrics)}), 
        reply_to=msg
    )

@app.evaluate()
def evaluate(msg: Message, context: Context):
    """
    Cluster Leader 评估逻辑
    """
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    
    # 加载模型
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # 加载该 Cluster 的验证集
    _, valloader = load_data(partition_id, num_partitions, batch_size)
    
    if valloader:
        eval_loss, eval_acc = test(model, valloader, device)
        num_examples = len(valloader.dataset)
    else:
        eval_loss, eval_acc, num_examples = 0.0, 0.0, 0

    return Message(
        content=RecordDict({
            "metrics": MetricRecord({
                "eval_loss": eval_loss, 
                "eval_acc": eval_acc, 
                "num-examples": num_examples
            })
        }), 
        reply_to=msg
    )