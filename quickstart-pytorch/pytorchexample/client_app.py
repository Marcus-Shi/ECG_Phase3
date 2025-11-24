# marcus-shi/ecg_phase3/ECG_Phase3-ec1bb3f03abae1d74da6110dad4af62b3796a611/quickstart-pytorch/pytorchexample/client_app.py

import torch
import copy
import time  # 引入时间模块
from collections import OrderedDict
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from pytorchexample.task import Net, load_data, train_one_epoch, test

app = ClientApp()

def aggregate_weights(weights_list, num_examples_list):
    """加权平均聚合"""
    total_examples = sum(num_examples_list)
    weighted_weights = [
        [layer * num for layer in weights] 
        for weights, num in zip(weights_list, num_examples_list)
    ]
    aggregated_weights = [
        sum(layers) / total_examples for layers in zip(*weighted_weights)
    ]
    return aggregated_weights

@app.train()
def train(msg: Message, context: Context):
    """分层联邦学习 Edge Server 训练逻辑"""
    
    # 1. 获取配置与初始化
    start_time = time.time()  # 开始计时
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    
    lr = msg.content["config"]["lr"]
    kappa_1 = int(context.run_config["kappa-1"])
    kappa_2 = int(context.run_config["kappa-2"])
    batch_size = int(context.run_config["batch-size"])

    print(f"[Cluster {partition_id}] Starting Hierarchical Training... (Kappa1={kappa_1}, Kappa2={kappa_2})")

    # 2. 加载数据
    patient_dataloaders, _ = load_data(partition_id, num_partitions, batch_size)
    num_patients = len(patient_dataloaders)
    print(f"[Cluster {partition_id}] Loaded data for {num_patients} simulated patients.")

    # 3. 初始化 Edge Model
    global_state_dict = msg.content["arrays"].to_torch_state_dict()
    edge_model = Net()
    edge_model.load_state_dict(global_state_dict)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    edge_model.to(device)

    # 4. Hierarchical Training Loops
    for edge_round in range(kappa_2):
        iter_start = time.time()
        patient_weights_list = []
        patient_num_examples = []
        
        current_edge_weights = [val.cpu().numpy() for val in edge_model.state_dict().values()]
        
        # 遍历病人 (模拟 Cross-Device)
        for i, (patient_id, loader) in enumerate(patient_dataloaders.items()):
            # A. 下载 Edge 模型
            patient_model = Net()
            params_dict = zip(edge_model.state_dict().keys(), current_edge_weights)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            patient_model.load_state_dict(state_dict)
            patient_model.to(device)
            
            # B. 本地训练
            for epoch in range(kappa_1):
                train_one_epoch(patient_model, loader, device, lr)
            
            # C. 收集权重
            p_weights = [val.cpu().numpy() for val in patient_model.state_dict().values()]
            patient_weights_list.append(p_weights)
            patient_num_examples.append(len(loader.dataset))
        
        # 5. Edge Aggregation
        if patient_weights_list:
            new_edge_weights = aggregate_weights(patient_weights_list, patient_num_examples)
            params_dict = zip(edge_model.state_dict().keys(), new_edge_weights)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            edge_model.load_state_dict(state_dict)
            
        print(f"[Cluster {partition_id}] Edge Round {edge_round+1}/{kappa_2} completed in {time.time()-iter_start:.2f}s")

    # 6. 结束计时与封装
    total_runtime = time.time() - start_time
    total_examples = sum([len(l.dataset) for l in patient_dataloaders.values()])
    print(f"[Cluster {partition_id}] Training Finished. Total Time: {total_runtime:.2f}s")

    final_edge_weights = edge_model.state_dict()
    model_record = ArrayRecord(final_edge_weights)
    
    # 将运行时间也放入 metrics 返回
    metrics = {
        "num-examples": total_examples,
        "cluster_id": partition_id,
        "runtime": total_runtime
    }
    
    return Message(
        content=RecordDict({"arrays": model_record, "metrics": MetricRecord(metrics)}), 
        reply_to=msg
    )

@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Cluster Leader 评估逻辑"""
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    
    print(f"[Cluster {partition_id}] Starting Evaluation...")
    
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    _, valloader = load_data(partition_id, num_partitions, batch_size)
    
    if valloader:
        # 使用更新后的 test 函数获取详细指标
        metrics_dict = test(model, valloader, device)
        num_examples = len(valloader.dataset)
        print(f"[Cluster {partition_id}] Eval Results: Acc={metrics_dict['accuracy']:.4f}, F1={metrics_dict['f1']:.4f}")
    else:
        metrics_dict = {"accuracy": 0.0, "loss": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}
        num_examples = 0
        print(f"[Cluster {partition_id}] No validation data.")

    # 将所有指标放入 MetricRecord
    out_metrics = {k: v for k, v in metrics_dict.items()}
    out_metrics["num-examples"] = num_examples
    
    return Message(
        content=RecordDict({"metrics": MetricRecord(out_metrics)}), 
        reply_to=msg
    )