"""pytorchexample: A Flower / PyTorch app."""

from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
from phase3.task import Net, load_centralized_testset, test
import torch
import time

app = ServerApp()

@app.main()
def main(grid: Grid, context: Context) -> None:
    
    # 读取配置
    num_rounds = context.run_config["num-server-rounds"]
    lr = context.run_config["learning-rate"]
    
    print(f"--- Starting HFL Server for {num_rounds} Rounds ---")

    # 初始化全局模型
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())
    
    # 定义 Cloud 端策略
    # 【注意】参数名已更新适配 Flower 1.23+ (ServerApp 模式)
    strategy = FedAvg(
        fraction_train=1.0,       # 原 fraction_fit
        fraction_evaluate=0,      # 暂时禁用
        min_train_nodes=2,        # 原 min_fit_clients
        min_evaluate_nodes=0,     # 暂时禁用
        min_available_nodes=2,    # 原 min_available_clients
    )

    # 启动 Cloud Server
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate, 
    )
    
    print("\n--- Federated Learning Finished ---")
    # 保存结果
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model_hierarchical.pt")

def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """在 Cloud 端使用全局测试集进行评估"""
    print(f"\n[Global Server] Round {server_round} Evaluation Started...")
    start_time = time.time()
    
    model = Net()
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    testloader = load_centralized_testset()
    
    # 获取详细指标
    metrics = test(model, testloader, device)
    
    eval_time = time.time() - start_time
    
    print(f"[Global Server] Round {server_round} Result:")
    print(f"   >>> Accuracy : {metrics['accuracy']:.4f}")
    print(f"   >>> F1 Score : {metrics['f1']:.4f}")
    print(f"   >>> Precision: {metrics['precision']:.4f}")
    print(f"   >>> Recall   : {metrics['recall']:.4f}")
    print(f"   >>> Loss     : {metrics['loss']:.4f}")
    print(f"   >>> Time     : {eval_time:.2f}s")
    
    # 返回所有指标
    return MetricRecord(metrics)