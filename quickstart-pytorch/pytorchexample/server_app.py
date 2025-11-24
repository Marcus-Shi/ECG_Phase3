# marcus-shi/ecg_phase3/ECG_Phase3-ec1bb3f03abae1d74da6110dad4af62b3796a611/quickstart-pytorch/pytorchexample/server_app.py

from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
from pytorchexample.task import Net, load_centralized_testset, test
import torch

app = ServerApp()

@app.main()
def main(grid: Grid, context: Context) -> None:
    
    # 读取配置
    num_rounds = context.run_config["num-server-rounds"]
    lr = context.run_config["learning-rate"]
    
    # 初始化全局模型
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())
    
    # 定义 Cloud 端策略
    strategy = FedAvg(
        fraction_fit=1.0,       # 每次都让所有 Cluster Leader 参与
        fraction_evaluate=1.0,
        min_fit_clients=2,
    )

    # 启动 Cloud Server
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate, 
    )
    
    # 保存结果
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model_hierarchical.pt")

def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """在 Cloud 端使用全局测试集进行评估"""
    model = Net()
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    testloader = load_centralized_testset()
    loss, acc = test(model, testloader, device)
    
    print(f"Round {server_round} Global Accuracy: {acc:.4f}")
    return MetricRecord({"accuracy": acc, "loss": loss})