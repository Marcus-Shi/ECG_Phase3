# marcus-shi/ecg_phase3/ECG_Phase3-ec1bb3f03abae1d74da6110dad4af62b3796a611/quickstart-pytorch/pytorchexample/server_app.py

from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
from pytorchexample.task import Net, load_centralized_testset, test
import torch
import time

app = ServerApp()

@app.main()
def main(grid: Grid, context: Context) -> None:
    
    num_rounds = context.run_config["num-server-rounds"]
    lr = context.run_config["learning-rate"]
    
    print(f"--- Starting HFL Server for {num_rounds} Rounds ---")

    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())
    
    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
    )

    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate, 
    )
    
    print("\n--- Federated Learning Finished ---")
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