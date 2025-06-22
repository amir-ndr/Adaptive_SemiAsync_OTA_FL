import torch
from server import Server
from client import Client
import matplotlib.pyplot as plt
from dataloader import load_mnist, partition_mnist_noniid
from model import CNNMnist
import time
import numpy as np
from torch.utils.data import DataLoader


def evaluate_model(model, test_loader, device='cpu'):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# Main simulation
def main():
    # Parameters
    NUM_CLIENTS = 10
    SIM_TIME = 300  # Simulate for 300 seconds
    EVAL_INTERVAL = 30  # Evaluate every 30 seconds
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {DEVICE}")
    
    # Load data
    train_dataset, test_dataset = load_mnist()
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    client_data_map = partition_mnist_noniid(train_dataset, NUM_CLIENTS)
    
    # Create clients
    clients = []
    for cid in range(NUM_CLIENTS):
        client = Client(
            client_id=cid,
            data_indices=client_data_map[cid],
            model=CNNMnist(),
            fk=np.random.uniform(1e9, 2e9),
            mu_k=1e-27,
            P_max=1.0,
            C=1000,
            Ak=32,
            train_dataset=train_dataset,
            device=DEVICE
        )
        clients.append(client)
    
    # Create server
    global_model = CNNMnist().to(DEVICE)
    server = Server(
        global_model=global_model,
        clients=clients,
        V=10.0,
        sigma_n=0.1,
        tau_cm=0.05,
        T_max=100,
        E_max={client.client_id: 1.0 for client in clients},
        device=DEVICE
    )
    
    # Simulation loop
    start_time = time.time()
    current_time = 0
    last_eval = 0
    accuracies = []
    eval_times = []
    
    while current_time < SIM_TIME:
        # Update client computations
        for client in clients:
            client.compute_gradient(current_time, global_model)
        
        # Attempt aggregation
        selected, power_alloc = server.select_clients(current_time)
        if selected:
            # Calculate round duration
            comp_times = [c.dt_k for c in selected]
            D_t = max(comp_times) + server.tau_cm
            
            # Perform aggregation
            aggregated = server.aggregate(selected, power_alloc)
            server.update_model(aggregated)
            server.update_queues(selected, power_alloc, D_t)
            
            # Update time
            current_time += D_t
            server.aggregation_times.append(current_time)
            print(f"Aggregation at {current_time:.2f}s: Selected {len(selected)} clients")
        else:
            # No clients ready - advance time
            next_avail = min(c.next_available for c in clients)
            current_time = max(current_time + 1, next_avail)
        
        # Periodic evaluation
        if current_time - last_eval > EVAL_INTERVAL:
            acc = evaluate_model(global_model, test_loader, DEVICE)
            accuracies.append(acc)
            eval_times.append(current_time)
            last_eval = current_time
            print(f"[{current_time:.2f}s] Accuracy: {acc:.2f}%")
    
    # Plot results
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.plot(eval_times, accuracies)
    plt.title("Test Accuracy")
    plt.xlabel("Simulation Time (s)")
    
    plt.subplot(132)
    selected_counts = [len(s) for s in server.selected_history]
    plt.plot(server.aggregation_times, selected_counts, 'o-')
    plt.title("Selected Clients per Aggregation")
    plt.xlabel("Simulation Time (s)")
    
    plt.subplot(133)
    queue_values = [max(q.values()) for q in server.queue_history]
    plt.plot(server.aggregation_times, queue_values)
    plt.title("Max Energy Queue")
    plt.xlabel("Simulation Time (s)")
    
    plt.tight_layout()
    plt.savefig("training_results.png")
    plt.show()

if __name__ == "__main__":
    main()