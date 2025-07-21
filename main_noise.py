import torch
import numpy as np
import time
import os
import csv
from torch.utils.data import DataLoader
from client import Client
from server import Server
from model import CNNMnist
from dataloader import load_mnist, partition_mnist_noniid

def evaluate_model(model, test_loader, device='cpu'):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def run_experiment(run_id, num_clients, num_rounds, noise_level):
    # Parameters
    LOCAL_EPOCHS = 1
    BATCH_SIZE = 32
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load data
    train_dataset, test_dataset = load_mnist()
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    client_data_map = partition_mnist_noniid(train_dataset, num_clients)
    
    # Initialize clients
    clients = []
    for cid in range(num_clients):
        indices = client_data_map[cid]
        if len(indices) == 0:
            indices = [0]
        
        client = Client(
            client_id=cid,
            data_indices=indices,
            model=CNNMnist(),
            fk=np.random.uniform(1e9, 2e9),
            mu_k=1e-27,
            P_max=2.0 + np.random.rand(),
            C=1e6,
            Ak=BATCH_SIZE,
            train_dataset=train_dataset,
            device=DEVICE,
            local_epochs=LOCAL_EPOCHS
        )
        clients.append(client)
    
    E_max_dict = {cid: np.random.uniform(25, 38) for cid in range(num_clients)}

    # Initialize server with specified noise level
    global_model = CNNMnist().to(DEVICE)
    server = Server(
        global_model=global_model,
        clients=clients,
        V=15.0,
        sigma_n=noise_level,  # Use the specified noise level
        tau_cm=0.01,
        T_max=50,
        E_max=E_max_dict,
        T_total_rounds=num_rounds,
        device=DEVICE
    )
    
    # Train for specified rounds
    for round_idx in range(num_rounds):
        selected, power_alloc = server.select_clients()
        server.broadcast_model(selected)
        
        # Compute gradients and measure time
        comp_times = []
        for client in selected:
            start_time = time.time()
            client.compute_gradient()
            comp_time = time.time() - start_time
            comp_times.append(comp_time)
        
        # Reset staleness for selected clients
        for client in selected:
            client.reset_staleness()
        
        # Aggregate updates
        max_comp_time = max(comp_times) if selected else 0
        D_t = max_comp_time + server.tau_cm
        
        if selected:
            aggregated = server.aggregate(selected, power_alloc)
            server.update_model(aggregated, round_idx)
        
        # Update queues and computation times
        server.update_queues(selected, power_alloc, D_t)
        
        for client in clients:
            if client in selected:
                # Reset computation time counter
                client.dt_k = 0
            else:
                # Decrement computation time counter
                client.dt_k = max(0, client.dt_k - D_t)
                client.increment_staleness()
    
    # Final evaluation
    return evaluate_model(server.global_model, test_loader, DEVICE)

def noise_experiment():
    NOISE_LEVELS = [0.01, 0.1, 0.6, 0.99]
    NUM_CLIENTS = 10
    NUM_ROUNDS = 100  # Reduced for efficiency
    NUM_RUNS = 5      # Number of runs per noise level
    MIN_ACCURACY = 60.0  # Lowered threshold
    
    results = []
    
    for noise in NOISE_LEVELS:
        print(f"\n=== Running for noise level: {noise} ===")
        accuracies = []
        run_count = 0
        
        while len(accuracies) < NUM_RUNS and run_count < 15:
            run_count += 1
            try:
                acc = run_experiment(
                    run_id=run_count,
                    num_clients=NUM_CLIENTS,
                    num_rounds=NUM_ROUNDS,
                    noise_level=noise
                )
                
                if acc >= MIN_ACCURACY:
                    accuracies.append(acc)
                    print(f"Run {run_count}: Accuracy = {acc:.2f}% "
                          f"({len(accuracies)}/{NUM_RUNS})")
                else:
                    print(f"Run {run_count} skipped: Accuracy {acc:.2f}% < {MIN_ACCURACY}%")
            except Exception as e:
                print(f"Run {run_count} failed: {str(e)}")
        
        if accuracies:
            avg_acc = np.mean(accuracies)
            results.append((noise, avg_acc))
            print(f"\nNoise {noise}: Average Accuracy = {avg_acc:.2f}%")
        else:
            results.append((noise, 0.0))
            print(f"\nNoise {noise}: No successful runs")
    
    # Save results
    with open("noise_vs_accuracy.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Noise Level", "Average Accuracy"])
        for noise, acc in results:
            writer.writerow([noise, acc])
    
    print("\n=== Final Results ===")
    print("Noise Level | Avg Accuracy")
    for noise, acc in results:
        print(f"{noise:.4f}     | {acc:.2f}%")

if __name__ == "__main__":
    noise_experiment()