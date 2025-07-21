import logging
import time
import numpy as np
import matplotlib.pyplot as plt
import csv
from torch.utils.data import DataLoader
from client_sync import SyncClient
from server_sync import SyncServer
import torch
from model import CNNMnist
from dataloader import load_mnist, partition_mnist_noniid

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

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

def run_noise_experiment(noise_level, num_runs=5, total_rounds=100):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_clients = 10
    batch_size = 32
    
    all_accuracies = []
    
    for run_id in range(num_runs):
        logging.info(f"\n=== Noise {noise_level} - Run {run_id+1}/{num_runs} ===")
        
        # Load data
        train_dataset, test_dataset = load_mnist()
        test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
        client_data_map = partition_mnist_noniid(train_dataset, num_clients)
        
        # Initialize clients with energy parameters
        clients = []
        energy_budgets = {}
        for cid in range(num_clients):
            # Energy parameters (paper-inspired values)
            en = np.random.uniform(1e-6, 5e-6)  # J/sample
            fk = np.random.uniform(1e9, 3e9)     # CPU frequency
            P_max = 2.0                           # Max transmit power
            
            clients.append(SyncClient(
                client_id=cid,
                data_indices=client_data_map[cid],
                model=CNNMnist(),
                fk=fk,
                en=en,
                P_max=P_max,
                train_dataset=train_dataset,
                device=device
            ))
            energy_budgets[cid] = np.random.uniform(0.5, 2.0)  # Joules
        
        # Initialize server with specified noise level
        server = SyncServer(
            global_model=CNNMnist(),
            clients=clients,
            total_rounds=total_rounds,
            batch_size=batch_size,
            gamma0=10.0,       # Target SNR
            sigma_n=noise_level,  # Use specified noise level
            G2=1.0,            # Gradient variance bound
            l_smooth=0.1,      # Smoothness constant
            energy_budgets=energy_budgets,
            device=device
        )
        
        # Initial gradient computation for EST-P
        global_state = server.global_model.state_dict()
        for client in clients:
            client.update_model(global_state)
            client.compute_gradient(batch_size)
        
        # Training loop
        for round_idx in range(total_rounds):
            # 1. Device selection
            V = 18.0
            selected, sigma_t = server.select_clients(round_idx, V)
            
            # 2. Model broadcast
            global_state = server.global_model.state_dict()
            for client in selected:
                client.update_model(global_state)
            
            # 3. Gradient computation and aggregation
            aggregated_update, actual_norms = server.aggregate_gradients(selected, sigma_t)
            
            # 4. Global model update
            server.update_model(aggregated_update, round_idx)
            
            # 5. Queue update
            server.update_queues(selected, actual_norms, sigma_t, round_idx)
        
        # Final evaluation
        final_acc = evaluate_model(server.global_model, test_loader, device)
        all_accuracies.append(final_acc)
        logging.info(f"Noise {noise_level} - Run {run_id+1}: Accuracy = {final_acc:.2f}%")
    
    return np.mean(all_accuracies)

def main():
    # Noise levels to test
    noise_levels = [0.001, 0.07, 0.5, 0.9]
    results = []
    
    # Run experiments for each noise level
    for noise in noise_levels:
        avg_acc = run_noise_experiment(noise, num_runs=5, total_rounds=100)
        results.append((noise, avg_acc))
        logging.info(f"Noise {noise}: Average Accuracy = {avg_acc:.2f}%")
    
    # Save results to CSV
    with open("noise_vs_accuracy_sync.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Noise Level", "Average Accuracy"])
        for noise, acc in results:
            writer.writerow([noise, acc])
    
    # Plot results
    plt.figure(figsize=(10, 6))
    noise_values, acc_values = zip(*results)
    plt.semilogx(noise_values, acc_values, 'o-', linewidth=2)
    plt.title("Accuracy vs Noise Level (Synchronous OTA-FL)")
    plt.xlabel("Noise Level (σ_n)")
    plt.ylabel("Final Accuracy (%)")
    plt.grid(True, which="both", ls="--")
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig("noise_vs_accuracy_sync.png", dpi=300)
    plt.show()
    
    # Print summary
    print("\n=== Final Results ===")
    print("Noise Level | Avg Accuracy")
    for noise, acc in results:
        print(f"{noise:.4f}     | {acc:.2f}%")

if __name__ == "__main__":
    main()