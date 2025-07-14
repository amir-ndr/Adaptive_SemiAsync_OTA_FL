import logging
import time
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from client_sync import SyncClient
from server_sync import SyncServer
import torch
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

def main():
    # Configuration
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_clients = 10
    total_rounds = 300
    batch_size = 32
    
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
    
    # Initialize server with algorithm parameters
    server = SyncServer(
        global_model=CNNMnist(),
        clients=clients,
        total_rounds=total_rounds,
        batch_size=batch_size,
        gamma0=10.0,       # Target SNR
        sigma_n=0.05,      # Noise std dev
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
        logging.info(f"Client {client.client_id} initial grad norm: {client.last_gradient_norm:.4f}")
    
    # Training loop
    accuracies = []
    round_times = []
    selection_counts = np.zeros(num_clients)
    energy_ratios = []
    snr_history = []
    
    for round_idx in range(total_rounds):
        start_time = time.time()
        logging.info(f"\n=== Round {round_idx+1}/{total_rounds} ===")
        
        # 1. Device selection
        V = 18.0 #* (0.9 ** (round_idx // 30))  # Lyapunov parameter
        selected, sigma_t = server.select_clients(round_idx, V)
        
        # 2. Model broadcast
        global_state = server.global_model.state_dict()
        for client in selected:
            client.update_model(global_state)
            selection_counts[client.client_id] += 1
        
        # 3. Gradient computation and aggregation
        aggregated_update, actual_norms = server.aggregate_gradients(selected, sigma_t)
        snr_history.append(server.actual_snr_history[-1])
        
        # 4. Global model update
        server.update_model(aggregated_update, round_idx)
        
        # 5. Queue update
        server.update_queues(selected, actual_norms, sigma_t, round_idx)
        
        # 6. Track energy ratios
        current_ratios = []
        for cid in range(num_clients):
            consumed = server.energy_consumed.get(cid, 0)
            budget = energy_budgets[cid]
            current_ratios.append(consumed / budget)
        energy_ratios.append(current_ratios)
        
        # 7. Evaluation
        round_time = time.time() - start_time
        round_times.append(round_time)
        
        if (round_idx + 1) % 10 == 0:
            acc = evaluate_model(server.global_model, test_loader, device)
            accuracies.append(acc)
            logging.info(f"Round {round_idx+1} accuracy: {acc:.2f}%")
            logging.info(f"SNR: {server.actual_snr_history[-1]:.2f} dB")
    
    # Final evaluation and visualization
    final_acc = evaluate_model(server.global_model, test_loader, device)
    logging.info(f"\n=== Training Complete ===")
    logging.info(f"Final accuracy: {final_acc:.2f}%")
    
    # Visualization
    plt.figure(figsize=(15, 12))
    
    # Accuracy plot
    plt.subplot(231)
    plt.plot(np.arange(10, total_rounds+1, 10), accuracies, 'o-')
    plt.title("Test Accuracy")
    plt.xlabel("Communication Rounds")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)
    
    # Client selection distribution
    plt.subplot(232)
    plt.bar(range(num_clients), selection_counts)
    plt.title("Client Selection Distribution")
    plt.xlabel("Client ID")
    plt.ylabel("Times Selected")
    plt.grid(True)
    
    # Energy consumption ratio
    plt.subplot(233)
    energy_ratio = [server.energy_consumed[cid] / energy_budgets[cid] 
                   for cid in range(num_clients)]
    plt.bar(range(num_clients), energy_ratio)
    plt.axhline(1.0, color='r', linestyle='--')
    plt.title("Energy Consumption Ratio")
    plt.xlabel("Client ID")
    plt.ylabel("Energy Used / Budget")
    plt.grid(True)
    
    # Virtual queue dynamics
    plt.subplot(234)
    queue_data = np.array([list(q.values()) for q in server.queue_history])
    plt.plot(queue_data.mean(axis=1), label='Mean')
    plt.plot(queue_data.max(axis=1), label='Max')
    plt.title("Virtual Queue Dynamics")
    plt.xlabel("Rounds")
    plt.ylabel("Queue Value")
    plt.legend()
    plt.grid(True)
    
    # SNR monitoring
    plt.subplot(235)
    plt.plot(snr_history)
    plt.axhline(server.gamma0, color='r', linestyle='--')
    plt.title("Actual SNR vs Target")
    plt.xlabel("Rounds")
    plt.ylabel("SNR (dB)")
    plt.yscale('log')
    plt.grid(True)
    
    # Energy ratio evolution
    plt.subplot(236)
    energy_ratio_data = np.array(energy_ratios)
    plt.plot(energy_ratio_data.mean(axis=1), label='Mean')
    plt.plot(energy_ratio_data.max(axis=1), label='Max')
    plt.axhline(1.0, color='r', linestyle='--')
    plt.title("Energy Ratio Evolution")
    plt.xlabel("Rounds")
    plt.ylabel("Energy Used / Budget")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("sync_ota_results.png")
    plt.show()

if __name__ == "__main__":
    main()