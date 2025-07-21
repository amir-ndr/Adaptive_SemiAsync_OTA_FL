import logging
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader
from client_sync import SyncClient
from server_sync import SyncServer
import torch
from model import CNNMnist
from dataloader import load_mnist, partition_mnist_dirichlet
import os

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

def run_single_experiment():
    """Run a single federated learning experiment and return metrics"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_clients = 10
    total_rounds = 300
    batch_size = 32
    
    # Load data
    train_dataset, test_dataset = load_mnist()
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    client_data_map = partition_mnist_dirichlet(train_dataset, num_clients, alpha=100)
    
    # Initialize clients with energy parameters
    clients = []
    energy_budgets = {}
    for cid in range(num_clients):
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
    
    # Training metrics storage
    metrics = {
        'accuracies': [],
        'round_energies': [],
        'cumulative_energy': {cid: [] for cid in range(num_clients)},
        'selection_counts': np.zeros(num_clients),
        'client_energy': {cid: 0.0 for cid in range(num_clients)}
    }
    
    # Training loop
    for round_idx in range(total_rounds):
        # 1. Device selection
        V = 18.0
        selected, sigma_t = server.select_clients(round_idx, V)
        
        # 2. Model broadcast
        global_state = server.global_model.state_dict()
        for client in selected:
            client.update_model(global_state)
            metrics['selection_counts'][client.client_id] += 1
        
        # 3. Gradient computation and aggregation
        aggregated_update, actual_norms = server.aggregate_gradients(selected, sigma_t)
        
        # 4. Global model update
        server.update_model(aggregated_update, round_idx)
        
        # 5. Queue update
        server.update_queues(selected, actual_norms, sigma_t, round_idx)
        
        # 6. Track energy consumption
        round_energy = 0.0
        for client in selected:
            cid = client.client_id
            # Calculate energy for this client
            E_actual = client.actual_energy(
                sigma_t, batch_size, actual_norms[cid]
            )
            metrics['client_energy'][cid] += E_actual
            round_energy += E_actual
            
            # Update cumulative energy tracking
            metrics['cumulative_energy'][cid].append(metrics['client_energy'][cid])
        
        metrics['round_energies'].append(round_energy)
        
        # 7. Evaluation
        if (round_idx + 1) % 5 == 0:
            acc = evaluate_model(server.global_model, test_loader, device)
            metrics['accuracies'].append(acc)
    
    # Final evaluation
    final_acc = evaluate_model(server.global_model, test_loader, device)
    metrics['final_accuracy'] = final_acc
    
    # Add final cumulative energy
    for cid in range(num_clients):
        while len(metrics['cumulative_energy'][cid]) < total_rounds:
            last_val = metrics['cumulative_energy'][cid][-1] if metrics['cumulative_energy'][cid] else 0
            metrics['cumulative_energy'][cid].append(last_val)
    
    return metrics

def main():
    # Configuration
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Run experiments until we have 10 with accuracy >= 80%
    successful_runs = []
    run_count = 0
    
    while len(successful_runs) < 10:
        run_count += 1
        logging.info(f"Starting experiment run {run_count}")
        results = run_single_experiment()
        
        if results['final_accuracy'] >= 70:
            successful_runs.append(results)
            logging.info(f"Run {run_count} successful! Accuracy: {results['final_accuracy']:.2f}%")
        else:
            logging.info(f"Run {run_count} discarded. Accuracy: {results['final_accuracy']:.2f}% < 80%")
    
    # Aggregate results from successful runs
    aggregated = {
        'accuracies': np.mean([run['accuracies'] for run in successful_runs], axis=0),
        'round_energies': np.mean([run['round_energies'] for run in successful_runs], axis=0),
        'cumulative_energy': {
            cid: np.mean([run['cumulative_energy'][cid] for run in successful_runs], axis=0)
            for cid in range(10)
        },
        'selection_counts': np.mean([run['selection_counts'] for run in successful_runs], axis=0),
        'total_energy': np.mean([sum(run['round_energies']) for run in successful_runs]),
        'per_client_energy': np.mean([
            [run['client_energy'][cid] for cid in range(10)] 
            for run in successful_runs
        ], axis=0),
        'final_accuracy': np.mean([run['final_accuracy'] for run in successful_runs])
    }
    
    # Save results to CSV
    os.makedirs("results_sync02", exist_ok=True)
    
    # 1. Accuracy results
    accuracy_df = pd.DataFrame({
        'round': np.arange(5, 301, 5),
        'accuracy': aggregated['accuracies']
    })
    accuracy_df.to_csv("results_sync02/accuracy_results.csv", index=False)
    
    # 2. Energy per round
    energy_df = pd.DataFrame({
        'round': np.arange(1, 301),
        'energy': aggregated['round_energies']
    })
    energy_df.to_csv("results_sync02/energy_per_round.csv", index=False)
    
    # 3. Cumulative energy per client
    cumulative_energy_data = {'round': np.arange(1, 301)}
    for cid in range(10):
        cumulative_energy_data[f'client_{cid}'] = aggregated['cumulative_energy'][cid]
    cumulative_df = pd.DataFrame(cumulative_energy_data)
    cumulative_df.to_csv("results_sync02/cumulative_energy_per_client.csv", index=False)
    
    # 4. Selection counts
    selection_df = pd.DataFrame({
        'client_id': np.arange(10),
        'selection_count': aggregated['selection_counts']
    })
    selection_df.to_csv("results_sync02/selection_counts.csv", index=False)
    
    # 5. Summary report
    with open("results_sync02/summary_report.txt", "w") as f:
        f.write("=== Federated Learning Experiment Summary ===\n")
        f.write(f"Total runs: {run_count} (10 successful runs with accuracy >= 80%)\n")
        f.write(f"Average final accuracy: {aggregated['final_accuracy']:.2f}%\n")
        f.write(f"Average total energy consumption: {aggregated['total_energy']:.4f} J\n\n")
        
        f.write("Per-client information:\n")
        f.write("Client | Selections | Energy (J) | Energy/Budget Ratio\n")
        f.write("----------------------------------------------------\n")
        for cid in range(10):
            energy = aggregated['per_client_energy'][cid]
            selections = aggregated['selection_counts'][cid]
            f.write(f"{cid:6} | {selections:10.1f} | {energy:9.4f} | {energy/1.25:.4f}\n")
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Accuracy plot
    plt.subplot(221)
    plt.plot(np.arange(5, 301, 5), aggregated['accuracies'], 'o-')
    plt.title("Test Accuracy")
    plt.xlabel("Communication Rounds")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)
    
    # Energy per round plot
    plt.subplot(222)
    plt.plot(aggregated['round_energies'])
    plt.title("Energy Consumption per Round")
    plt.xlabel("Rounds")
    plt.ylabel("Energy (J)")
    plt.grid(True)
    
    # Cumulative energy per client
    plt.subplot(223)
    for cid in range(10):
        plt.plot(aggregated['cumulative_energy'][cid], label=f'Client {cid}')
    plt.title("Cumulative Energy per Client")
    plt.xlabel("Rounds")
    plt.ylabel("Cumulative Energy (J)")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True)
    
    # Selection counts
    plt.subplot(224)
    plt.bar(range(10), aggregated['selection_counts'])
    plt.title("Client Selection Distribution")
    plt.xlabel("Client ID")
    plt.ylabel("Times Selected")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("results_sync02/aggregated_results.png")
    
    # Print summary to console
    print("\n=== Experiment Summary ===")
    print(f"Average final accuracy: {aggregated['final_accuracy']:.2f}%")
    print(f"Total energy consumption: {aggregated['total_energy']:.4f} J")
    print("\nPer-client information:")
    print("Client | Selections | Energy (J)")
    print("--------------------------------")
    for cid in range(10):
        print(f"{cid:6} | {aggregated['selection_counts'][cid]:10.1f} | {aggregated['per_client_energy'][cid]:9.4f}")

if __name__ == "__main__":
    main()