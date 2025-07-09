import torch
import numpy as np
import time
import logging
import matplotlib.pyplot as plt
import copy
from torch.utils.data import DataLoader
from dataloader import load_mnist, partition_mnist_noniid
from model import CNNMnist
from client import Client
from server import Server
from client_otaenergy import SyncOTAClient
from server_otaenergy import SyncOTAServer

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

def run_semi_async_experiment(config):
    """Run semi-asynchronous OTA FL experiment"""
    # Unpack config
    device = config['device']
    num_clients = config['num_clients']
    num_rounds = config['num_rounds']
    batch_size = config['batch_size']
    local_epochs = config['local_epochs']
    
    # Load data
    train_dataset, test_dataset = load_mnist()
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    client_data_map = partition_mnist_noniid(train_dataset, num_clients)
    
    # Initialize clients
    clients = []
    for cid in range(num_clients):
        indices = client_data_map[cid]
        if len(indices) == 0:
            indices = [0]  # Prevent empty datasets
            
        client = Client(
            client_id=cid,
            data_indices=indices,
            model=CNNMnist(),
            fk=np.random.uniform(1e9, 2e9),
            mu_k=1e-27,
            P_max=2.0 + np.random.rand(),
            C=1e6,
            Ak=batch_size,
            train_dataset=train_dataset,
            device=device,
            local_epochs=local_epochs
        )
        clients.append(client)
    
    # Energy budgets
    E_max_dict = {cid: np.random.uniform(25, 38) for cid in range(num_clients)}
    
    # Initialize server
    global_model = CNNMnist().to(device)
    server = Server(
        global_model=global_model,
        clients=clients,
        V=config['V_async'],
        sigma_n=config['sigma_n'],
        tau_cm=config['tau_cm'],
        T_max=config['T_max'],
        E_max=E_max_dict,
        T_total_rounds=num_rounds,
        device=device
    )
    
    # Training metrics
    metrics = {
        'accuracies': [],
        'round_durations': [],
        'energy_queues': [],
        'avg_staleness': [],
        'selected_counts': [],
        'selection_counts': {cid: 0 for cid in range(num_clients)},
        'cumulative_energy': {cid: 0 for cid in range(num_clients)}
    }
    
    for round_idx in range(num_rounds):
        round_start = time.time()
        
        # Select clients and broadcast
        selected, power_alloc = server.select_clients()
        selected_ids = [c.client_id for c in selected]
        metrics['selected_counts'].append(len(selected))
        
        # Update selection counts
        for cid in selected_ids:
            metrics['selection_counts'][cid] += 1
            
        server.broadcast_model(selected)
        
        # Compute gradients
        comp_times = []
        for client in selected:
            start_comp = time.time()
            client.compute_gradient()
            comp_time = time.time() - start_comp
            comp_times.append(comp_time)
        
        # Reset staleness for selected
        for client in selected:
            client.reset_staleness()
        
        # Aggregate and update
        D_t = max(comp_times) + server.tau_cm if selected else 0
        if selected:
            aggregated = server.aggregate(selected, power_alloc)
            server.update_model(aggregated, round_idx)
        
        # Update queues and client states
        server.update_queues(selected, power_alloc, D_t)
        for client in clients:
            if client in selected:
                client.reset_computation()
            else:
                client.dt_k = max(0, client.dt_k - D_t)
                client.increment_staleness()
        
        # Record metrics
        metrics['avg_staleness'].append(np.mean([c.tau_k for c in clients]))
        metrics['round_durations'].append(D_t)
        
        # Energy tracking
        for client in selected:
            # Simplified energy tracking
            E_comp = client.mu_k * client.fk**2 * client.C * client.Ak
            E_comm = (power_alloc[client.client_id] * client.gradient_norm / abs(client.h_t_k))**2
            metrics['cumulative_energy'][client.client_id] += E_comp + E_comm
        
        # Periodic evaluation
        if (round_idx + 1) % 5 == 0 or round_idx == 0:
            acc = evaluate_model(server.global_model, test_loader, device)
            metrics['accuracies'].append(acc)
        
        # Queue tracking
        metrics['energy_queues'].append(max(server.Q_e.values()))
        
        # Log progress
        if (round_idx + 1) % 10 == 0:
            print(f"Round {round_idx+1}/{num_rounds} | "
                  f"Acc: {metrics['accuracies'][-1] if metrics['accuracies'] else 'N/A':.2f}% | "
                  f"Duration: {D_t:.4f}s")
    
    # Final evaluation
    final_acc = evaluate_model(server.global_model, test_loader, device)
    metrics['accuracies'].append(final_acc)
    
    return metrics, server.global_model

def run_sync_ota_experiment(config):
    """Run synchronous OTA FEEL baseline experiment"""
    # Unpack config
    device = config['device']
    num_clients = config['num_clients']
    num_rounds = config['num_rounds']
    batch_size = config['batch_size']
    
    # Load data
    train_dataset, test_dataset = load_mnist()
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    client_data_map = partition_mnist_noniid(train_dataset, num_clients)
    
    # Initialize clients
    clients = []
    for cid in range(num_clients):
        indices = list(client_data_map[cid])  # Ensure list type
        if len(indices) == 0:
            indices = [0]  # Prevent empty datasets
            
        client = SyncOTAClient(
            client_id=cid,
            data_indices=indices,
            model=CNNMnist(),
            en=1e-6,  # Energy per sample (J)
            Lb=batch_size,
            train_dataset=train_dataset,
            device=device
        )
        clients.append(client)
    
    # Energy budgets
    E_bars = {cid: np.random.uniform(25, 38) for cid in range(num_clients)}
    
    # Initialize server
    global_model = CNNMnist().to(device)
    server = SyncOTAServer(
        global_model=global_model,
        clients=clients,
        E_bars=E_bars,
        T=num_rounds,
        V=config['V_sync'],
        gamma0=config['gamma0'],
        sigma0=config['sigma_n'],
        l=config['l'],
        G=config['G'],
        Lb=batch_size,
        eta_base=0.1,
        decay_rate=0.95,
        device=device
    )
    
    # Training metrics
    metrics = {
        'accuracies': [],
        'round_durations': [],
        'energy_queues': [],
        'selected_counts': [],
        'selection_counts': {cid: 0 for cid in range(num_clients)},
        'cumulative_energy': {cid: 0 for cid in range(num_clients)}
    }
    
    # Initialization round
    server.initialize_clients()
    
    for round_idx in range(num_rounds):
        round_start = time.time()
        
        # Run round
        round_metrics = server.run_round(round_idx)
        metrics['selected_counts'].append(len(round_metrics['selected']))
        metrics['round_durations'].append(time.time() - round_start)
        
        # Update selection counts
        for cid in round_metrics['selected']:
            metrics['selection_counts'][cid] += 1
        
        # Update energy tracking
        for cid, energy in round_metrics.get('actual_energies', {}).items():
            metrics['cumulative_energy'][cid] += energy
        
        # Queue tracking
        metrics['energy_queues'].append(max(server.qn.values()))
        
        # Periodic evaluation
        if (round_idx + 1) % 5 == 0 or round_idx == 0:
            acc = evaluate_model(server.global_model, test_loader, device)
            metrics['accuracies'].append(acc)
        
        # Log progress
        if (round_idx + 1) % 10 == 0:
            print(f"Round {round_idx+1}/{num_rounds} | "
                  f"Acc: {metrics['accuracies'][-1] if metrics['accuracies'] else 'N/A':.2f}% | "
                  f"Selected: {len(round_metrics['selected'])} clients")
    
    # Final evaluation
    final_acc = evaluate_model(server.global_model, test_loader, device)
    metrics['accuracies'].append(final_acc)
    
    return metrics, server.global_model

def plot_comparison_results(async_metrics, sync_metrics, config):
    """Plot comparative results from both experiments"""
    plt.figure(figsize=(18, 15))
    
    # Accuracy comparison
    plt.subplot(231)
    async_eval_rounds = [5*i for i in range(len(async_metrics['accuracies']))]
    sync_eval_rounds = [5*i for i in range(len(sync_metrics['accuracies']))]
    plt.plot(async_eval_rounds, async_metrics['accuracies'], 'o-', label='Semi-Async OTA')
    plt.plot(sync_eval_rounds, sync_metrics['accuracies'], 's-', label='Sync OTA')
    plt.title("Test Accuracy Comparison", fontsize=14)
    plt.xlabel("Communication Rounds", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)
    
    # Client selection pattern
    plt.subplot(232)
    plt.plot(async_metrics['selected_counts'], 'b-', label='Semi-Async')
    plt.plot(sync_metrics['selected_counts'], 'r-', label='Sync')
    plt.title("Selected Clients per Round", fontsize=14)
    plt.xlabel("Rounds", fontsize=12)
    plt.ylabel("Number of Clients", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)
    
    # Energy queue comparison
    plt.subplot(233)
    plt.plot(async_metrics['energy_queues'], 'b-', label='Semi-Async')
    plt.plot(sync_metrics['energy_queues'], 'r-', label='Sync')
    plt.title("Max Energy Queue Value", fontsize=14)
    plt.xlabel("Rounds", fontsize=12)
    plt.ylabel("Queue Value", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)
    
    # Round duration comparison
    plt.subplot(234)
    plt.plot(async_metrics['round_durations'], 'b-', label='Semi-Async')
    plt.plot(sync_metrics['round_durations'], 'r-', label='Sync')
    plt.title("Round Duration Comparison", fontsize=14)
    plt.xlabel("Rounds", fontsize=12)
    plt.ylabel("Time (s)", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)
    
    # Client selection distribution
    plt.subplot(235)
    async_counts = [async_metrics['selection_counts'][i] for i in range(config['num_clients'])]
    sync_counts = [sync_metrics['selection_counts'][i] for i in range(config['num_clients'])]
    x = np.arange(config['num_clients'])
    width = 0.35
    plt.bar(x - width/2, async_counts, width, label='Semi-Async')
    plt.bar(x + width/2, sync_counts, width, label='Sync')
    plt.title("Client Selection Distribution", fontsize=14)
    plt.xlabel("Client ID", fontsize=12)
    plt.ylabel("Times Selected", fontsize=12)
    plt.xticks(x)
    plt.legend(fontsize=12)
    plt.grid(True)
    
    # Cumulative energy comparison
    plt.subplot(236)
    async_energy = [async_metrics['cumulative_energy'][i] for i in range(config['num_clients'])]
    sync_energy = [sync_metrics['cumulative_energy'][i] for i in range(config['num_clients'])]
    plt.bar(x - width/2, async_energy, width, label='Semi-Async')
    plt.bar(x + width/2, sync_energy, width, label='Sync')
    plt.title("Cumulative Energy per Client", fontsize=14)
    plt.xlabel("Client ID", fontsize=12)
    plt.ylabel("Energy (J)", fontsize=12)
    plt.xticks(x)
    plt.legend(fontsize=12)
    plt.grid(True)
    
    plt.tight_layout(pad=3.0)
    plt.savefig("sync_vs_async_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("fl_comparison.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting FL comparison experiment")

    # Experiment configuration
    config = {
        'num_clients': 10,
        'num_rounds': 300,
        'batch_size': 32,
        'local_epochs': 1,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'V_async': 14.0,     # For your semi-async algorithm
        'V_sync': 10.0,      # For sync baseline
        'sigma_n': 0.04,     # Noise std
        'tau_cm': 0.01,      # Communication latency
        'T_max': 500,        # Time budget
        'gamma0': 10.0,      # Target SNR for sync
        'l': 0.1,            # Smoothness constant
        'G': 5.0             # Gradient bound
    }
    
    print(f"Using device: {config['device']}")
    print("=== Running Semi-Async OTA Experiment ===")
    async_metrics, async_model = run_semi_async_experiment(config)
    
    print("\n=== Running Sync OTA Baseline Experiment ===")
    sync_metrics, sync_model = run_sync_ota_experiment(config)
    
    # Print final results
    print("\n=== Final Results ===")
    print(f"Semi-Async Final Accuracy: {async_metrics['accuracies'][-1]:.2f}%")
    print(f"Sync OTA Final Accuracy: {sync_metrics['accuracies'][-1]:.2f}%")
    print(f"Semi-Async Avg Round Duration: {np.mean(async_metrics['round_durations']):.4f}s")
    print(f"Sync OTA Avg Round Duration: {np.mean(sync_metrics['round_durations']):.4f}s")
    
    # Plot comparison
    plot_comparison_results(async_metrics, sync_metrics, config)
    
    # Save models and metrics
    torch.save(async_model.state_dict(), "semi_async_model.pth")
    torch.save(sync_model.state_dict(), "sync_ota_model.pth")
    np.savez("comparison_metrics.npz", 
             async_metrics=async_metrics, 
             sync_metrics=sync_metrics,
             config=config)

if __name__ == "__main__":
    main()