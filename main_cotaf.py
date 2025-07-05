import torch
import numpy as np
import time
import copy
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from model import CNNMnist
from server_cotaf import COTAFServer
from client_cotaf import COTAFClient
from dataloader import load_mnist, partition_mnist_noniid

def evaluate_model(model, test_loader, device='cpu'):
    # Create new loader with proper device settings
    test_loader = DataLoader(test_loader.dataset, 
                            batch_size=test_loader.batch_size,
                            shuffle=False,
                            pin_memory=(device=='cuda'))
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

def run_cotaf_experiment(clients, NUM_ROUNDS, BATCH_SIZE, DEVICE):
    """Run COTAF experiment with identical setup"""
    # Initialize global model
    train_data, test_data = load_mnist()
    global_model = CNNMnist().to(DEVICE)
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False)
    
    # Initialize COTAF server
    server = COTAFServer(
        global_model=global_model,
        clients=clients,
        P_max=1.0,
        noise_var=0.01,
        H_local=1,
        device=DEVICE
    )

    energy_metrics = {
        'per_round_total': [],
        'cumulative_per_client': {c.client_id: 0.0 for c in clients},
        'per_round_per_client': []
    }

    # Training metrics
    accuracies = []
    evaluation_rounds = []  # Track actual round numbers of evaluations
    round_durations = []
    time_points = []
    cumulative_time = 0
    
    print("\n=== Starting COTAF Experiment ===")
    
    for round_idx in range(NUM_ROUNDS):
        round_start = time.time()
        print(f"\n=== Round {round_idx+1}/{NUM_ROUNDS} ===")
        
        # 1. Broadcast global model to ALL clients
        server.broadcast_model()
        
        # 2. Local training (H steps)
        comp_times = []
        for client in clients:
            client_start = time.time()
            client.local_train()
            comp_times.append(time.time() - client_start)
        
        # 3. COTAF aggregation (all clients participate)
        new_state = server.aggregate()
        server.update_model(new_state)
        
        # 4. Record metrics
        round_time = max(comp_times)  # Synchronous = slowest client
        cumulative_time += round_time
        round_durations.append(round_time)

        # Energy tracking
        if server.energy_tracker['per_round_total']:  # Check if not empty
            energy_metrics['per_round_total'].append(
                server.energy_tracker['per_round_total'][-1]
            )
            energy_metrics['per_round_per_client'].append(
                server.energy_tracker['per_client_per_round'][-1]
            )
            for cid, energy in server.energy_tracker['per_client_per_round'][-1].items():
                energy_metrics['cumulative_per_client'][cid] += energy
        
        # 5. Evaluate periodically
        if (round_idx + 1) % 5 == 0 or round_idx == 0:
            acc = evaluate_model(server.global_model, test_loader, DEVICE)
            accuracies.append(acc)
            evaluation_rounds.append(round_idx+1)
            time_points.append(cumulative_time)
            print(f"Global model accuracy: {acc:.2f}%")
        
        print(f"Round duration: {round_time:.4f}s | "
              f"Cumulative time: {cumulative_time:.2f}s")
    
    # Final evaluation (if not already done)
    if NUM_ROUNDS % 5 != 0:
        final_acc = evaluate_model(server.global_model, test_loader, DEVICE)
        accuracies.append(final_acc)
        evaluation_rounds.append(NUM_ROUNDS)
        time_points.append(cumulative_time)
    else:
        final_acc = accuracies[-1]
    
    return {
        'accuracies': accuracies,
        'evaluation_rounds': evaluation_rounds,
        'final_acc': final_acc,
        'round_durations': round_durations,
        'time_points': time_points,
        'total_time': cumulative_time,
        'energy_metrics': energy_metrics,
        'total_energy': sum(energy_metrics['per_round_total'])
    }

def plot_cotaf_results(results):
    """Plot COTAF experiment results"""
    plt.figure(figsize=(15, 10))
    
    # Accuracy vs Rounds
    plt.subplot(2, 2, 1)
    plt.plot(results['evaluation_rounds'], results['accuracies'], 'o-')
    plt.title("COTAF: Accuracy vs Communication Rounds")
    plt.xlabel("Communication Rounds")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)
    
    # Accuracy vs Time
    plt.subplot(2, 2, 2)
    plt.plot(results['time_points'], results['accuracies'], 's-')
    plt.title("COTAF: Accuracy vs Wall-clock Time")
    plt.xlabel("Time Elapsed (seconds)")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)
    
    # Round Duration
    plt.subplot(2, 2, 3)
    plt.plot(results['round_durations'], 'o-', markersize=3)
    plt.title("Round Duration Over Time")
    plt.xlabel("Round Index")
    plt.ylabel("Duration (seconds)")
    plt.grid(True)
    
    # Cumulative Time
    plt.subplot(2, 2, 4)
    plt.plot(np.cumsum(results['round_durations']), 's-', markersize=3)
    plt.title("Cumulative Training Time")
    plt.xlabel("Round Index")
    plt.ylabel("Total Time (seconds)")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("cotaf_results.png", dpi=150)
    plt.show()

def plot_energy_results(results):
    """Plot energy consumption results for COTAF"""
    energy = results['energy_metrics']
    
    plt.figure(figsize=(15, 5))
    
    # Total energy per round
    plt.subplot(1, 3, 1)
    plt.plot(energy['per_round_total'], 'o-')
    plt.title("Total System Energy per Round")
    plt.xlabel("Round Index")
    plt.ylabel("Energy (Joules)")
    plt.grid(True)
    
    # Cumulative energy
    plt.subplot(1, 3, 2)
    plt.plot(np.cumsum(energy['per_round_total']), 's-')
    plt.title("Cumulative System Energy")
    plt.xlabel("Round Index")
    plt.ylabel("Total Energy (Joules)")
    plt.grid(True)
    
    # Per-client energy distribution
    plt.subplot(1, 3, 3)
    client_ids = list(energy['cumulative_per_client'].keys())
    cumulative_energy = [energy['cumulative_per_client'][cid] for cid in client_ids]
    plt.bar(client_ids, cumulative_energy)
    plt.title("Cumulative Energy per Client")
    plt.xlabel("Client ID")
    plt.ylabel("Total Energy (Joules)")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("cotaf_energy_results.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    # Parameters
    NUM_CLIENTS = 10
    NUM_ROUNDS = 100
    BATCH_SIZE = 32
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {DEVICE}")
    
    # Load and partition data
    train_dataset, test_data = load_mnist()
    client_data_map = partition_mnist_noniid(train_dataset, NUM_CLIENTS)
    
    # Create clients
    cotaf_clients = [
        COTAFClient(
            client_id=cid,
            data_indices=client_data_map[cid],
            model=CNNMnist(),
            fk=np.random.uniform(1e9, 2e9),
            mu_k=1e-27,
            P_max=1.0 + np.random.rand(),
            C=1e6,
            Ak=BATCH_SIZE,
            train_dataset=train_dataset,
            device=DEVICE,
            local_epochs=1
        )
        for cid in range(NUM_CLIENTS)
    ]
    
    # Initialize optimizers
    for client in cotaf_clients:
        client.optimizer = torch.optim.SGD(client.local_model.parameters(), lr=0.01)
    
    # Run COTAF experiment
    results = run_cotaf_experiment(
        clients=cotaf_clients,
        NUM_ROUNDS=NUM_ROUNDS,
        BATCH_SIZE=BATCH_SIZE,
        DEVICE=DEVICE
    )
    
    # Plot results
    plot_cotaf_results(results)
    plot_energy_results(results)
    
    # Print final results
    print("\n=== COTAF Final Results ===")
    print(f"Final accuracy: {results['final_acc']:.2f}%")
    print(f"Total training time: {results['total_time']:.2f} seconds")
    print(f"Average round duration: {np.mean(results['round_durations']):.4f} seconds")
    print(f"Total rounds: {NUM_ROUNDS}")

    print("\nEnergy Consumption Summary:")
    print(f"Total system energy: {results['total_energy']:.2f} J")
    print(f"Average per-round energy: {np.mean(results['energy_metrics']['per_round_total']):.2f} J")
    
    print("\nPer-client cumulative energy:")
    for cid, energy in results['energy_metrics']['cumulative_per_client'].items():
        print(f"Client {cid}: {energy:.2f} J")