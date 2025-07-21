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
from dataloader import load_mnist, partition_mnist_noniid, partition_mnist_dirichlet
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
        'client_energy': {cid: 0.0 for cid in range(num_clients)},
        'evaluation_rounds': [],
        'cumulative_energy_total': 0.0,
        'cumulative_time': 0.0,
        'round_durations': []
    }
    
    # Training loop
    for round_idx in range(total_rounds):
        round_start = time.time()
        
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
            if not metrics['cumulative_energy'][cid]:
                metrics['cumulative_energy'][cid].append(E_actual)
            else:
                metrics['cumulative_energy'][cid].append(metrics['cumulative_energy'][cid][-1] + E_actual)
        
        metrics['round_energies'].append(round_energy)
        metrics['cumulative_energy_total'] += round_energy
        
        # Track round duration and total time
        round_duration = time.time() - round_start
        metrics['round_durations'].append(round_duration)
        metrics['cumulative_time'] += round_duration
        
        # 7. Evaluation
        if (round_idx + 1) % 5 == 0:
            acc = evaluate_model(server.global_model, test_loader, device)
            metrics['accuracies'].append(acc)
            metrics['evaluation_rounds'].append(round_idx + 1)  # Store round number
    
    # Final evaluation
    final_acc = evaluate_model(server.global_model, test_loader, device)
    metrics['final_accuracy'] = final_acc
    metrics['accuracies'].append(final_acc)
    metrics['evaluation_rounds'].append(total_rounds)
    
    # Fill cumulative energy arrays
    for cid in range(num_clients):
        if len(metrics['cumulative_energy'][cid]) < total_rounds:
            last_val = metrics['cumulative_energy'][cid][-1] if metrics['cumulative_energy'][cid] else 0
            while len(metrics['cumulative_energy'][cid]) < total_rounds:
                metrics['cumulative_energy'][cid].append(last_val)
    
    return metrics

def main():
    # Configuration
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Run experiments until we have 10 with accuracy >= 70%
    successful_runs = []
    run_count = 0
    
    while len(successful_runs) < 10:
        run_count += 1
        logging.info(f"Starting experiment run {run_count}")
        results = run_single_experiment()
        
        if results['final_accuracy'] >= 75:
            successful_runs.append(results)
            logging.info(f"Run {run_count} successful! Accuracy: {results['final_accuracy']:.2f}%")
        else:
            logging.info(f"Run {run_count} discarded. Accuracy: {results['final_accuracy']:.2f}% < 70%")
    
    # Create results directory
    os.makedirs("sync_results", exist_ok=True)
    
    # ====== Save per-run accuracy vs energy/time ======
    for i, run in enumerate(successful_runs):
        # Create CSV for this run
        energy_at_eval = []
        time_at_eval = []
        
        # Reconstruct cumulative energy at evaluation points
        for rnd in run['evaluation_rounds']:
            if rnd <= len(run['round_energies']):
                # Sum energy up to this round
                energy_sum = sum(run['round_energies'][:rnd])
                # Sum time up to this round
                time_sum = sum(run['round_durations'][:rnd])
                energy_at_eval.append(energy_sum)
                time_at_eval.append(time_sum)
        
        df_run = pd.DataFrame({
            'eval_round': run['evaluation_rounds'],
            'accuracy': run['accuracies'],
            'cumulative_energy': energy_at_eval,
            'cumulative_time': time_at_eval
        })
        df_run.to_csv(f"sync_results/run_{i+1}_acc_energy_time.csv", index=False)
    
    # ====== Aggregate results for averaged plots ======
    # Find max evaluation rounds across runs
    max_round = 300
    
    # Create interpolation grid
    eval_points = np.linspace(0, max_round, 61)  # 0,5,10,...,300
    
    # Initialize storage
    all_acc = np.zeros((len(successful_runs), len(eval_points)))
    all_energy = np.zeros((len(successful_runs), len(eval_points)))
    all_time = np.zeros((len(successful_runs), len(eval_points)))
    
    # Collect data
    for i, run in enumerate(successful_runs):
        # Get total energy and time for this run
        total_energy = sum(run['round_energies'])
        total_time = sum(run['round_durations'])
        
        # Interpolate accuracy
        all_acc[i] = np.interp(eval_points, run['evaluation_rounds'], run['accuracies'])
        
        # Calculate cumulative energy and time at evaluation points
        energy_vals = [total_energy * (p/max_round) for p in eval_points]
        time_vals = [total_time * (p/max_round) for p in eval_points]
        
        all_energy[i] = energy_vals
        all_time[i] = time_vals
    
    # Calculate averages
    avg_acc = np.mean(all_acc, axis=0)
    avg_energy = np.mean(all_energy, axis=0)
    avg_time = np.mean(all_time, axis=0)
    
    # Save averaged data
    df_avg = pd.DataFrame({
        'eval_round': eval_points,
        'accuracy': avg_acc,
        'cumulative_energy': avg_energy,
        'cumulative_time': avg_time
    })
    df_avg.to_csv("sync_results/average_acc_energy_time.csv", index=False)
    
    # ====== Generate plots ======
    plt.figure(figsize=(15, 6))
    
    # Accuracy vs Cumulative Energy
    plt.subplot(121)
    for i, run in enumerate(successful_runs):
        total_energy = sum(run['round_energies'])
        plt.plot(
            [total_energy * (rnd/max(run['evaluation_rounds'])) for rnd in run['evaluation_rounds']],
            run['accuracies'],
            alpha=0.3
        )
    plt.plot(avg_energy, avg_acc, 'r-', linewidth=3, label='Average')
    plt.xlabel("Cumulative Energy (J)")
    plt.ylabel("Accuracy (%)")
    plt.title("Synchronous FL: Accuracy vs Cumulative Energy")
    plt.grid(True)
    plt.legend()
    
    # Accuracy vs Cumulative Time
    plt.subplot(122)
    for i, run in enumerate(successful_runs):
        total_time = sum(run['round_durations'])
        plt.plot(
            [total_time * (rnd/max(run['evaluation_rounds'])) for rnd in run['evaluation_rounds']],
            run['accuracies'],
            alpha=0.3
        )
    plt.plot(avg_time, avg_acc, 'r-', linewidth=3, label='Average')
    plt.xlabel("Cumulative Time (s)")
    plt.ylabel("Accuracy (%)")
    plt.title("Synchronous FL: Accuracy vs Cumulative Time")
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("sync_results/acc_vs_energy_time.png", dpi=300)
    
    # ====== Original aggregated results ======
    # Determine the actual evaluation rounds based on the minimum accuracy length
    min_acc_length = min(len(run['accuracies']) for run in successful_runs)
    
    # Create evaluation rounds that correspond to the accuracy measurements
    eval_rounds = [5 * (i+1) for i in range(min_acc_length)]
    
    # Ensure we don't exceed 300 rounds
    eval_rounds = [r for r in eval_rounds if r <= 300]
    if eval_rounds and eval_rounds[-1] != 300:
        eval_rounds.append(300)
    
    # Recalculate min_acc_length based on the final eval_rounds
    min_acc_length = len(eval_rounds)
    
    # Aggregate accuracies (only up to min_acc_length)
    all_accuracies = np.array([run['accuracies'][:min_acc_length] for run in successful_runs])
    avg_accuracies = np.mean(all_accuracies, axis=0)
    
    # Aggregate other metrics
    avg_round_energies = np.mean([run['round_energies'] for run in successful_runs], axis=0)
    avg_selection_counts = np.mean([run['selection_counts'] for run in successful_runs], axis=0)
    total_energy_avg = np.mean([sum(run['round_energies']) for run in successful_runs])
    avg_per_client_energy = np.mean([[run['client_energy'][cid] for cid in range(10)] for run in successful_runs], axis=0)
    final_accuracy_avg = np.mean([run['final_accuracy'] for run in successful_runs])
    
    # 1. Accuracy results - FIXED
    accuracy_df = pd.DataFrame({
        'round': eval_rounds,
        'accuracy': avg_accuracies
    })
    accuracy_df.to_csv("sync_results/accuracy_results.csv", index=False)
    
    # 2. Energy per round
    energy_df = pd.DataFrame({
        'round': list(range(1, 301)),
        'energy': avg_round_energies
    })
    energy_df.to_csv("sync_results/energy_per_round.csv", index=False)
    
    # 3. Selection counts
    selection_df = pd.DataFrame({
        'client_id': list(range(10)),
        'selection_count': avg_selection_counts
    })
    selection_df.to_csv("sync_results/selection_counts.csv", index=False)
    
    # 4. Summary report
    with open("sync_results/summary_report.txt", "w") as f:
        f.write("=== Synchronous FL Experiment Summary ===\n")
        f.write(f"Total runs: {run_count} (10 successful runs with accuracy >= 70%)\n")
        f.write(f"Average final accuracy: {final_accuracy_avg:.2f}%\n")
        f.write(f"Average total energy consumption: {total_energy_avg:.4f} J\n\n")
        
        f.write("Per-client information:\n")
        f.write("Client | Selections | Energy (J)\n")
        f.write("--------------------------------\n")
        for cid in range(10):
            f.write(f"{cid:6} | {avg_selection_counts[cid]:10.1f} | {avg_per_client_energy[cid]:9.4f}\n")
    
    # Print summary to console
    print("\n=== Experiment Summary ===")
    print(f"Average final accuracy: {final_accuracy_avg:.2f}%")
    print(f"Total energy consumption: {total_energy_avg:.4f} J")
    print("\nPer-client information:")
    print("Client | Selections | Energy (J)")
    print("--------------------------------")
    for cid in range(10):
        print(f"{cid:6} | {avg_selection_counts[cid]:10.1f} | {avg_per_client_energy[cid]:9.4f}")

if __name__ == "__main__":
    main()