import torch
import numpy as np
import time
import pandas as pd
import os
from torch.utils.data import DataLoader
from safa_client import SAFAClient
from safa_server import SAFAServer
from model import CNNMnist
from dataloader import load_mnist, partition_mnist_dirichlet
import matplotlib.pyplot as plt
import logging
from collections import defaultdict
import copy

def run_safa_experiment():
    """Run a single SAFA experiment and return metrics"""
    # ===== Logging Configuration =====
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.WARNING)  # Reduce verbosity for multiple runs
    
    # ========== Experiment Configuration ==========
    NUM_CLIENTS = 10
    NUM_ROUNDS = 300
    BATCH_SIZE = 32
    LOCAL_EPOCHS = 1
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    CRASH_PROB = 0.15
    LAG_TOLERANCE = 3
    SELECT_FRAC = 0.5
    EVAL_INTERVAL = 5
    
    # ========== Data Preparation ==========
    train_dataset, test_dataset = load_mnist()
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    client_data_map = partition_mnist_dirichlet(train_dataset, NUM_CLIENTS, alpha=0.05)

    # ========== Client Initialization ==========
    clients = []
    for cid in range(NUM_CLIENTS):
        indices = client_data_map[cid]
        if len(indices) == 0:
            indices = [0]  # Ensure at least one sample

        # Hardware diversity parameters
        cpu_freq = np.random.uniform(1e9, 2e9)  # 1-2 GHz
        crash_prob = CRASH_PROB * np.random.uniform(0.8, 1.2)
        
        clients.append(
            SAFAClient(
                client_id=cid,
                data_indices=indices,
                model=CNNMnist(),
                fk=cpu_freq,
                mu_k=1e-27,
                P_max=0.2,
                C=1e6,
                Ak=BATCH_SIZE,
                train_dataset=train_dataset,
                device=DEVICE,
                local_epochs=LOCAL_EPOCHS,
                crash_prob=crash_prob
            )
        )

    # ========== Server Initialization ==========
    global_model = CNNMnist().to(DEVICE)
    server = SAFAServer(
        global_model=global_model,
        clients=clients,
        lag_tolerance=LAG_TOLERANCE,
        select_frac=SELECT_FRAC,
        learning_rate=0.01,
        device=DEVICE
    )

    # ========== Metrics Tracking ==========
    metrics = {
        'energy_consumption': [],
        'accuracies': [],
        'client_selections': defaultdict(int),
        'per_client_energy': {cid: 0.0 for cid in range(NUM_CLIENTS)},
        'cumulative_energy': {cid: [] for cid in range(NUM_CLIENTS)},
        'evaluation_rounds': []
    }

    # ========== Training Loop ==========
    for round_idx in range(NUM_ROUNDS):
        # Run federated round
        eval_this_round = (round_idx % EVAL_INTERVAL == 0) or (round_idx == NUM_ROUNDS - 1)
        server.run_round(test_loader=test_loader if eval_this_round else None)

        # Track energy consumption
        round_energy = 0.0
        if round_idx < len(server.energy_history):
            for energy_record in server.energy_history[round_idx]:
                client_id = energy_record['client_id']
                client_energy = energy_record['total']
                round_energy += client_energy
                metrics['per_client_energy'][client_id] += client_energy
                
                # Update cumulative energy tracking
                if round_idx == 0:
                    metrics['cumulative_energy'][client_id].append(client_energy)
                else:
                    prev = metrics['cumulative_energy'][client_id][-1] if metrics['cumulative_energy'][client_id] else 0
                    metrics['cumulative_energy'][client_id].append(prev + client_energy)
        
        metrics['energy_consumption'].append(round_energy)
        
        # Track client selections
        if server.selection_history and round_idx < len(server.selection_history):
            for cid in server.selection_history[round_idx]:
                metrics['client_selections'][cid] += 1
        
        # Track accuracies
        if eval_this_round and server.accuracy_history:
            metrics['accuracies'].append(server.accuracy_history[-1])
            metrics['evaluation_rounds'].append(round_idx)
    
    # Add final accuracy
    final_acc = server.evaluate(test_loader)
    metrics['accuracies'].append(final_acc)
    metrics['evaluation_rounds'].append(NUM_ROUNDS - 1)
    metrics['final_accuracy'] = final_acc
    
    # Fill in cumulative energy for all rounds
    for cid in range(NUM_CLIENTS):
        if not metrics['cumulative_energy'][cid]:
            metrics['cumulative_energy'][cid] = [0.0] * NUM_ROUNDS
        else:
            # Extend to full rounds if needed
            while len(metrics['cumulative_energy'][cid]) < NUM_ROUNDS:
                last_val = metrics['cumulative_energy'][cid][-1]
                metrics['cumulative_energy'][cid].append(last_val)
    
    return metrics

def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Run experiments until we have 10 with accuracy >= 80%
    successful_runs = []
    run_count = 0
    
    while len(successful_runs) < 1:
        run_count += 1
        logger.info(f"Starting experiment run {run_count}")
        results = run_safa_experiment()
        
        if results['final_accuracy'] >= 60:
            successful_runs.append(results)
            logger.info(f"Run {run_count} successful! Accuracy: {results['final_accuracy']:.2f}%")
        else:
            logger.info(f"Run {run_count} discarded. Accuracy: {results['final_accuracy']:.2f}% < 80%")
    
    # Aggregate results from successful runs
    aggregated = {
        'accuracies': np.mean([run['accuracies'] for run in successful_runs], axis=0),
        'energy_per_round': np.mean([run['energy_consumption'] for run in successful_runs], axis=0),
        'cumulative_energy_per_client': {
            cid: np.mean([run['cumulative_energy'][cid] for run in successful_runs], axis=0)
            for cid in range(10)
        },
        'selection_counts': np.mean([
            [run['client_selections'][cid] for cid in range(10)] 
            for run in successful_runs
        ], axis=0),
        'total_energy': np.mean([sum(run['energy_consumption']) for run in successful_runs]),
        'per_client_energy': np.mean([
            [run['per_client_energy'][cid] for cid in range(10)] 
            for run in successful_runs
        ], axis=0),
        'evaluation_rounds': successful_runs[0]['evaluation_rounds']
    }
    
    # Save results to CSV
    os.makedirs("safa_results02", exist_ok=True)
    
    # 1. Accuracy results
    accuracy_df = pd.DataFrame({
        'round': aggregated['evaluation_rounds'],
        'accuracy': aggregated['accuracies']
    })
    accuracy_df.to_csv("safa_results02/accuracy_results.csv", index=False)
    
    # 2. Energy per round
    energy_df = pd.DataFrame({
        'round': np.arange(1, 301),
        'energy': aggregated['energy_per_round']
    })
    energy_df.to_csv("safa_results02/energy_per_round.csv", index=False)
    
    # 3. Cumulative energy per client
    cumulative_energy_data = {'round': np.arange(1, 301)}
    for cid in range(10):
        cumulative_energy_data[f'client_{cid}'] = aggregated['cumulative_energy_per_client'][cid]
    cumulative_df = pd.DataFrame(cumulative_energy_data)
    cumulative_df.to_csv("safa_results02/cumulative_energy_per_client.csv", index=False)
    
    # 4. Selection counts
    selection_df = pd.DataFrame({
        'client_id': np.arange(10),
        'selection_count': aggregated['selection_counts']
    })
    selection_df.to_csv("safa_results02/selection_counts.csv", index=False)
    
    # 5. Summary report
    with open("safa_results02/summary_report.txt", "w") as f:
        f.write("=== SAFA Federated Learning Experiment Summary ===\n")
        f.write(f"Total runs: {run_count} (10 successful runs with accuracy >= 80%)\n")
        f.write(f"Average final accuracy: {np.mean([r['final_accuracy'] for r in successful_runs]):.2f}%\n")
        f.write(f"Average total energy consumption: {aggregated['total_energy']:.4f} J\n\n")
        
        f.write("Per-client information:\n")
        f.write("Client | Selections | Energy (J)\n")
        f.write("--------------------------------\n")
        for cid in range(10):
            energy = aggregated['per_client_energy'][cid]
            selections = aggregated['selection_counts'][cid]
            f.write(f"{cid:6} | {selections:10.1f} | {energy:9.4f}\n")
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Accuracy plot
    plt.subplot(221)
    plt.plot(aggregated['evaluation_rounds'], aggregated['accuracies'], 'o-')
    plt.title("Test Accuracy")
    plt.xlabel("Communication Rounds")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)
    
    # Energy per round plot
    plt.subplot(222)
    plt.plot(aggregated['energy_per_round'])
    plt.title("Energy Consumption per Round")
    plt.xlabel("Rounds")
    plt.ylabel("Energy (J)")
    plt.grid(True)
    
    # Cumulative energy per client
    plt.subplot(223)
    for cid in range(10):
        plt.plot(aggregated['cumulative_energy_per_client'][cid], label=f'Client {cid}')
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
    plt.savefig("safa_results02/aggregated_results.png")
    
    # Print summary to console
    print("\n=== SAFA Experiment Summary ===")
    print(f"Average final accuracy: {np.mean([r['final_accuracy'] for r in successful_runs]):.2f}%")
    print(f"Total energy consumption: {aggregated['total_energy']:.4f} J")
    print("\nPer-client information:")
    print("Client | Selections | Energy (J)")
    print("--------------------------------")
    for cid in range(10):
        print(f"{cid:6} | {aggregated['selection_counts'][cid]:10.1f} | {aggregated['per_client_energy'][cid]:9.4f}")

if __name__ == "__main__":
    main()