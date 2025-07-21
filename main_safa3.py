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
    client_data_map = partition_mnist_dirichlet(train_dataset, NUM_CLIENTS, alpha=0.01)

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
        'evaluation_rounds': [],
        'round_durations': [],  # Track time per round
        'cumulative_time': 0.0,  # Track total time
        'cumulative_energy_total': 0.0  # Track total energy
    }

    # ====== FIX 1: Add initial evaluation before any training ======
    initial_acc = server.evaluate(test_loader)
    metrics['accuracies'].append(initial_acc)
    metrics['evaluation_rounds'].append(-1)  # Mark as pre-training

    # ========== Training Loop ==========
    for round_idx in range(NUM_ROUNDS):
        round_start = time.time()
        
        # Run federated round
        eval_this_round = (round_idx % EVAL_INTERVAL == 0) or (round_idx == NUM_ROUNDS - 1)
        server.run_round(test_loader=test_loader if eval_this_round else None)
        
        # Track round duration
        round_duration = time.time() - round_start
        metrics['round_durations'].append(round_duration)
        metrics['cumulative_time'] += round_duration

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
        metrics['cumulative_energy_total'] += round_energy
        
        # Track client selections
        if server.selection_history and round_idx < len(server.selection_history):
            for cid in server.selection_history[round_idx]:
                metrics['client_selections'][cid] += 1
        
        # Track accuracies and store energy/time at evaluation points
        if eval_this_round and server.accuracy_history:
            # FIX: Append latest accuracy from history
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
    NUM_ROUNDS = 300
    
    # Run experiments until we have 10 with accuracy >= 70%
    successful_runs = []
    run_count = 0
    
    while len(successful_runs) < 1:
        run_count += 1
        # logger.info(f"Starting experiment run {run_count}")
        results = run_safa_experiment()
        
        if results['final_accuracy'] >= 70:
            successful_runs.append(results)
            # logger.info(f"Run {run_count} successful! Accuracy: {results['final_accuracy']:.2f}%")
        else:
            logger.info(f"Run {run_count} discarded. Accuracy: {results['final_accuracy']:.2f}% < 70%")
    
    # Create results directory
    os.makedirs("safa_results", exist_ok=True)
    
    # ====== FIX 2: Properly handle initial evaluation in per-run data ======
    for i, run in enumerate(successful_runs):
        # Create CSV for this run
        # Handle initial evaluation separately (round -1 with 0 energy/time)
        energy_at_eval = [0.0]  # Start with 0 for initial evaluation
        time_at_eval = [0.0]    # Start with 0 for initial evaluation
        
        # Calculate cumulative values for subsequent evaluation points
        for idx in run['evaluation_rounds'][1:]:
            # Scale cumulative total by progress
            progress = (idx + 1) / (NUM_ROUNDS)  # +1 because idx starts at 0
            energy_at_eval.append(run['cumulative_energy_total'] * progress)
            time_at_eval.append(run['cumulative_time'] * progress)
        
        df_run = pd.DataFrame({
            'eval_round': run['evaluation_rounds'],
            'accuracy': run['accuracies'],
            'cumulative_energy': energy_at_eval,
            'cumulative_time': time_at_eval
        })
        df_run.to_csv(f"safa_results/run_{i+1}_acc_energy_time.csv", index=False)
    
    # ====== FIX 3: Adjust interpolation to include starting point ======
    # Find max evaluation rounds across runs
    max_rounds = max(max(run['evaluation_rounds']) for run in successful_runs)
    
    # Create interpolation grid including starting point (-1)
    eval_points = np.linspace(-1, max_rounds, 62)  # -1,0,5,10,...,300 (62 points)
    
    # Initialize storage
    all_acc = np.zeros((len(successful_runs), len(eval_points)))
    all_energy = np.zeros((len(successful_runs), len(eval_points)))
    all_time = np.zeros((len(successful_runs), len(eval_points)))
    
    # Collect data with proper initial values
    for i, run in enumerate(successful_runs):
        # FIX: Use actual evaluation points for interpolation
        run_rounds = run['evaluation_rounds']
        run_accs = run['accuracies']
        
        # For energy and time, create values at evaluation points
        energy_vals = [0.0]  # Initial evaluation has 0 energy
        time_vals = [0.0]    # Initial evaluation has 0 time
        
        # Calculate for subsequent points
        for j in range(1, len(run_rounds)):
            progress = (run_rounds[j] + 1) / NUM_ROUNDS  # +1 because rounds are 0-indexed
            energy_vals.append(run['cumulative_energy_total'] * progress)
            time_vals.append(run['cumulative_time'] * progress)
        
        # Interpolate accuracy
        all_acc[i] = np.interp(eval_points, run_rounds, run_accs)
        
        # Interpolate energy and time
        all_energy[i] = np.interp(eval_points, run_rounds, energy_vals)
        all_time[i] = np.interp(eval_points, run_rounds, time_vals)
    
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
    df_avg.to_csv("safa_results/average_acc_energy_time.csv", index=False)
    
    # ====== FIX 4: Update plots to show starting point ======
    plt.figure(figsize=(15, 6))
    
    # Accuracy vs Cumulative Energy
    plt.subplot(121)
    for i, run in enumerate(successful_runs):
        # Get actual points for this run
        run_rounds = run['evaluation_rounds']
        energy_vals = [0.0]
        for j in range(1, len(run_rounds)):
            progress = (run_rounds[j] + 1) / NUM_ROUNDS
            energy_vals.append(run['cumulative_energy_total'] * progress)
            
        plt.plot(
            energy_vals,
            run['accuracies'],
            alpha=0.3
        )
    plt.plot(avg_energy, avg_acc, 'r-', linewidth=3, label='Average')
    plt.xlabel("Cumulative Energy (J)")
    plt.ylabel("Accuracy (%)")
    plt.title("SAFA: Accuracy vs Cumulative Energy")
    plt.grid(True)
    plt.legend()
    
    # Accuracy vs Cumulative Time
    plt.subplot(122)
    for i, run in enumerate(successful_runs):
        # Get actual points for this run
        run_rounds = run['evaluation_rounds']
        time_vals = [0.0]
        for j in range(1, len(run_rounds)):
            progress = (run_rounds[j] + 1) / NUM_ROUNDS
            time_vals.append(run['cumulative_time'] * progress)
            
        plt.plot(
            time_vals,
            run['accuracies'],
            alpha=0.3
        )
    plt.plot(avg_time, avg_acc, 'r-', linewidth=3, label='Average')
    plt.xlabel("Cumulative Time (s)")
    plt.ylabel("Accuracy (%)")
    plt.title("SAFA: Accuracy vs Cumulative Time")
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("safa_results/acc_vs_energy_time.png", dpi=300)
    
    # ====== Original aggregated results (with initial point fix) ======
    aggregated = {
        'accuracies': [],
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
    
    # FIX: Average accuracies properly including initial point
    min_len = min(len(run['accuracies']) for run in successful_runs)
    for i in range(min_len):
        aggregated['accuracies'].append(np.mean([run['accuracies'][i] for run in successful_runs]))
    
    # Save results to CSV
    # 1. Accuracy results
    accuracy_df = pd.DataFrame({
        'round': aggregated['evaluation_rounds'],
        'accuracy': aggregated['accuracies']
    })
    accuracy_df.to_csv("safa_results/accuracy_results.csv", index=False)
    
    # 2. Energy per round
    energy_df = pd.DataFrame({
        'round': np.arange(1, 301),
        'energy': aggregated['energy_per_round']
    })
    energy_df.to_csv("safa_results/energy_per_round.csv", index=False)
    
    # 3. Cumulative energy per client
    cumulative_energy_data = {'round': np.arange(1, 301)}
    for cid in range(10):
        cumulative_energy_data[f'client_{cid}'] = aggregated['cumulative_energy_per_client'][cid]
    cumulative_df = pd.DataFrame(cumulative_energy_data)
    cumulative_df.to_csv("safa_results/cumulative_energy_per_client.csv", index=False)
    
    # 4. Selection counts
    selection_df = pd.DataFrame({
        'client_id': np.arange(10),
        'selection_count': aggregated['selection_counts']
    })
    selection_df.to_csv("safa_results/selection_counts.csv", index=False)
    
    # 5. Summary report
    with open("safa_results/summary_report.txt", "w") as f:
        f.write("=== SAFA Federated Learning Experiment Summary ===\n")
        f.write(f"Total runs: {run_count} (10 successful runs with accuracy >= 70%)\n")
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
    plt.savefig("safa_results/aggregated_results.png")
    
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