import os
import time
import logging
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import torch
import copy

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

from torch.utils.data import DataLoader
from dataloader import load_mnist, partition_mnist_noniid
from model import CNNMnist
from client import Client
from server import Server
from client_sync import SyncClient
from server_sync import SyncServer
from server_cotaf import COTAFServer
from main_cotaf import run_cotaf_experiment
from client_cotaf import COTAFClient
from safa_client import SAFAClient
from safa_server import SAFAServer

def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += len(data)
    accuracy = 100. * correct / total
    return accuracy

def jains_fairness(counts):
    numerator = sum(counts) ** 2
    denominator = len(counts) * sum([x ** 2 for x in counts])
    return numerator / denominator if denominator != 0 else 0

# ====================== ALGORITHM RUNNERS ======================
def run_semi_async_ota(num_clients=10, num_rounds=300, device='cpu'):
    # Setup logging
    logger = logging.getLogger("SemiAsync")
    logger.setLevel(logging.INFO)
    
    # Load data
    train_dataset, test_dataset = load_mnist()
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    client_data_map = partition_mnist_noniid(train_dataset, num_clients)
    
    # Initialize clients
    clients = []
    E_max_dict = {}
    for cid in range(num_clients):
        indices = client_data_map[cid]
        if len(indices) == 0:
            indices = [0]
            logger.warning(f"Client {cid} has no data! Adding dummy sample")
        
        client = Client(  # Using your Client class
            client_id=cid,
            data_indices=indices,
            model=CNNMnist(),
            fk=np.random.uniform(1e9, 2e9),
            mu_k=1e-27,
            P_max=2.0 + np.random.rand(),
            C=1e6,
            Ak=32,
            train_dataset=train_dataset,
            device=device,
            local_epochs=1
        )
        clients.append(client)
        E_max_dict[cid] = np.random.uniform(25, 38)
    
    # Initialize server
    global_model = CNNMnist().to(device)
    server = Server(  # Using your Server class
        global_model=global_model,
        clients=clients,
        V=14.0,
        sigma_n=0.04,
        tau_cm=0.01,
        T_max=500,
        E_max=E_max_dict,
        T_total_rounds=num_rounds,
        device=device
    )
    
    # Training metrics
    metrics = {
        'accuracies': [],
        'round_durations': [],
        'energy_per_round': [],
        'selection_counts': {cid: 0 for cid in range(num_clients)},
        'cumulative_energy_per_client': {cid: 0.0 for cid in range(num_clients)},
        'time_points': [],
        'time_based_accuracies': [],
        'start_time': time.time()
    }
    
    # Training loop (using your original main loop)
    accuracies = []
    round_durations = []
    energy_queues = []
    avg_staleness_per_round = []
    selected_counts = []
    client_selection_counts = {cid: 0 for cid in range(num_clients)}

    for round_idx in range(num_rounds):
        round_start = time.time()
        
        # 1. Select clients and broadcast current model
        selected, power_alloc = server.select_clients()
        selected_ids = [c.client_id for c in selected]
        selected_counts.append(len(selected))
        
        # Update selection counts
        for cid in selected_ids:
            client_selection_counts[cid] += 1
            metrics['selection_counts'][cid] += 1
            
        # Broadcast model to selected clients
        server.broadcast_model(selected)
        
        # 2. Compute gradients on selected clients
        comp_times = []
        for client in selected:
            start_comp = time.time()
            client.compute_gradient()
            comp_time = time.time() - start_comp
            comp_times.append(comp_time)
        
        # 3. Reset staleness AFTER computation
        for client in selected:
            client.reset_staleness()
        
        # 4. Calculate round duration and aggregate
        max_comp_time = max(comp_times) if selected else 0
        D_t = max_comp_time + server.tau_cm
        
        if selected:
            aggregated = server.aggregate(selected, power_alloc)
            server.update_model(aggregated, round_idx)
        
        # 5. Update queues
        server.update_queues(selected, power_alloc, D_t)
        
        # 6. Update computation for all clients
        for client in clients:
            if client in selected:
                client.reset_computation()
            else:
                client.dt_k = max(0, client.dt_k - D_t)
                client.increment_staleness()
        
        # 7. Record metrics
        metrics['round_durations'].append(time.time() - round_start)
        
        # Energy tracking
        round_energy = 0
        for client in selected:
            comm_energy = power_alloc[client.client_id] * server.tau_cm
            total_energy = server.energy_this_round + comm_energy
            metrics['cumulative_energy_per_client'][client.client_id] += total_energy
            round_energy += total_energy
            server.energy_this_round = 0
        metrics['energy_per_round'].append(round_energy)
        
        # Evaluate every 5 rounds
        if (round_idx + 1) % 5 == 0 or round_idx == 0:
            acc = evaluate_model(server.global_model, test_loader, device)
            metrics['accuracies'].append(acc)
            elapsed_time = time.time() - metrics['start_time']
            metrics['time_points'].append(elapsed_time)
            metrics['time_based_accuracies'].append(acc)
    
    return metrics

def run_sync_ota(num_clients=10, num_rounds=300, device='cpu'):
    # Setup logging
    logger = logging.getLogger("SyncOTA")
    logger.setLevel(logging.INFO)
    
    # Load data
    train_dataset, test_dataset = load_mnist()
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    client_data_map = partition_mnist_noniid(train_dataset, num_clients)
    
    # Initialize clients
    clients = []
    energy_budgets = {}
    for cid in range(num_clients):
        clients.append(SyncClient(  # Using your SyncClient class
            client_id=cid,
            data_indices=client_data_map[cid],
            model=CNNMnist(),
            fk=np.random.uniform(1e9, 3e9),
            en=np.random.uniform(1e-6, 5e-6),
            P_max=2.0,
            train_dataset=train_dataset,
            device=device
        ))
        energy_budgets[cid] = np.random.uniform(0.5, 2.0)
    
    # Initialize server
    server = SyncServer(  # Using your SyncServer class
        global_model=CNNMnist(),
        clients=clients,
        total_rounds=num_rounds,
        batch_size=32,
        gamma0=10.0,
        sigma_n=0.05,
        G2=1.0,
        l_smooth=0.1,
        energy_budgets=energy_budgets,
        device=device
    )
    
    # Training metrics
    metrics = {
        'accuracies': [],
        'round_durations': [],
        'energy_per_round': [],
        'selection_counts': {cid: 0 for cid in range(num_clients)},
        'cumulative_energy_per_client': {cid: 0.0 for cid in range(num_clients)},
        'time_points': [],
        'time_based_accuracies': [],
        'start_time': time.time()
    }
    
    # Training loop (using your original main loop)
    accuracies = []
    round_times = []
    selection_counts = np.zeros(num_clients)
    energy_ratios = []
    snr_history = []
    
    for round_idx in range(num_rounds):
        start_time = time.time()
        
        # 1. Device selection
        V = 18.0
        selected, sigma_t = server.select_clients(round_idx, V)
        
        # Update selection counts
        for client in selected:
            selection_counts[client.client_id] += 1
            metrics['selection_counts'][client.client_id] += 1
        
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
        
        # 6. Record metrics
        metrics['round_durations'].append(time.time() - start_time)
        
        # Energy tracking
        round_energy = 0
        for client in selected:
            comm_energy = 0.1 * client.P_max * (1 + round_idx % 3)
            total_energy = client.energy_consumed + comm_energy
            metrics['cumulative_energy_per_client'][client.client_id] += total_energy
            round_energy += total_energy
            client.energy_consumed = 0
        metrics['energy_per_round'].append(round_energy)
        
        # Evaluate every 5 rounds
        if (round_idx + 1) % 5 == 0 or round_idx == 0:
            acc = evaluate_model(server.global_model, test_loader, device)
            metrics['accuracies'].append(acc)
            elapsed_time = time.time() - metrics['start_time']
            metrics['time_points'].append(elapsed_time)
            metrics['time_based_accuracies'].append(acc)
    
    return metrics

def run_cotaf(num_clients=10, num_rounds=300, device='cpu'):
    # Setup logging
    logger = logging.getLogger("COTAF")
    logger.setLevel(logging.INFO)
    
    # Load data
    train_dataset, test_dataset = load_mnist()
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    client_data_map = partition_mnist_noniid(train_dataset, num_clients)
    
    # Initialize clients
    clients = []
    for cid in range(num_clients):
        clients.append(COTAFClient(  # Using your COTAFClient class
            client_id=cid,
            data_indices=client_data_map[cid],
            model=CNNMnist(),
            fk=np.random.uniform(1e9, 2e9),
            mu_k=1e-27,
            P_max=0.5 + np.random.rand() * 0.5,
            C=1e6,
            Ak=32,
            train_dataset=train_dataset,
            device=device,
            local_epochs=1
        ))
    
    # Run experiment (using your run_cotaf_experiment function)
    metrics = run_cotaf_experiment(clients, num_rounds, 32, device)
    
    # Map COTAF metrics to our standard format
    standard_metrics = {
        'accuracies': metrics['accuracies'],
        'round_durations': metrics['round_times'],
        'energy_per_round': metrics['total_energy'],
        'selection_counts': {cid: num_rounds for cid in range(num_clients)},  # All participate
        'cumulative_energy_per_client': {client.client_id: 0.0 for client in clients},
        'time_points': [],
        'time_based_accuracies': [],
    }
    
    return standard_metrics

def run_safa(num_clients=10, num_rounds=300, device='cpu'):
    # Setup logging
    logger = logging.getLogger("SAFA")
    logger.setLevel(logging.INFO)
    
    # Parameters
    CRASH_PROB = 0.15
    LAG_TOLERANCE = 3
    SELECT_FRAC = 0.5
    
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
            logger.warning(f"Client {cid} received dummy data")
        
        clients.append(SAFAClient(  # Using your SAFAClient class
            client_id=cid,
            data_indices=indices,
            model=CNNMnist(),
            fk=np.random.uniform(1e9, 2e9),
            mu_k=1e-27,
            P_max=0.2,
            C=1e6,
            Ak=32,
            train_dataset=train_dataset,
            device=device,
            local_epochs=1,
            crash_prob=CRASH_PROB * np.random.uniform(0.8, 1.2)
        ))
    
    # Initialize server
    global_model = CNNMnist().to(device)
    server = SAFAServer(  # Using your SAFAServer class
        global_model=global_model,
        clients=clients,
        lag_tolerance=LAG_TOLERANCE,
        select_frac=SELECT_FRAC,
        learning_rate=0.01,
        device=device
    )
    
    # Training metrics (using your original metrics structure)
    metrics = {
        'accuracies': [],
        'round_durations': [],
        'energy_per_round': [],
        'selection_counts': {cid: 0 for cid in range(num_clients)},
        'cumulative_energy_per_client': {cid: 0.0 for cid in range(num_clients)},
        'time_points': [],
        'time_based_accuracies': [],
        'staleness': [],
        'comm_ratio': [],
        'start_time': time.time()
    }
    
    # Training loop (using your original main loop)
    for round_idx in range(num_rounds):
        # Run round
        eval_this_round = (round_idx % 5 == 0) or (round_idx == num_rounds - 1)
        round_duration = server.run_round(
            test_loader=test_loader if eval_this_round else None
        )
        
        # Record metrics
        metrics['round_durations'].append(round_duration)
        
        # Energy tracking
        round_energy = 0
        comm_energy = 0
        if round_idx < len(server.energy_history):
            round_energy_data = server.energy_history[round_idx]
            for energy_data in round_energy_data:
                cid = energy_data['client_id']
                total_energy = energy_data['total']
                metrics['cumulative_energy_per_client'][cid] += total_energy
                round_energy += total_energy
                comm_energy += energy_data['communication']
        metrics['energy_per_round'].append(round_energy)
        metrics['comm_ratio'].append(comm_energy / round_energy if round_energy > 0 else 0)
        
        # Staleness
        if round_idx < len(server.staleness_history):
            metrics['staleness'].append(server.staleness_history[round_idx])
        
        # Selection counts
        if round_idx < len(server.selection_history):
            for cid in server.selection_history[round_idx]:
                metrics['selection_counts'][cid] += 1
        
        # Accuracy tracking
        if eval_this_round and server.accuracy_history:
            metrics['accuracies'].append(server.accuracy_history[-1])
            elapsed_time = time.time() - metrics['start_time']
            metrics['time_points'].append(elapsed_time)
            metrics['time_based_accuracies'].append(server.accuracy_history[-1])
    
    return metrics

# ====================== COMPARISON AND PLOTTING ======================
def run_comparison(num_runs=10, num_clients=10, num_rounds=300):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    algorithms = {
        'semi_async': run_semi_async_ota,
        'sync': run_sync_ota,
        'cotaf': run_cotaf,
        'safa': run_safa
    }
    
    # Store results for all algorithms across runs
    all_results = {alg: [] for alg in algorithms}
    
    # Run each algorithm multiple times
    for run in range(num_runs):
        print(f"\n=== Run {run+1}/{num_runs} ===")
        for alg_name, alg_func in algorithms.items():
            print(f"Running {alg_name}...")
            start_time = time.time()
            results = alg_func(num_clients, num_rounds, device)
            duration = time.time() - start_time
            print(f"Completed {alg_name} in {duration:.2f} seconds")
            all_results[alg_name].append(results)
    
    # Average results across runs
    avg_results = {}
    for alg_name, runs in all_results.items():
        # Initialize averaged metrics
        avg_metrics = {
            'accuracies': [],
            'round_durations': [],
            'energy_per_round': [],
            'selection_counts': np.zeros(num_clients),
            'cumulative_energy_per_client': np.zeros(num_clients),
            'time_points': [],
            'time_based_accuracies': [],
            'staleness': [],
            'comm_ratio': []
        }
        
        num_eval_points = len(runs[0]['accuracies'])
        
        # Average each metric across runs
        for i in range(num_eval_points):
            acc_sum = sum(run['accuracies'][i] for run in runs)
            avg_metrics['accuracies'].append(acc_sum / num_runs)
            
            if i < len(runs[0]['time_points']):
                time_sum = sum(run['time_points'][i] for run in runs)
                avg_metrics['time_points'].append(time_sum / num_runs)
                acc_time_sum = sum(run['time_based_accuracies'][i] for run in runs)
                avg_metrics['time_based_accuracies'].append(acc_time_sum / num_runs)
        
        for i in range(num_rounds):
            if i < len(runs[0]['round_durations']):
                dur_sum = sum(run['round_durations'][i] for run in runs)
                avg_metrics['round_durations'].append(dur_sum / num_runs)
            
            if i < len(runs[0]['energy_per_round']):
                energy_sum = sum(run['energy_per_round'][i] for run in runs)
                avg_metrics['energy_per_round'].append(energy_sum / num_runs)
            
            # SAFA-specific metrics
            if alg_name == 'safa' and i < len(runs[0]['staleness']):
                staleness_sum = sum(run['staleness'][i] for run in runs)
                avg_metrics['staleness'].append(staleness_sum / num_runs)
            
            if alg_name == 'safa' and i < len(runs[0]['comm_ratio']):
                comm_sum = sum(run['comm_ratio'][i] for run in runs)
                avg_metrics['comm_ratio'].append(comm_sum / num_runs)
        
        # Sum selection counts and energy per client
        for cid in range(num_clients):
            select_sum = sum(run['selection_counts'][cid] for run in runs)
            avg_metrics['selection_counts'][cid] = select_sum / num_runs
            
            energy_sum = sum(run['cumulative_energy_per_client'][cid] for run in runs)
            avg_metrics['cumulative_energy_per_client'][cid] = energy_sum / num_runs
        
        avg_results[alg_name] = avg_metrics
    
    return avg_results

def plot_comparison(results_dict, num_clients, num_rounds):
    """Generate comparison plots for all methods"""
    methods = list(results_dict.keys())
    colors = {'semi_async': 'blue', 'sync': 'orange', 'cotaf': 'green', 'safa': 'red'}
    markers = {'semi_async': 'o', 'sync': 's', 'cotaf': 'd', 'safa': '^'}
    
    plt.figure(figsize=(30, 25))
    
    # 1. Accuracy comparison
    plt.subplot(331)
    for method in methods:
        plt.plot(results_dict[method]['accuracies'], 
                 marker=markers[method], markersize=8, markevery=5,
                 color=colors[method], linewidth=2, label=method.capitalize())
    plt.title("Test Accuracy Comparison", fontsize=16)
    plt.xlabel("Evaluation Rounds", fontsize=14)
    plt.ylabel("Accuracy (%)", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.ylim(0, 100)
    
    # 2. Energy per client
    plt.subplot(332)
    x = np.arange(num_clients)
    width = 0.2
    offset = -0.3
    
    for method in methods:
        energy_vals = [results_dict[method]['cumulative_energy_per_client'][cid] for cid in range(num_clients)]
        plt.bar(x + offset, energy_vals, width, label=method.capitalize(), color=colors[method])
        offset += width
    
    plt.title("Cumulative Energy per Client", fontsize=16)
    plt.xlabel("Client ID", fontsize=14)
    plt.ylabel("Total Energy (J)", fontsize=14)
    plt.xticks(x)
    plt.legend(fontsize=12)
    plt.grid(True)
    
    # 3. Total energy per round
    plt.subplot(333)
    for method in methods:
        plt.plot(results_dict[method]['energy_per_round'], 
                 color=colors[method], linewidth=2, label=method.capitalize())
    plt.title("Energy Consumption per Round", fontsize=16)
    plt.xlabel("Rounds", fontsize=14)
    plt.ylabel("Energy (J)", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    
    # 4. Client fairness (selection fairness)
    plt.subplot(334)
    fairness_vals = []
    for method in methods:
        counts = list(results_dict[method]['selection_counts'])
        fairness_vals.append(jains_fairness(counts))
    
    plt.bar(methods, fairness_vals, color=[colors[m] for m in methods])
    plt.ylim(0, 1.1)
    plt.title("Client Selection Fairness", fontsize=16)
    plt.ylabel("Jain's Fairness Index", fontsize=14)
    plt.grid(True)
    
    # 5. Round duration
    plt.subplot(335)
    for method in methods:
        plt.plot(results_dict[method]['round_durations'], 
                 color=colors[method], linewidth=1, label=method.capitalize())
    plt.title("Round Duration Comparison", fontsize=16)
    plt.xlabel("Rounds", fontsize=14)
    plt.ylabel("Time (s)", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    
    # 6. Accuracy vs. Wall-clock time
    plt.subplot(336)
    for method in methods:
        plt.plot(results_dict[method]['time_points'], 
                 results_dict[method]['time_based_accuracies'], 
                 marker=markers[method], markersize=8,
                 color=colors[method], linewidth=2, label=method.capitalize())
    plt.title("Accuracy vs. Wall-clock Time", fontsize=16)
    plt.xlabel("Time Elapsed (seconds)", fontsize=14)
    plt.ylabel("Test Accuracy (%)", fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=12)
    
    # 7. Client selection distribution
    plt.subplot(337)
    x = np.arange(num_clients)
    offset = -0.3
    for method in methods:
        counts = [results_dict[method]['selection_counts'][cid] for cid in range(num_clients)]
        plt.bar(x + offset, counts, width, label=method.capitalize(), color=colors[method])
        offset += width
    plt.title("Client Selection Distribution", fontsize=16)
    plt.xlabel("Client ID", fontsize=14)
    plt.ylabel("Times Selected", fontsize=14)
    plt.xticks(x)
    plt.legend(fontsize=12)
    plt.grid(True)
    
    # 8. Staleness comparison (SAFA only)
    if 'safa' in results_dict:
        plt.subplot(338)
        plt.plot(results_dict['safa']['staleness'], 'r-', linewidth=2, label='SAFA')
        plt.title("Staleness in SAFA", fontsize=16)
        plt.xlabel("Rounds", fontsize=14)
        plt.ylabel("Average Staleness", fontsize=14)
        plt.grid(True)
        plt.legend(fontsize=12)
    
    # 9. Communication energy ratio (SAFA only)
    if 'safa' in results_dict:
        plt.subplot(339)
        plt.plot(results_dict['safa']['comm_ratio'], 'b-', linewidth=2, label='SAFA')
        plt.title("Communication Energy Ratio (SAFA)", fontsize=16)
        plt.xlabel("Rounds", fontsize=14)
        plt.ylabel("Comm/Total Energy", fontsize=14)
        plt.grid(True)
        plt.legend(fontsize=12)
    
    plt.tight_layout(pad=3.0)
    plt.savefig("fl_comparison_results.png", dpi=300, bbox_inches='tight')
    plt.show()

# ====================== MAIN EXECUTION ======================
if __name__ == "__main__":
    # Run comparison
    num_runs = 1
    num_clients = 10
    num_rounds = 10  # Reduced for faster experimentation
    
    print(f"Starting FL algorithm comparison with {num_runs} runs...")
    comparison_results = run_comparison(num_runs, num_clients, num_rounds)
    
    # Plot results
    print("Generating comparison plots...")
    plot_comparison(comparison_results, num_clients, num_rounds)
    print("Comparison complete! Results saved to fl_comparison_results.png")