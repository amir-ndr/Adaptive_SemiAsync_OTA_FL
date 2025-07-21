import torch
import numpy as np
import time
import logging
import matplotlib.pyplot as plt
import pandas as pd
import os
import copy
import math
from collections import defaultdict
from torch.utils.data import DataLoader
from dataloader import load_mnist, partition_mnist_noniid
from model import CNNMnist
from client import Client
from server import Server
from client_sync import SyncClient
from server_sync import SyncServer
from server_cotaf import COTAFServer
from client_cotaf import COTAFClient
from safa_client import SAFAClient
from safa_server import SAFAServer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("fl_comparison.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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

def jains_fairness(values):
    """Calculate Jain's fairness index"""
    return (sum(values) ** 2) / (len(values) * sum(v ** 2 for v in values))

# def run_semi_async_experiment(clients, E_max_dict, NUM_ROUNDS, BATCH_SIZE, DEVICE, test_dataset):
    # """Run semi-asynchronous OTA experiment"""
    # logger.info("Running Semi-Async OTA Experiment")
    
    # # Initialize server
    # global_model = CNNMnist().to(DEVICE)
    # server = Server(
    #     global_model=global_model,
    #     clients=clients,
    #     V=15.0,
    #     sigma_n=0.04,
    #     tau_cm=0.01,
    #     T_max=50,
    #     E_max=E_max_dict,
    #     T_total_rounds=NUM_ROUNDS,
    #     device=DEVICE
    # )
    
    # # Training loop
    # results = {
    #     'accuracies': [],
    #     'round_durations': [],
    #     'selected_counts': [],
    #     'selection_counts': {cid: 0 for cid in range(len(clients))},
    #     'total_energy_per_round': [],
    #     'energy_per_client_per_round': [],
    #     'cumulative_energy_per_client': {client.client_id: 0.0 for client in clients},
    #     'time_points': [],
    #     'time_based_accuracies': []
    # }
    
    # start_time = time.time()
    # for round_idx in range(NUM_ROUNDS):
    #     round_start = time.time()
        
    #     # Select clients and broadcast
    #     selected, power_alloc = server.select_clients()
    #     selected_ids = [c.client_id for c in selected]
    #     results['selected_counts'].append(len(selected))
        
    #     for cid in selected_ids:
    #         results['selection_counts'][cid] += 1
            
    #     server.broadcast_model(selected)
        
    #     # Compute gradients
    #     comp_times = []
    #     for client in selected:
    #         start_comp = time.time()
    #         client.compute_gradient()
    #         comp_time = time.time() - start_comp
    #         comp_times.append(comp_time)
        
    #     # Reset staleness
    #     for client in selected:
    #         client.reset_staleness()
        
    #     # Aggregate and update
    #     max_comp_time = max(comp_times) if selected else 0
    #     D_t = max_comp_time + server.tau_cm
    #     results['round_durations'].append(D_t)
        
    #     if selected:
    #         aggregated = server.aggregate(selected, power_alloc)
    #         server.update_model(aggregated, round_idx)
        
    #     # Update queues and track energy
    #     round_energy = 0
    #     per_client_energy = {}
    #     for client in selected:
    #         # Compute actual energy
    #         comp_energy = client.mu_k * client.fk**2 * client.C * client.Ak
    #         tx_energy = (power_alloc[client.client_id] * client.gradient_norm)**2 / (abs(client.h_t_k)**2 + 1e-8)
    #         client_energy = tx_energy #+ comp_energy
    #         round_energy += client_energy
    #         per_client_energy[client.client_id] = client_energy
    #         results['cumulative_energy_per_client'][client.client_id] += client_energy
        
    #     results['total_energy_per_round'].append(round_energy)
    #     results['energy_per_client_per_round'].append(per_client_energy)
    #     server.update_queues(selected, power_alloc, D_t)
        
    #     # Update computation time
    #     for client in clients:
    #         if client in selected:
    #             client.reset_computation()
    #         else:
    #             client.dt_k = max(0, client.dt_k - D_t)
    #             client.increment_staleness()
        
    #     # Evaluate
    #     if (round_idx + 1) % 5 == 0:
    #         test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    #         acc = evaluate_model(server.global_model, test_loader, DEVICE)
    #         results['accuracies'].append(acc)
    #         elapsed_time = time.time() - start_time
    #         results['time_points'].append(elapsed_time)
    #         results['time_based_accuracies'].append(acc)
    
    # # Final evaluation
    # test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    # final_acc = evaluate_model(server.global_model, test_loader, DEVICE)
    # results['accuracies'].append(final_acc)
    # results['time_based_accuracies'].append(final_acc)
    # results['time_points'].append(time.time() - start_time)
    
    # logger.info(f"Semi-Async Final accuracy: {final_acc:.2f}%")
    # return results

def run_semi_async_experiment(clients, E_max_dict, NUM_ROUNDS, BATCH_SIZE, DEVICE, test_dataset):
    """Run semi-asynchronous OTA experiment"""
    logger.info("Running Semi-Async OTA Experiment")
    
    # Initialize server
    global_model = CNNMnist().to(DEVICE)
    server = Server(
        global_model=global_model,
        clients=clients,
        V=15.0,
        sigma_n=0.009,
        tau_cm=0.01,
        T_max=500,
        E_max=E_max_dict,
        T_total_rounds=NUM_ROUNDS,
        device=DEVICE
    )
    
    # Training loop
    results = {
        'accuracies': [],
        'round_durations': [],
        'selected_counts': [],
        'selection_counts': {cid: 0 for cid in range(len(clients))},
        'total_energy_per_round': [],
        'energy_per_client_per_round': [],
        'time_points': [],
        'time_based_accuracies': []
    }
    
    start_time = time.time()
    for round_idx in range(NUM_ROUNDS):
        round_start = time.time()
        
        # Select clients and broadcast
        selected, power_alloc = server.select_clients()
        selected_ids = [c.client_id for c in selected]
        results['selected_counts'].append(len(selected))
        
        for cid in selected_ids:
            results['selection_counts'][cid] += 1
            
        server.broadcast_model(selected)
        
        # Compute gradients
        comp_times = []
        for client in selected:
            start_comp = time.time()
            client.compute_gradient()
            comp_time = time.time() - start_comp
            comp_times.append(comp_time)
        
        # Reset staleness
        for client in selected:
            client.reset_staleness()
        
        # Aggregate and update
        max_comp_time = max(comp_times) if selected else 0
        D_t = max_comp_time + server.tau_cm
        results['round_durations'].append(D_t)
        
        if selected:
            aggregated = server.aggregate(selected, power_alloc)
            server.update_model(aggregated, round_idx)
        
        # Update queues - server will track energy internally
        server.update_queues(selected, power_alloc, D_t)
        
        # Get energy metrics from server after queue update
        results['total_energy_per_round'].append(server.energy_this_round)
        results['energy_per_client_per_round'].append(server.per_round_energy[-1])
        
        # Update computation time
        for client in clients:
            if client in selected:
                client.reset_computation()
            else:
                client.dt_k = max(0, client.dt_k - D_t)
                client.increment_staleness()
        
        # Evaluate
        if (round_idx + 1) % 5 == 0:
            test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
            acc = evaluate_model(server.global_model, test_loader, DEVICE)
            results['accuracies'].append(acc)
            elapsed_time = time.time() - start_time
            results['time_points'].append(elapsed_time)
            results['time_based_accuracies'].append(acc)
    
    # Final evaluation
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    final_acc = evaluate_model(server.global_model, test_loader, DEVICE)
    results['accuracies'].append(final_acc)
    results['time_based_accuracies'].append(final_acc)
    results['time_points'].append(time.time() - start_time)
    
    # Add cumulative energy for compatibility with main plotting
    cumulative_energy = {cid: 0.0 for cid in range(len(clients))}
    for r in range(len(results['energy_per_client_per_round'])):
        for cid, energy in results['energy_per_client_per_round'][r].items():
            cumulative_energy[cid] += energy
    results['cumulative_energy_per_client'] = cumulative_energy
    
    logger.info(f"Semi-Async Final accuracy: {final_acc:.2f}%")
    return results

def run_sync_ota_experiment(clients, E_max_dict, NUM_ROUNDS, BATCH_SIZE, DEVICE, test_dataset):
    """Run synchronous OTA experiment"""
    logger.info("Running Sync OTA Experiment")
    
    # Initialize server
    server = SyncServer(
        global_model=CNNMnist(),
        clients=clients,
        total_rounds=NUM_ROUNDS,
        batch_size=BATCH_SIZE,
        gamma0=10.0,
        sigma_n=0.05,
        G2=1.0,
        l_smooth=0.1,
        energy_budgets=E_max_dict,
        device=DEVICE
    )
    
    # Training loop
    results = {
        'accuracies': [],
        'round_durations': [],
        'selected_counts': [],
        'selection_counts': {cid: 0 for cid in range(len(clients))},
        'total_energy_per_round': [],
        'energy_per_client_per_round': [],
        'cumulative_energy_per_client': {client.client_id: 0.0 for client in clients},
        'time_points': [],
        'time_based_accuracies': []
    }
    
    start_time = time.time()
    for round_idx in range(NUM_ROUNDS):
        round_start = time.time()
        
        # Device selection
        V = 15.0
        selected, sigma_t = server.select_clients(round_idx, V)
        selected_ids = [c.client_id for c in selected]
        results['selected_counts'].append(len(selected))
        
        for cid in selected_ids:
            results['selection_counts'][cid] += 1
            
        # Broadcast model
        global_state = server.global_model.state_dict()
        for client in selected:
            client.update_model(global_state)
        
        # Gradient computation and aggregation
        aggregated_update, actual_norms = server.aggregate_gradients(selected, sigma_t)
        
        # Update model
        server.update_model(aggregated_update, round_idx)
        
        # Track energy
        round_energy = 0
        per_client_energy = {}
        round_duration = time.time() - round_start
        results['round_durations'].append(round_duration)
        
        for client in selected:
            comp_energy = client.en * BATCH_SIZE
            h_sq = abs(client.channel_gain) ** 2
            comm_energy = (sigma_t ** 2) / h_sq * (actual_norms[client.client_id] ** 2)
            client_energy = comm_energy #+comp_energy
            round_energy += client_energy
            per_client_energy[client.client_id] = client_energy
            results['cumulative_energy_per_client'][client.client_id] += client_energy
        
        results['total_energy_per_round'].append(round_energy)
        results['energy_per_client_per_round'].append(per_client_energy)
        server.update_queues(selected, actual_norms, sigma_t, round_idx)
        
        # Evaluate
        if (round_idx + 1) % 5 == 0:
            test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
            acc = evaluate_model(server.global_model, test_loader, DEVICE)
            results['accuracies'].append(acc)
            elapsed_time = time.time() - start_time
            results['time_points'].append(elapsed_time)
            results['time_based_accuracies'].append(acc)
    
    # Final evaluation
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    final_acc = evaluate_model(server.global_model, test_loader, DEVICE)
    results['accuracies'].append(final_acc)
    results['time_based_accuracies'].append(final_acc)
    results['time_points'].append(time.time() - start_time)
    
    logger.info(f"Sync Final accuracy: {final_acc:.2f}%")
    return results

def run_cotaf_experiment(clients, E_max_dict, NUM_ROUNDS, BATCH_SIZE, DEVICE, test_dataset):
    """Run COTAF experiment with proper energy calculation"""
    logger = logging.getLogger("COTAF")
    logger.setLevel(logging.INFO)
    logger.info("Running COTAF Experiment")
    
    # Initialize global model
    global_model = CNNMnist().to(DEVICE)
    
    # Initialize COTAF server
    server = COTAFServer(
        global_model=global_model,
        clients=clients,
        P_max=0.4,
        noise_var=0.1,
        H_local=1,
        device=DEVICE
    )

    # Training loop
    results = {
        'accuracies': [],
        'round_durations': [],
        'selected_counts': [],
        'selection_counts': {cid: 0 for cid in range(len(clients))},
        'total_energy_per_round': [],
        'energy_per_client_per_round': [],
        'cumulative_energy_per_client': {client.client_id: 0.0 for client in clients},
        'time_points': [],
        'time_based_accuracies': [],
        'start_time': time.time()
    }
    
    for round_idx in range(NUM_ROUNDS):
        round_start = time.time()
        
        # 1. Broadcast global model
        server.broadcast_model()
        
        # 2. Local training
        for client in clients:
            client.local_train()
        
        # 3. COTAF aggregation (energy is tracked inside this method)
        new_state = server.aggregate()
        server.update_model(new_state)
        
        # Track all clients as selected
        results['selected_counts'].append(len(clients))
        for client in clients:
            results['selection_counts'][client.client_id] += 1
        
        # 4. Get energy from server's tracker (key change)
        if server.energy_tracker['per_round_total']:
            round_energy = server.energy_tracker['per_round_total'][-1]
            per_client_energy = {client.client_id: 0 for client in clients}#server.energy_tracker['per_client_per_round'][-1]
        else:
            round_energy = 0
            per_client_energy = {client.client_id: 0 for client in clients}
            logger.warning("No energy data recorded for this round!")

        # Update energy tracking
        for client_id, energy_val in per_client_energy.items():
            results['cumulative_energy_per_client'][client_id] += energy_val

        results['total_energy_per_round'].append(round_energy)
        results['energy_per_client_per_round'].append(per_client_energy)
        round_duration = time.time() - round_start
        results['round_durations'].append(round_duration)
        
        # 5. Evaluate periodically
        if (round_idx + 1) % 5 == 0 or round_idx == 0:
            test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
            acc = evaluate_model(server.global_model, test_loader, DEVICE)
            results['accuracies'].append(acc)
            elapsed_time = time.time() - results['start_time']
            results['time_points'].append(elapsed_time)
            results['time_based_accuracies'].append(acc)
            
            logger.info(f"Round {round_idx+1}/{NUM_ROUNDS} | "
                        f"Accuracy: {acc:.2f}% | "
                        f"Round Energy: {round_energy:.2f}J | "
                        f"Duration: {round_duration:.2f}s")
        
        # 6. Reset client states for next round
        for client in clients:
            client.reset_round()
    
    # Final evaluation
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    final_acc = evaluate_model(server.global_model, test_loader, DEVICE)
    results['accuracies'].append(final_acc)
    results['time_based_accuracies'].append(final_acc)
    results['time_points'].append(time.time() - results['start_time'])
    
    logger.info(f"COTAF Final accuracy: {final_acc:.2f}%")
    logger.info(f"Total experiment time: {results['time_points'][-1]:.2f} seconds")
    return results

def run_safa_experiment(clients, E_max_dict, NUM_ROUNDS, BATCH_SIZE, DEVICE, test_dataset):
    """Run SAFA experiment"""
    logger.info("Running SAFA Experiment")
    
    # Initialize global model
    global_model = CNNMnist().to(DEVICE)
    
    # SAFA parameters
    LAG_TOLERANCE = 3
    SELECT_FRAC = 0.4
    EVAL_INTERVAL = 5
    
    # Initialize SAFA server
    server = SAFAServer(
        global_model=global_model,
        clients=clients,
        lag_tolerance=LAG_TOLERANCE,
        select_frac=SELECT_FRAC,
        learning_rate=0.01,
        device=DEVICE
    )

    # Training loop
    results = {
        'accuracies': [],
        'round_durations': [],
        'selected_counts': [],
        'selection_counts': {cid: 0 for cid in range(len(clients))},
        'total_energy_per_round': [],
        'energy_per_client_per_round': [],
        'cumulative_energy_per_client': {client.client_id: 0.0 for client in clients},
        'time_points': [],
        'time_based_accuracies': [],
        'staleness': [],
        'comm_ratio': []
    }
    
    start_time = time.time()
    for round_idx in range(NUM_ROUNDS):
        round_start = time.time()
        
        # Run federated round
        eval_this_round = (round_idx % EVAL_INTERVAL == 0) or (round_idx == NUM_ROUNDS - 1)
        round_duration = server.run_round(
            test_loader=DataLoader(test_dataset, batch_size=256, shuffle=False) if eval_this_round else None
        )
        
        # Track metrics
        results['round_durations'].append(round_duration)
        results['selected_counts'].append(len(server.selection_history[-1]))
        
        # Record client selection counts
        for cid in server.selection_history[-1]:
            results['selection_counts'][cid] += 1
        
        # Energy tracking
        round_energy = 0
        per_client_energy = {}
        if server.energy_history and round_idx < len(server.energy_history):
            round_energy_list = server.energy_history[round_idx]
            for energy_dict in round_energy_list:
                cid = energy_dict['client_id']
                client_energy = energy_dict['total']
                round_energy += client_energy
                per_client_energy[cid] = client_energy
                results['cumulative_energy_per_client'][cid] += client_energy
            
            # Calculate communication ratio
            comm_energy = sum(e['communication'] for e in round_energy_list)
            results['comm_ratio'].append(comm_energy / round_energy if round_energy > 0 else 0)
        else:
            results['comm_ratio'].append(0)
        
        results['total_energy_per_round'].append(round_energy)
        results['energy_per_client_per_round'].append(per_client_energy)
        
        # Staleness tracking
        if server.staleness_history and round_idx < len(server.staleness_history):
            results['staleness'].append(server.staleness_history[round_idx])
        else:
            results['staleness'].append(0)
        
        # Accuracy tracking
        if eval_this_round and server.accuracy_history:
            acc = server.accuracy_history[-1]
            results['accuracies'].append(acc)
            elapsed_time = time.time() - start_time
            results['time_points'].append(elapsed_time)
            results['time_based_accuracies'].append(acc)
    
    # Final evaluation
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    final_acc = server.evaluate(test_loader)
    results['accuracies'].append(final_acc)
    results['time_based_accuracies'].append(final_acc)
    results['time_points'].append(time.time() - start_time)
    
    logger.info(f"SAFA Final accuracy: {final_acc:.2f}%")
    return results

def create_clients_for_algorithm(algorithm, base_clients, train_dataset, device):
    if algorithm == 'semi_async':
        return copy.deepcopy(base_clients)
    elif algorithm == 'sync':
        return [
            SyncClient(
                client_id=client.client_id,
                data_indices=client.data_indices,
                model=CNNMnist(),
                fk=client.fk,
                en=1e-6,
                P_max=client.P_max,
                train_dataset=train_dataset,
                device=device
            ) for client in base_clients
        ]
    elif algorithm == 'cotaf':
        return [
            COTAFClient(
                client_id=client.client_id,
                data_indices=client.data_indices,
                model=CNNMnist(),
                fk=client.fk,
                mu_k=1e-27,
                P_max=2.0 + np.random.rand(),
                C=1e6,
                Ak=client.Ak,
                train_dataset=train_dataset,
                device=device,
                local_epochs=1
            ) for client in base_clients
        ]
    elif algorithm == 'safa':
        return [
            SAFAClient(
                client_id=client.client_id,
                data_indices=client.data_indices,
                model=CNNMnist(),
                fk=client.fk,
                mu_k=1e-27,
                P_max=2.0 + np.random.rand(),
                C=1e6,
                Ak=client.Ak,
                train_dataset=train_dataset,
                device=device,
                local_epochs=1,
                crash_prob=0.15
            ) for client in base_clients
        ]

def run_one_experiment(algorithm, clients, E_max_dict, NUM_ROUNDS, BATCH_SIZE, device, test_dataset):
    if algorithm == 'semi_async':
        return run_semi_async_experiment(clients, E_max_dict, NUM_ROUNDS, BATCH_SIZE, device, test_dataset)
    elif algorithm == 'sync':
        return run_sync_ota_experiment(clients, E_max_dict, NUM_ROUNDS, BATCH_SIZE, device, test_dataset)
    elif algorithm == 'cotaf':
        return run_cotaf_experiment(clients, E_max_dict, NUM_ROUNDS, BATCH_SIZE, device, test_dataset)
    elif algorithm == 'safa':
        return run_safa_experiment(clients, E_max_dict, NUM_ROUNDS, BATCH_SIZE, device, test_dataset)

def save_raw_data(raw_data, filename):
    df = pd.DataFrame(raw_data)
    df.to_csv(filename, index=False)
    logger.info(f"Saved raw data to {filename}")

def compute_average_metrics(all_results):
    """Compute average metrics across multiple runs with proper type handling"""
    avg_results = {}
    if not all_results:
        return avg_results
    
    # Handle each metric type separately
    for metric in all_results[0].keys():
        # Skip unsupported metrics
        if metric in ['energy_per_client_per_round']:
            continue
            
        # Dictionary metrics (e.g., cumulative_energy_per_client, selection_counts)
        if isinstance(all_results[0][metric], dict):
            avg_dict = {}
            all_keys = set().union(*(d[metric].keys() for d in all_results))
            
            for key in all_keys:
                values = []
                for run in all_results:
                    if key in run[metric]:
                        values.append(run[metric][key])
                if values:  # Only average if we have values
                    avg_dict[key] = np.mean(values)
                    
            avg_results[metric] = avg_dict
            
        # List metrics (e.g., accuracies, round_durations)
        elif isinstance(all_results[0][metric], list):
            # Handle numeric lists
            if all_results[0][metric] and isinstance(all_results[0][metric][0], (int, float)):
                min_len = min(len(run[metric]) for run in all_results)
                truncated = [run[metric][:min_len] for run in all_results]
                avg_results[metric] = np.mean(truncated, axis=0).tolist()
                
            # Handle list of dictionaries (like SAFA's staleness)
            elif all_results[0][metric] and isinstance(all_results[0][metric][0], dict):
                # We'll convert to a more manageable format
                pass
                
        # Single value metrics
        elif isinstance(all_results[0][metric], (int, float)):
            values = [run[metric] for run in all_results]
            avg_results[metric] = np.mean(values)
            
    return avg_results

def plot_comparison(results_dict, num_clients, num_rounds):
    """Generate comparison plots for all methods including Accuracy vs Energy"""
    methods = list(results_dict.keys())
    colors = {'semi_async': 'blue', 'sync': 'orange', 'cotaf': 'green', 'safa': 'red'}
    markers = {'semi_async': 'o', 'sync': 's', 'cotaf': 'd', 'safa': '^'}
    
    plt.figure(figsize=(30, 25))
    
    # 1. Accuracy comparison
    plt.subplot(331)
    for method in methods:
        eval_points = [5*i for i in range(len(results_dict[method]['accuracies']))]
        plt.plot(eval_points, results_dict[method]['accuracies'], 
                 marker=markers[method], label=method.capitalize())
    plt.title("Test Accuracy Comparison")
    plt.xlabel("Communication Rounds")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)
    
    # 2. Energy per client
    plt.subplot(332)
    x = np.arange(num_clients)
    width = 0.2
    offset = -0.3
    
    for method in methods:
        energy_vals = [results_dict[method]['cumulative_energy_per_client'][cid] for cid in range(num_clients)]
        plt.bar(x + offset, energy_vals, width, label=method.capitalize())
        offset += width
    
    plt.title("Cumulative Energy per Client")
    plt.xlabel("Client ID")
    plt.ylabel("Total Energy (J)")
    plt.xticks(x)
    plt.legend()
    plt.grid(True)
    
    # 3. Total energy per round
    plt.subplot(333)
    for method in methods:
        plt.plot(results_dict[method]['total_energy_per_round'], label=method.capitalize())
    plt.title("Energy Consumption per Round")
    plt.xlabel("Rounds")
    plt.ylabel("Energy (J)")
    plt.legend()
    plt.grid(True)
    
    # 4. Client fairness (selection fairness)
    plt.subplot(334)
    fairness_vals = []
    for method in methods:
        counts = list(results_dict[method]['selection_counts'].values())
        fairness_vals.append(jains_fairness(counts))
    
    plt.bar(methods, fairness_vals, color=[colors[m] for m in methods])
    plt.ylim(0, 1.1)
    plt.title("Client Selection Fairness")
    plt.ylabel("Jain's Fairness Index")
    plt.grid(True)
    
    # 5. Round duration
    plt.subplot(335)
    for method in methods:
        plt.plot(results_dict[method]['round_durations'], label=method.capitalize())
    plt.title("Round Duration Comparison")
    plt.xlabel("Rounds")
    plt.ylabel("Time (s)")
    plt.legend()
    plt.grid(True)
    
    # 6. Accuracy vs. Wall-clock time
    plt.subplot(336)
    for method in methods:
        plt.plot(results_dict[method]['time_points'], 
                 results_dict[method]['time_based_accuracies'], 
                 marker=markers[method], label=method.capitalize())
    plt.title("Accuracy vs. Wall-clock Time")
    plt.xlabel("Time Elapsed (seconds)")
    plt.ylabel("Test Accuracy (%)")
    plt.grid(True)
    plt.legend()
    
    # 7. Client selection distribution
    plt.subplot(337)
    x = np.arange(num_clients)
    offset = -0.3
    for method in methods:
        counts = [results_dict[method]['selection_counts'][cid] for cid in range(num_clients)]
        plt.bar(x + offset, counts, width, label=method.capitalize())
        offset += width
    plt.title("Client Selection Distribution")
    plt.xlabel("Client ID")
    plt.ylabel("Times Selected")
    plt.xticks(x)
    plt.legend()
    plt.grid(True)
    
    # 8. Accuracy vs. Cumulative Energy (NEW PLOT)
    plt.subplot(338)
    for method in methods:
        # Calculate cumulative energy
        cumulative_energy = np.cumsum(results_dict[method]['total_energy_per_round'])
        
        # Get accuracy values and their corresponding time points
        accuracies = results_dict[method]['accuracies']
        time_points = results_dict[method]['time_points']
        
        # Find the closest energy points for each accuracy measurement
        energy_at_eval = []
        for t in time_points:
            # Find the index where this time point would be inserted
            idx = np.searchsorted(results_dict[method]['time_points'], t)
            if idx >= len(cumulative_energy):
                energy_at_eval.append(cumulative_energy[-1])
            else:
                energy_at_eval.append(cumulative_energy[idx])
        
        plt.plot(energy_at_eval, accuracies, 
                 marker=markers[method], 
                 label=method.capitalize())
    
    plt.title("Accuracy vs. Cumulative Energy")
    plt.xlabel("Cumulative Energy (J)")
    plt.ylabel("Test Accuracy (%)")
    plt.grid(True)
    plt.legend()
    
    # 9. Hide the 9th subplot (no longer needed)
    plt.subplot(339)
    plt.axis('off')
    
    plt.tight_layout(pad=3.0)
    plt.savefig("fl_comparison_results.png", dpi=300)
    plt.show()

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Parameters
    NUM_CLIENTS = 10
    NUM_ROUNDS = 300
    BATCH_SIZE = 32
    NUM_RUNS = 1
    algorithms = ['semi_async', 'sync', 'cotaf', 'safa']
    
    # Load data once
    train_dataset, test_dataset = load_mnist()
    
    # Data structures for results
    all_results = {alg: [] for alg in algorithms}
    raw_per_round = []
    raw_per_eval = []
    raw_per_client = []
    raw_energy_per_round = []
    
    for run_idx in range(NUM_RUNS):
        logger.info(f"Starting run {run_idx+1}/{NUM_RUNS}")
        np.random.seed(run_idx)
        torch.manual_seed(run_idx)
        
        # Partition data for this run
        client_data_map = partition_mnist_noniid(train_dataset, NUM_CLIENTS)
        
        # Create base clients with new random parameters
        base_clients = []
        for cid in range(NUM_CLIENTS):
            indices = client_data_map[cid]
            if len(indices) == 0:
                indices = [0]
                logger.warning(f"Client {cid} has no data! Adding dummy sample")
            
            base_clients.append(Client(
                client_id=cid,
                data_indices=indices,
                model=CNNMnist(),
                fk=np.random.uniform(1e9, 2e9),
                mu_k=1e-27,
                P_max=2.0 + np.random.rand(),
                C=1e6,
                Ak=BATCH_SIZE,
                train_dataset=train_dataset,
                device=device,
                local_epochs=1
            ))
        
        # Energy budgets for this run
        E_max_dict = {cid: np.random.uniform(25, 38) for cid in range(NUM_CLIENTS)}
        
        for algorithm in algorithms:
            logger.info(f"Running {algorithm} algorithm")
            clients = create_clients_for_algorithm(algorithm, base_clients, train_dataset, device)
            results = run_one_experiment(
                algorithm, 
                clients, 
                E_max_dict, 
                NUM_ROUNDS, 
                BATCH_SIZE, 
                device, 
                test_dataset
            )
            
            # Store results
            all_results[algorithm].append(results)
            
            # Flatten and save raw per-round data
            for round_idx in range(NUM_ROUNDS):
                row = {
                    'run': run_idx,
                    'algorithm': algorithm,
                    'round': round_idx,
                    'duration': results['round_durations'][round_idx],
                    'selected_count': results['selected_counts'][round_idx],
                    'energy': results['total_energy_per_round'][round_idx],
                }
                if algorithm == 'safa':
                    row['staleness'] = results['staleness'][round_idx]
                    row['comm_ratio'] = results['comm_ratio'][round_idx] if round_idx < len(results['comm_ratio']) else 0
                raw_per_round.append(row)
            
            # Flatten and save raw per-evaluation data
            for eval_idx in range(len(results['accuracies'])):
                raw_per_eval.append({
                    'run': run_idx,
                    'algorithm': algorithm,
                    'eval_index': eval_idx,
                    'accuracy': results['accuracies'][eval_idx],
                    'time': results['time_points'][eval_idx]
                })
            
            # Flatten and save raw per-client data
            for client_id in range(NUM_CLIENTS):
                raw_per_client.append({
                    'run': run_idx,
                    'algorithm': algorithm,
                    'client_id': client_id,
                    'selection_count': results['selection_counts'][client_id],
                    'cumulative_energy': results['cumulative_energy_per_client'][client_id]
                })
            
            # Flatten and save raw energy per client per round
            for round_idx, per_client_energy in enumerate(results['energy_per_client_per_round']):
                for client_id, energy in per_client_energy.items():
                    raw_energy_per_round.append({
                        'run': run_idx,
                        'algorithm': algorithm,
                        'round': round_idx,
                        'client_id': client_id,
                        'energy': energy
                    })
    
    # Save raw data
    os.makedirs('results', exist_ok=True)
    save_raw_data(raw_per_round, 'results/raw_per_round.csv')
    save_raw_data(raw_per_eval, 'results/raw_per_eval.csv')
    save_raw_data(raw_per_client, 'results/raw_per_client.csv')
    save_raw_data(raw_energy_per_round, 'results/raw_energy_per_round.csv')
    
    # Compute and save averaged results
    avg_results_dict = {}
    for algorithm in algorithms:
        avg_results = compute_average_metrics(all_results[algorithm])
        avg_results_dict[algorithm] = avg_results
    
    # Save averaged metrics
    avg_per_round = []
    for algorithm, results in avg_results_dict.items():
        for round_idx in range(NUM_ROUNDS):
            row = {
                'algorithm': algorithm,
                'round': round_idx,
                'avg_duration': results['round_durations'][round_idx],
                'avg_selected_count': results['selected_counts'][round_idx],
                'avg_energy': results['total_energy_per_round'][round_idx],
            }
            if algorithm == 'safa':
                row['avg_staleness'] = results['staleness'][round_idx]
                row['avg_comm_ratio'] = results['comm_ratio'][round_idx] if round_idx < len(results['comm_ratio']) else 0
            avg_per_round.append(row)
    
    avg_per_eval = []
    for algorithm, results in avg_results_dict.items():
        for eval_idx in range(len(results['accuracies'])):
            avg_per_eval.append({
                'algorithm': algorithm,
                'eval_index': eval_idx,
                'avg_accuracy': results['accuracies'][eval_idx],
                'avg_time': results['time_points'][eval_idx]
            })
    
    avg_per_client = []
    for algorithm, results in avg_results_dict.items():
        for client_id in range(NUM_CLIENTS):
            avg_per_client.append({
                'algorithm': algorithm,
                'client_id': client_id,
                'avg_selection_count': results['selection_counts'][client_id],
                'avg_cumulative_energy': results['cumulative_energy_per_client'][client_id]
            })
    
    save_raw_data(avg_per_round, 'results/avg_per_round.csv')
    save_raw_data(avg_per_eval, 'results/avg_per_eval.csv')
    save_raw_data(avg_per_client, 'results/avg_per_client.csv')
    
    # Generate plots using averaged data
    plot_comparison(avg_results_dict, NUM_CLIENTS, NUM_ROUNDS)
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    for algorithm, results in avg_results_dict.items():
        print(f"{algorithm.capitalize()} Final Accuracy: {results['accuracies'][-1]:.2f}%")
    
    print("\nEnergy Efficiency:")
    for algorithm, results in avg_results_dict.items():
        avg_energy = np.mean(results['total_energy_per_round'])
        print(f"  {algorithm.capitalize()} Avg Energy/Round: {avg_energy:.2f} J")
    
    print("\nTime Efficiency:")
    for algorithm, results in avg_results_dict.items():
        avg_time = np.mean(results['round_durations'])
        print(f"  {algorithm.capitalize()} Avg Round Time: {avg_time:.2f} s")
    
    print("\nTotal Wall-clock Time:")
    for algorithm, results in avg_results_dict.items():
        print(f"  {algorithm.capitalize()}: {results['time_points'][-1]:.2f} s")
    
    print("\nClient Selection Fairness (Jain's Index):")
    for algorithm, results in avg_results_dict.items():
        counts = list(results['selection_counts'].values())
        fairness = jains_fairness(counts)
        print(f"  {algorithm.capitalize()}: {fairness:.4f}")
    
    print("\nTotal System Energy:")
    for algorithm, results in avg_results_dict.items():
        total_energy = sum(results['total_energy_per_round'])
        print(f"  {algorithm.capitalize()}: {total_energy:.2f} J")
    
    # SAFA-specific metrics
    if 'safa' in avg_results_dict:
        safa_results = avg_results_dict['safa']
        print("\nSAFA Specific Metrics:")
        print(f"  Average Staleness: {np.mean(safa_results['staleness']):.2f} rounds")
        print(f"  Max Staleness: {np.max(safa_results['staleness']):.2f} rounds")
        print(f"  Average Comm Ratio: {np.mean(safa_results['comm_ratio']):.2f}")
        print(f"  Crash Probability: 15% (simulated)")

if __name__ == "__main__":
    main()