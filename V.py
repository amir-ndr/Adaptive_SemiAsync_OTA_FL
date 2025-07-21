import torch
import numpy as np
import time
import os
import csv
from torch.utils.data import DataLoader
from client import Client
from server import Server
from model import CNNMnist
from dataloader import load_mnist, partition_mnist_dirichlet
import matplotlib.pyplot as plt
import logging

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

def run_experiment(run_id, num_clients=10, num_rounds=300, batch_size=32, V=5000.0):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"fl_system_run{run_id}_V{V}.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Starting FL simulation - Run {run_id} with V={V}")

    # Parameters
    LOCAL_EPOCHS = 1
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load data
    train_dataset, test_dataset = load_mnist()
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    client_data_map = partition_mnist_dirichlet(train_dataset, num_clients, alpha=100)
    
    # Initialize clients
    clients = []
    for cid in range(num_clients):
        indices = client_data_map[cid]
        if len(indices) == 0:
            indices = [0]
            logger.warning(f"Client {cid} has no data! Adding dummy sample")
        
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
            device=DEVICE,
            local_epochs=LOCAL_EPOCHS
        )
        clients.append(client)
    
    E_max_dict = {cid: np.random.uniform(25, 38) for cid in range(num_clients)}

    # Initialize server
    global_model = CNNMnist().to(DEVICE)
    server = Server(
        global_model=global_model,
        clients=clients,
        V=V,  # Use parameter V
        sigma_n=0.04,
        tau_cm=0.01,
        T_max=50,
        E_max=E_max_dict,
        T_total_rounds=num_rounds,
        device=DEVICE
    )
    
    # Training metrics
    accuracies = []
    round_durations = []
    energy_queues = []
    avg_staleness_per_round = []
    selected_counts = []
    client_selection_counts = {cid: 0 for cid in range(num_clients)}
    eval_points = []
    cumulative_energy = 0
    cumulative_energy_list = []

    for round_idx in range(num_rounds):
        round_start = time.time()
        
        # 1. Select clients and broadcast model
        selected, power_alloc = server.select_clients()
        selected_ids = [c.client_id for c in selected]
        selected_counts.append(len(selected))
        
        for cid in selected_ids:
            client_selection_counts[cid] += 1
            
        server.broadcast_model(selected)
        
        # 2. Compute gradients
        comp_times = []
        for client in selected:
            start_comp = time.time()
            client.compute_gradient()
            comp_times.append(time.time() - start_comp)
        
        # 3. Reset staleness
        for client in selected:
            client.reset_staleness()
        
        # 4. Aggregate updates
        max_comp_time = max(comp_times) if selected else 0
        D_t = max_comp_time + server.tau_cm
        
        if selected:
            aggregated = server.aggregate(selected, power_alloc)
            server.update_model(aggregated, round_idx)
        
        # 5. Update queues
        server.update_queues(selected, power_alloc, D_t)
        
        # 6. Update computation time
        for client in clients:
            if client in selected:
                client.reset_computation()
            else:
                client.dt_k = max(0, client.dt_k - D_t)
                client.increment_staleness()
        
        # 7. Record metrics
        current_avg_staleness = np.mean([client.tau_k for client in clients])
        avg_staleness_per_round.append(current_avg_staleness)
        round_durations.append(D_t)
        
        # Track energy usage
        round_energy = sum(server.per_round_energy[-1].values())
        cumulative_energy += round_energy
        cumulative_energy_list.append(cumulative_energy)
        
        # Evaluate periodically
        if (round_idx + 1) % 5 == 0 or round_idx == 0:
            acc = evaluate_model(server.global_model, test_loader, DEVICE)
            accuracies.append(acc)
            eval_points.append(round_idx)
    
    # Final evaluation
    final_acc = evaluate_model(server.global_model, test_loader, DEVICE)
    accuracies.append(final_acc)
    eval_points.append(num_rounds)
    
    # Calculate energy consumption
    total_energy = 0
    client_energy = {}
    for cid in range(num_clients):
        energy = sum(server.per_round_energy[r].get(cid, 0) 
                  for r in range(len(server.per_round_energy)))
        client_energy[cid] = energy
        total_energy += energy
    
    # Return all collected results
    return {
        'run_id': run_id,
        'V': V,
        'final_accuracy': final_acc,
        'accuracies': accuracies,
        'eval_points': eval_points,
        'round_duration': round_durations,
        'energy_queue': energy_queues,
        'staleness': avg_staleness_per_round,
        'selected_count': selected_counts,
        'client_selection_counts': client_selection_counts,
        'client_energy': client_energy,
        'total_energy': total_energy,
        'per_round_energy': server.per_round_energy,
        'cumulative_energy_per_round': cumulative_energy_list
    }

def main():
    V_VALUES = [12, 50, 100, 1000, 5000]
    NUM_RUNS = 6
    NUM_CLIENTS = 10
    NUM_ROUNDS = 300
    
    # Create results directory
    os.makedirs("results_v_study", exist_ok=True)
    
    # Run experiments for each V value
    for v in V_VALUES:
        v_results = []
        print(f"\n=== Running experiments for V={v} ===")
        
        for run_id in range(1, NUM_RUNS + 1):
            print(f"Run {run_id}/{NUM_RUNS} for V={v}")
            try:
                results = run_experiment(run_id, V=v, num_rounds=NUM_ROUNDS)
                v_results.append(results)
                print(f"Completed Run {run_id} - Accuracy: {results['final_accuracy']:.2f}%")
            except Exception as e:
                print(f"Run {run_id} failed: {str(e)}")
        
        # Save results for this V value
        save_v_results(v, v_results)

def save_v_results(v, results_list):
    """Save results for a specific V value to CSV files"""
    # Create directory for this V
    v_dir = f"results_v_study/V_{v}"
    os.makedirs(v_dir, exist_ok=True)
    NUM_CLIENTS = 10
    NUM_ROUNDS = 300
    
    # 1. Save per-run summary
    with open(f"{v_dir}/summary.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['run_id', 'final_accuracy', 'total_energy', 
                         *[f'client_{i}_energy' for i in range(NUM_CLIENTS)],
                         *[f'client_{i}_selections' for i in range(NUM_CLIENTS)]])
        
        for run in results_list:
            row = [
                run['run_id'],
                run['final_accuracy'],
                run['total_energy']
            ]
            # Client energies
            for cid in range(NUM_CLIENTS):
                row.append(run['client_energy'].get(cid, 0))
            # Client selections
            for cid in range(NUM_CLIENTS):
                row.append(run['client_selection_counts'].get(cid, 0))
            writer.writerow(row)
    
    # 2. Save detailed per-round metrics
    with open(f"{v_dir}/round_metrics.csv", "w", newline='') as f:
        writer = csv.writer(f)
        header = ['run_id', 'round', 'accuracy', 'cumulative_energy', 'selected_count']
        writer.writerow(header)
        
        for run in results_list:
            # Create mapping from round to accuracy
            acc_dict = dict(zip(run['eval_points'], run['accuracies']))
            
            for round_idx in range(NUM_ROUNDS):
                # Get accuracy if evaluated this round
                acc = acc_dict.get(round_idx, None)
                
                row = [
                    run['run_id'],
                    round_idx,
                    acc if acc is not None else '',
                    run['cumulative_energy_per_round'][round_idx],
                    run['selected_count'][round_idx]
                ]
                writer.writerow(row)
    
    # 3. Save per-client selection fractions
    with open(f"{v_dir}/client_selection_fractions.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['client_id', 'avg_selection_fraction'])
        
        # Calculate average selection fraction per client
        for cid in range(NUM_CLIENTS):
            total_fraction = 0
            for run in results_list:
                count = run['client_selection_counts'].get(cid, 0)
                total_fraction += count / NUM_ROUNDS
            avg_fraction = total_fraction / len(results_list)
            writer.writerow([cid, avg_fraction])
    
    print(f"Saved results for V={v} in {v_dir}")

def plot_results():
    """Plot results for all V values with unified scales"""
    V_VALUES = [12, 50, 100, 1000, 5000]
    plt.figure(figsize=(15, 10))
    
    # Plot accuracy progression
    plt.subplot(2, 2, 1)
    for v in V_VALUES:
        v_dir = f"results_v_study/V_{v}"
        avg_acc = []
        rounds = []
        
        with open(f"{v_dir}/round_metrics.csv", "r") as f:
            reader = csv.DictReader(f)
            current_round = -1
            sum_acc = 0
            count = 0
            
            for row in reader:
                if row['accuracy']:
                    round_idx = int(row['round'])
                    if round_idx != current_round:
                        if current_round != -1:
                            avg_acc.append(sum_acc / count)
                            rounds.append(current_round)
                        current_round = round_idx
                        sum_acc = 0
                        count = 0
                    sum_acc += float(row['accuracy'])
                    count += 1
            
            if count > 0:
                avg_acc.append(sum_acc / count)
                rounds.append(current_round)
        
        plt.plot(rounds, avg_acc, 'o-', label=f'V={v}')
    
    plt.title("Test Accuracy Progression")
    plt.xlabel("Communication Rounds")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)
    
    # Plot unified cumulative energy (0-1)
    plt.subplot(2, 2, 2)
    for v in V_VALUES:
        v_dir = f"results_v_study/V_{v}"
        energy_data = {r: [] for r in range(300)}
        
        with open(f"{v_dir}/round_metrics.csv", "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['cumulative_energy']:
                    round_idx = int(row['round'])
                    energy_data[round_idx].append(float(row['cumulative_energy']))
        
        # Find max energy for normalization
        max_energy = max(max(energies) for energies in energy_data.values() if energies)
        
        # Calculate normalized average
        avg_norm_energy = []
        for round_idx in range(300):
            if energy_data[round_idx]:
                avg = np.mean(energy_data[round_idx])
                avg_norm_energy.append(avg / max_energy)
            else:
                avg_norm_energy.append(0)
        
        plt.plot(range(300), avg_norm_energy, label=f'V={v}')
    
    plt.title("Normalized Cumulative Energy (0-1)")
    plt.xlabel("Communication Rounds")
    plt.ylabel("Normalized Energy")
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1.1)

    # Plot unified client selection fraction (0-1)
    plt.subplot(2, 2, 3)
    for v in V_VALUES:
        v_dir = f"results_v_study/V_{v}"
        selection_data = {r: [] for r in range(300)}
        
        with open(f"{v_dir}/round_metrics.csv", "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['selected_count']:
                    round_idx = int(row['round'])
                    # Convert count to fraction (0-1)
                    fraction = float(row['selected_count']) / 10
                    selection_data[round_idx].append(fraction)
        
        # Calculate average fraction
        avg_selection = []
        for round_idx in range(300):
            if selection_data[round_idx]:
                avg_selection.append(np.mean(selection_data[round_idx]))
            else:
                avg_selection.append(0)
        
        plt.plot(range(300), avg_selection, label=f'V={v}')
    
    plt.title("Average Client Selection Fraction (0-1)")
    plt.xlabel("Communication Rounds")
    plt.ylabel("Selection Fraction")
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1.1)

    # Plot client selection fractions (bar chart)
    plt.subplot(2, 2, 4)
    bar_width = 0.15
    for i, v in enumerate(V_VALUES):
        v_dir = f"results_v_study/V_{v}"
        fractions = []
        client_ids = []
        
        with open(f"{v_dir}/client_selection_fractions.csv", "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fractions.append(float(row['avg_selection_fraction']))
                client_ids.append(int(row['client_id']))
        
        positions = np.arange(10) + i * bar_width
        plt.bar(positions, fractions, width=bar_width, label=f'V={v}')
    
    plt.title("Average Client Selection Fraction")
    plt.xlabel("Client ID")
    plt.ylabel("Selection Fraction")
    plt.legend()
    plt.grid(True)
    plt.xticks(np.arange(10) + bar_width * len(V_VALUES) / 2, range(10))
    
    plt.tight_layout()
    plt.savefig("results_v_study/comparison_plots_unified.png")
    plt.close()

if __name__ == "__main__":
    main()
    plot_results()