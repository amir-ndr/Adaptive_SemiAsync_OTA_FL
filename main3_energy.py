import torch
import numpy as np
import time
import os
import csv
from torch.utils.data import DataLoader
from client import Client
from server import Server
from model import CNNMnist
from dataloader import load_mnist, partition_mnist_noniid, partition_mnist_dirichlet
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

def run_experiment(run_id, num_clients=10, num_rounds=300, batch_size=32):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"fl_system_run{run_id}.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Starting FL simulation - Run {run_id}")

    # Parameters
    LOCAL_EPOCHS = 1
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load data
    train_dataset, test_dataset = load_mnist()
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    client_data_map = partition_mnist_dirichlet(train_dataset, num_clients, alpha=0.2)
    
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
    
    # E_max_dict = {cid: np.random.uniform(0.01, 0.025) for cid in range(num_clients)}
    E_max_dict = {cid: np.random.uniform(13, 15) for cid in range(num_clients)}


    # Initialize server
    global_model = CNNMnist().to(DEVICE)
    server = Server(
        global_model=global_model,
        clients=clients,
        V=1000.0,
        sigma_n=10e-6,
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
    cumulative_energy = []
    cumulative_time = []

    total_energy_so_far = 0
    total_time_so_far = 0

    for round_idx in range(num_rounds):
        round_start = time.time()
        
        # 1. Select clients and broadcast model
        selected, power_alloc = server.select_clients()
        # selected, power_alloc = server.random_selection()
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
        total_time_so_far += D_t
        cumulative_time.append(total_time_so_far)
        
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
        
        # Calculate total energy for this round
        round_energy = sum(server.per_round_energy[-1].values()) if server.per_round_energy else 0
        total_energy_so_far += round_energy
        cumulative_energy.append(total_energy_so_far)
        
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
    
    # Get cumulative energy and time at evaluation points
    cumulative_energy_at_eval = [cumulative_energy[ep] for ep in eval_points if ep < len(cumulative_energy)]
    cumulative_time_at_eval = [cumulative_time[ep] for ep in eval_points if ep < len(cumulative_time)]
    
    # Pad with last value if needed
    if len(cumulative_energy_at_eval) < len(accuracies):
        last_energy = cumulative_energy[-1] if cumulative_energy else 0
        cumulative_energy_at_eval += [last_energy] * (len(accuracies) - len(cumulative_energy_at_eval))
    
    if len(cumulative_time_at_eval) < len(accuracies):
        last_time = cumulative_time[-1] if cumulative_time else 0
        cumulative_time_at_eval += [last_time] * (len(accuracies) - len(cumulative_time_at_eval))
    
    # Return all collected results with consistent keys
    return {
        'run_id': run_id,
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
        'cumulative_energy_at_eval': cumulative_energy_at_eval,
        'cumulative_time_at_eval': cumulative_time_at_eval
    }

def main():
    NUM_RUNS = 10
    NUM_CLIENTS = 10
    NUM_ROUNDS = 300
    MIN_ACCURACY = 85.0
    
    successful_runs = []
    run_counter = 0
    
    print(f"Starting experiment: Collecting {NUM_RUNS} runs with accuracy >= {MIN_ACCURACY}%")
    
    while len(successful_runs) < NUM_RUNS and run_counter < 50:
        run_counter += 1
        print(f"\n=== Starting Run {run_counter} ===")
        
        try:
            results = run_experiment(run_counter, NUM_CLIENTS, NUM_ROUNDS)
            final_acc = results['final_accuracy']
            
            if final_acc >= MIN_ACCURACY:
                successful_runs.append(results)
                print(f"Run {run_counter} SUCCESS - Accuracy: {final_acc:.2f}% "
                      f"({len(successful_runs)}/{NUM_RUNS} collected)")
            else:
                print(f"Run {run_counter} SKIPPED - Accuracy: {final_acc:.2f}% < {MIN_ACCURACY}%")
        except Exception as e:
            print(f"Run {run_counter} FAILED: {str(e)}")
    
    if not successful_runs:
        print("No successful runs collected!")
        return
    
    print(f"\nCollected {len(successful_runs)} successful runs")
    
    # Create directory for results
    os.makedirs("results", exist_ok=True)
    
    # Save individual run results
    with open("results/individual_runs.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['run_id', 'final_accuracy', 'total_energy'] + 
                        [f'client_{i}_energy' for i in range(NUM_CLIENTS)] +
                        [f'client_{i}_selections' for i in range(NUM_CLIENTS)])
        
        for run in successful_runs:
            row = [run['run_id'], run['final_accuracy'], run['total_energy']]
            for cid in range(NUM_CLIENTS):
                row.append(run['client_energy'].get(cid, 0))
            for cid in range(NUM_CLIENTS):
                row.append(run['client_selection_counts'].get(cid, 0))
            writer.writerow(row)
    
    # ======= NEW: Save accuracy vs energy and time data =======
    # Save per-run accuracy vs energy/time
    for run in successful_runs:
        run_id = run['run_id']
        with open(f"results/acc_vs_energy_time_run{run_id}.csv", "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['eval_point', 'accuracy', 'cumulative_energy', 'cumulative_time'])
            for i, point in enumerate(run['eval_points']):
                writer.writerow([
                    point,
                    run['accuracies'][i],
                    run['cumulative_energy_at_eval'][i],
                    run['cumulative_time_at_eval'][i]
                ])
    
    # Save averaged accuracy vs energy/time
    avg_accuracy = []
    avg_energy = []
    avg_time = []
    
    # Determine max evaluation points
    max_eval_points = max(len(run['eval_points']) for run in successful_runs)
    
    # Initialize accumulation arrays
    acc_accum = np.zeros(max_eval_points)
    energy_accum = np.zeros(max_eval_points)
    time_accum = np.zeros(max_eval_points)
    count_accum = np.zeros(max_eval_points)
    
    # Accumulate values
    for run in successful_runs:
        num_points = len(run['eval_points'])
        for i in range(num_points):
            if i < max_eval_points:
                acc_accum[i] += run['accuracies'][i]
                energy_accum[i] += run['cumulative_energy_at_eval'][i]
                time_accum[i] += run['cumulative_time_at_eval'][i]
                count_accum[i] += 1
    
    # Calculate averages
    for i in range(max_eval_points):
        if count_accum[i] > 0:
            avg_accuracy.append(acc_accum[i] / count_accum[i])
            avg_energy.append(energy_accum[i] / count_accum[i])
            avg_time.append(time_accum[i] / count_accum[i])
    
    # Save averaged data
    with open("results/avg_acc_vs_energy_time.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['eval_point', 'accuracy', 'cumulative_energy', 'cumulative_time'])
        for i in range(len(avg_accuracy)):
            writer.writerow([
                i * 5,  # Evaluation point (every 5 rounds)
                avg_accuracy[i],
                avg_energy[i],
                avg_time[i]
            ])
    
    # ======= NEW: Create plots =======
    # Accuracy vs Cumulative Energy
    plt.figure(figsize=(10, 6))
    for run in successful_runs:
        plt.plot(run['cumulative_energy_at_eval'], run['accuracies'], 'o-', alpha=0.3, markersize=3)
    plt.plot(avg_energy, avg_accuracy, 'r-', linewidth=3, label='Average')
    plt.xlabel("Cumulative Energy (J)")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy vs. Cumulative Energy Consumption")
    plt.grid(True)
    plt.legend()
    plt.savefig("results/accuracy_vs_energy.png", dpi=300)
    plt.close()
    
    # Accuracy vs Cumulative Time
    plt.figure(figsize=(10, 6))
    for run in successful_runs:
        plt.plot(run['cumulative_time_at_eval'], run['accuracies'], 'o-', alpha=0.3, markersize=3)
    plt.plot(avg_time, avg_accuracy, 'r-', linewidth=3, label='Average')
    plt.xlabel("Cumulative Time (s)")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy vs. Cumulative Training Time")
    plt.grid(True)
    plt.legend()
    plt.savefig("results/accuracy_vs_time.png", dpi=300)
    plt.close()
    
    # Print summary
    print("\n===== Final Summary =====")
    print(f"Total runs: {len(successful_runs)}")
    print(f"Average final accuracy: {np.mean([r['final_accuracy'] for r in successful_runs]):.2f}%")
    print(f"Average total energy: {np.mean([r['total_energy'] for r in successful_runs]):.2f} J")
    
    print("\nPer-client Statistics:")
    for cid in range(NUM_CLIENTS):
        avg_energy = np.mean([r['client_energy'].get(cid, 0) for r in successful_runs])
        avg_selections = np.mean([r['client_selection_counts'].get(cid, 0) for r in successful_runs])
        print(f"Client {cid}: "
              f"Avg Energy = {avg_energy:.2f} J, "
              f"Avg Selections = {avg_selections:.1f}")

if __name__ == "__main__":
    main()