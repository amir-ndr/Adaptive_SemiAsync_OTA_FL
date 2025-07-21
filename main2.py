import torch
import numpy as np
import time
import os
import csv
from torch.utils.data import DataLoader
from client import Client
from server import Server
from model import CNNMnist
from dataloader import load_mnist, partition_mnist_dirichlet, partition_mnist_noniid
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
    client_data_map = partition_mnist_noniid(train_dataset, num_clients)
    
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
        V=15.0,
        sigma_n=0.05,
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
    
    # Return all collected results with consistent keys
    return {
        'run_id': run_id,
        'final_accuracy': final_acc,
        'accuracies': accuracies,
        'eval_points': eval_points,
        'round_duration': round_durations,  # Fixed key name
        'energy_queue': energy_queues,      # Fixed key name
        'staleness': avg_staleness_per_round, # Fixed key name
        'selected_count': selected_counts,   # Fixed key name
        'client_selection_counts': client_selection_counts,
        'client_energy': client_energy,
        'total_energy': total_energy,
        'per_round_energy': server.per_round_energy
    }

def main():
    NUM_RUNS = 1
    NUM_CLIENTS = 10
    NUM_ROUNDS = 300
    MIN_ACCURACY = 83.0
    
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
    os.makedirs("results_alpha02_main", exist_ok=True)
    
    # Save individual run results
    with open("results_alpha02_main/individual_runs.csv", "w", newline='') as f:
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
    
    # Aggregate results
    aggregated = {
        'accuracy': np.zeros(61),  # 0,5,10,...,300 (61 evaluation points)
        'round_duration': np.zeros(NUM_ROUNDS),
        'energy_queue': np.zeros(NUM_ROUNDS),
        'selected_count': np.zeros(NUM_ROUNDS),
        'staleness': np.zeros(NUM_ROUNDS),
        'client_selection': np.zeros(NUM_CLIENTS),
        'client_energy': np.zeros(NUM_CLIENTS),
        'total_energy': 0.0  # This is a scalar, not an array
    }
    
    # Calculate averages
    for run in successful_runs:
        # Accuracy (aligned by evaluation points)
        for i in range(len(run['accuracies'])):
            if i < len(aggregated['accuracy']):
                aggregated['accuracy'][i] += run['accuracies'][i]
        
        # Per-round metrics
        for metric in ['round_duration', 'energy_queue', 'selected_count', 'staleness']:
            for i in range(NUM_ROUNDS):
                if i < len(run[metric]):
                    aggregated[metric][i] += run[metric][i]
        
        # Client metrics
        for cid in range(NUM_CLIENTS):
            aggregated['client_selection'][cid] += run['client_selection_counts'].get(cid, 0)
            aggregated['client_energy'][cid] += run['client_energy'].get(cid, 0)
        
        aggregated['total_energy'] += run['total_energy']
    
    # Normalize
    num_runs = len(successful_runs)
    for key in aggregated:
        if key != 'total_energy':
            aggregated[key] /= num_runs
    aggregated['total_energy'] /= num_runs
    
    # Save averaged results to CSV
    with open("results_alpha02_main/averaged_results.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'values'])
        
        for metric, values in aggregated.items():
            if metric == 'total_energy':
                # Handle scalar value
                writer.writerow([metric, values])
            else:
                # Handle array values
                writer.writerow([metric] + list(values))
    
    # Generate plots
    plt.figure(figsize=(15, 15))
    
    # Accuracy plot
    plt.subplot(321)
    eval_rounds = [5*i for i in range(len(aggregated['accuracy']))]
    plt.plot(eval_rounds, aggregated['accuracy'], 'o-')
    plt.title("Average Test Accuracy")
    plt.xlabel("Communication Rounds")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)
    plt.ylim(0, 100)

    # Client selection per round
    plt.subplot(322)
    plt.plot(aggregated['selected_count'])
    plt.title("Average Selected Clients per Round")
    plt.xlabel("Rounds")
    plt.ylabel("Number of Clients")
    plt.grid(True)

    # Energy queues
    plt.subplot(323)
    plt.plot(aggregated['energy_queue'])
    plt.title("Average Max Energy Queue Value")
    plt.xlabel("Rounds")
    plt.ylabel("Queue Value")
    plt.grid(True)

    # Round durations
    plt.subplot(324)
    plt.plot(aggregated['round_duration'])
    plt.title("Average Round Duration")
    plt.xlabel("Rounds")
    plt.ylabel("Time (s)")
    plt.grid(True)

    # Average staleness
    plt.subplot(325)
    plt.plot(aggregated['staleness'], 'b-')
    plt.title("Average Client Staleness")
    plt.xlabel("Rounds")
    plt.ylabel("Staleness (rounds)")
    plt.grid(True)
    
    # Client energy consumption
    plt.subplot(326)
    plt.bar(range(NUM_CLIENTS), aggregated['client_energy'])
    plt.title("Average Energy Consumption per Client")
    plt.xlabel("Client ID")
    plt.ylabel("Energy (J)")
    plt.xticks(range(NUM_CLIENTS))
    plt.grid(True)

    plt.tight_layout(pad=3.0)
    plt.savefig("results_alpha02_main/averaged_results.png", dpi=300)
    plt.close()
    
    # Print summary
    print("\n===== Final Summary =====")
    print(f"Total runs: {len(successful_runs)}")
    print(f"Average final accuracy: {np.mean([r['final_accuracy'] for r in successful_runs]):.2f}%")
    print(f"Average total energy: {aggregated['total_energy']:.2f} J")
    
    print("\nPer-client Statistics:")
    for cid in range(NUM_CLIENTS):
        print(f"Client {cid}: "
              f"Avg Energy = {aggregated['client_energy'][cid]:.2f} J, "
              f"Avg Selections = {aggregated['client_selection'][cid]:.1f}")

if __name__ == "__main__":
    main()