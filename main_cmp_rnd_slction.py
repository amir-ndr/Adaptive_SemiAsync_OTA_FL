import torch
import numpy as np
import time
from torch.utils.data import DataLoader
from client import Client
from server import Server
from model import CNNMnist
from dataloader import load_mnist, partition_mnist_noniid
import matplotlib.pyplot as plt
import logging
import collections
import copy

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

def run_experiment(selection_method, clients, E_max_dict, NUM_ROUNDS, BATCH_SIZE, DEVICE):
    """Run a complete FL experiment with the specified selection method"""
    # Initialize global model
    global_model = CNNMnist().to(DEVICE)
    train_dataset, test_dataset = load_mnist()
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    # Initialize server
    server = Server(
        global_model=global_model,
        clients=clients,
        V=18.12,               # Lyapunov parameter
        sigma_n=0.09,          # Noise std
        tau_cm=0.01,           # Comm latency
        T_max=1041,             # Time budget (s)
        E_max=E_max_dict,      # Energy budget
        T_total_rounds=NUM_ROUNDS,
        device=DEVICE
    )
    
    # Training metrics
    accuracies = []
    round_durations = []
    energy_queues = []
    avg_staleness_per_round = []
    selected_counts = []
    client_selection_counts = {cid: 0 for cid in range(len(clients))}
    
    # Reset client states
    for client in clients:
        client.reset_staleness()
        client.reset_computation()
        client.gradient_norm = 1.0  # Reset gradient norm
    
    print(f"\n=== Starting experiment with {selection_method.__name__} selection ===")
    
    for round_idx in range(NUM_ROUNDS):
        round_start = time.time()
        print(f"\n=== Round {round_idx+1}/{NUM_ROUNDS} ===")
        
        # 1. Select clients using the specified method
        selected, power_alloc = selection_method(server)
        selected_ids = [c.client_id for c in selected]
        selected_counts.append(len(selected))
        
        # Update selection counts
        for cid in selected_ids:
            client_selection_counts[cid] += 1
            
        print(f"Selected {len(selected)} clients: {selected_ids}")
        print(f"Selection counts: {client_selection_counts}")
        
        # Broadcast model to selected clients
        server.broadcast_model(selected)
        
        # 2. Compute gradients on selected clients
        comp_times = []
        for client in selected:
            start_comp = time.time()
            client.compute_gradient()
            comp_time = time.time() - start_comp
            comp_times.append(comp_time)
            print(f"  Client {client.client_id}: "
                  f"Grad norm={client.gradient_norm:.4f}, "
                  f"Actual comp={comp_time:.4f}s")
        
        # 3. Reset staleness AFTER computation
        for client in selected:
            client.reset_staleness()
        
        # 4. Calculate round duration and aggregate
        max_comp_time = max(comp_times) if selected else 0
        D_t = max_comp_time + server.tau_cm
        
        if selected:
            aggregated = server.aggregate(selected, power_alloc)
            server.update_model(aggregated, round_idx)  # Pass round index for LR decay
        else:
            print("No clients selected - communication only round")
        
        # 5. Update queues
        server.update_queues(selected, power_alloc, D_t)
        
        # 6. Update computation time for ALL clients
        for client in clients:
            if client in selected:
                # Reset for next round (new model)
                client.reset_computation()
            else:
                # Progress computation
                client.dt_k = max(0, client.dt_k - D_t)
                client.increment_staleness()
        
        # 7. Record metrics and evaluate
        current_avg_staleness = np.mean([client.tau_k for client in clients])
        avg_staleness_per_round.append(current_avg_staleness)
        round_durations.append(D_t)
        
        # Evaluate every 5 rounds
        if (round_idx + 1) % 5 == 0 or round_idx == 0:
            acc = evaluate_model(server.global_model, test_loader, DEVICE)
            accuracies.append(acc)
            print(f"Global model accuracy: {acc:.2f}%")
        
        # Log round metrics
        round_time = time.time() - round_start
        max_energy_q = max(server.Q_e.values()) if server.Q_e else 0
        energy_queues.append(max_energy_q)
        
        print(f"Round duration: {D_t:.4f}s | "
              f"Wall time: {round_time:.2f}s | "
              f"Max energy queue: {max_energy_q:.2f} | "
              f"Time queue: {server.Q_time:.2f} | "
              f"Avg staleness: {current_avg_staleness:.2f}")
    
    # Final evaluation
    final_acc = evaluate_model(server.global_model, test_loader, DEVICE)
    accuracies.append(final_acc)
    
    return {
        'accuracies': accuracies,
        'final_acc': final_acc,
        'round_durations': round_durations,
        'energy_queues': energy_queues,
        'avg_staleness': avg_staleness_per_round,
        'selected_counts': selected_counts,
        'selection_counts': client_selection_counts,
        'model': server.global_model,
        'total_energy_per_round': server.total_energy_per_round,
        'cumulative_energy_per_client': server.cumulative_energy_per_client,
        'per_round_energy': server.per_round_energy
    }

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("fl_system.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting FL simulation")

    # Parameters
    NUM_CLIENTS = 10
    NUM_ROUNDS = 995
    BATCH_SIZE = 16
    LOCAL_EPOCHS = 1
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {DEVICE}")
    
    # Load data
    train_dataset, test_dataset = load_mnist()
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    client_data_map = partition_mnist_noniid(train_dataset, NUM_CLIENTS)
    
    # Initialize clients (will be reused for both experiments)
    base_clients = []
    for cid in range(NUM_CLIENTS):
        # Ensure client has at least 1 sample
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
            P_max=1.0 + np.random.rand(),
            C=1e6,
            Ak=BATCH_SIZE,
            train_dataset=train_dataset,
            device=DEVICE,
            local_epochs=LOCAL_EPOCHS
        )
        base_clients.append(client)
        print(f"Client {cid}: {len(client_data_map[cid])} samples | "
              f"Comp time: {client.dt_k:.4f}s")
    
    E_max_dict = {cid: np.random.uniform(5.71, 15.7) for cid in range(NUM_CLIENTS)}
    print("Client Energy Budgets:")
    for cid, budget in E_max_dict.items():
        print(f"  Client {cid}: {budget:.2f} J")

    # Run both experiments
    results = {}
    
    # 1. Run with algorithm selection
    algo_clients = copy.deepcopy(base_clients)
    results['algorithm'] = run_experiment(
        selection_method=Server.select_clients,
        clients=algo_clients,
        E_max_dict=E_max_dict,
        NUM_ROUNDS=NUM_ROUNDS,
        BATCH_SIZE=BATCH_SIZE,
        DEVICE=DEVICE
    )
    
    # 2. Run with random selection
    random_clients = copy.deepcopy(base_clients)
    results['random'] = run_experiment(
        selection_method=Server.random_selection,
        clients=random_clients,
        E_max_dict=E_max_dict,
        NUM_ROUNDS=NUM_ROUNDS,
        BATCH_SIZE=BATCH_SIZE,
        DEVICE=DEVICE
    )

    # Print final results
    print("\n=== Final Results ===")
    for method, res in results.items():
        print(f"{method.capitalize()} selection:")
        print(f"  Final accuracy: {res['final_acc']:.2f}%")
        print(f"  Average round duration: {np.mean(res['round_durations']):.2f}s")
        print(f"  Max energy queue: {max(res['energy_queues']):.2f}")
        
        # Selection statistics
        sorted_counts = sorted(res['selection_counts'].items(), key=lambda x: x[1], reverse=True)
        print("  Client selection counts:")
        for cid, count in sorted_counts:
            print(f"    Client {cid}: {count} times ({count/NUM_ROUNDS:.1%})")

    print("\n=== Energy Consumption Analysis ===")
    for method, res in results.items():
        total_energy = sum(res['cumulative_energy_per_client'].values())
        avg_per_round = np.mean(res['total_energy_per_round'])
        
        print(f"\n{method.capitalize()} selection:")
        print(f"  Total system energy: {total_energy:.2f} J")
        print(f"  Average energy per round: {avg_per_round:.2f} J")
        print(f"  Energy per client:")
        
        # Print client energy distribution
        sorted_energy = sorted(res['cumulative_energy_per_client'].items(), key=lambda x: x[1], reverse=True)
        for cid, energy in sorted_energy:
            budget_utilization = energy / E_max_dict[cid]
            print(f"    Client {cid}: {energy:.2f} J ({budget_utilization:.1%} of budget)")
    
    # Calculate and print energy savings
    algo_total = sum(results['algorithm']['cumulative_energy_per_client'].values())
    random_total = sum(results['random']['cumulative_energy_per_client'].values())
    savings = (random_total - algo_total) / random_total * 100
    
    print(f"\nEnergy Savings: Algorithm uses {savings:.1f}% less energy than Random selection")


    # Plot comparison
    plt.figure(figsize=(30, 25))
    
    # Accuracy comparison
    plt.subplot(331)
    eval_rounds_algo = [5*i for i in range(len(results['algorithm']['accuracies']))]
    eval_rounds_random = [5*i for i in range(len(results['random']['accuracies']))]
    plt.plot(eval_rounds_algo, results['algorithm']['accuracies'], 'o-', label='Algorithm Selection')
    plt.plot(eval_rounds_random, results['random']['accuracies'], 's-', label='Random Selection')
    plt.title("Accuracy Comparison")
    plt.xlabel("Communication Rounds")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)
    
    # Selection count comparison
    plt.subplot(332)
    plt.plot(results['algorithm']['selected_counts'], label='Algorithm')
    plt.plot(results['random']['selected_counts'], label='Random')
    plt.title("Selected Clients per Round")
    plt.xlabel("Rounds")
    plt.ylabel("Number of Clients")
    plt.legend()
    plt.grid(True)
    
    # Energy queue comparison
    plt.subplot(333)
    plt.plot(results['algorithm']['energy_queues'], label='Algorithm')
    plt.plot(results['random']['energy_queues'], label='Random')
    plt.title("Max Energy Queue Value")
    plt.xlabel("Rounds")
    plt.ylabel("Queue Value")
    plt.legend()
    plt.grid(True)
    
    # Staleness comparison
    plt.subplot(334)
    plt.plot(results['algorithm']['avg_staleness'], label='Algorithm')
    plt.plot(results['random']['avg_staleness'], label='Random')
    plt.title("Average Client Staleness")
    plt.xlabel("Rounds")
    plt.ylabel("Staleness (rounds)")
    plt.legend()
    plt.grid(True)
    
    # Selection distribution comparison
    plt.subplot(335)
    algo_counts = [results['algorithm']['selection_counts'][cid] for cid in range(NUM_CLIENTS)]
    random_counts = [results['random']['selection_counts'][cid] for cid in range(NUM_CLIENTS)]
    x = np.arange(NUM_CLIENTS)
    width = 0.35
    plt.bar(x - width/2, algo_counts, width, label='Algorithm')
    plt.bar(x + width/2, random_counts, width, label='Random')
    plt.title("Client Selection Distribution")
    plt.xlabel("Client ID")
    plt.ylabel("Times Selected")
    plt.xticks(x)
    plt.legend()
    plt.grid(True)
    
    # Energy distribution comparison
    # plt.subplot(336)
    # algo_energy = [results['algorithm']['energy_queues'][-1]] * NUM_CLIENTS  # Final energy queue
    # random_energy = [results['random']['energy_queues'][-1]] * NUM_CLIENTS
    # plt.bar(x - width/2, algo_energy, width, label='Algorithm')
    # plt.bar(x + width/2, random_energy, width, label='Random')
    # plt.title("Final Energy Queue Distribution")
    # plt.xlabel("Client ID")
    # plt.ylabel("Energy Queue Value")
    # plt.xticks(x)
    # plt.legend()
    # plt.grid(True)

    plt.subplot(336)  # 3x3 grid, position 7
    algo_energy = [results['algorithm']['cumulative_energy_per_client'][cid] for cid in range(NUM_CLIENTS)]
    random_energy = [results['random']['cumulative_energy_per_client'][cid] for cid in range(NUM_CLIENTS)]
    x = np.arange(NUM_CLIENTS)
    width = 0.35
    plt.bar(x - width/2, algo_energy, width, label='Algorithm')
    plt.bar(x + width/2, random_energy, width, label='Random')
    plt.title("Cumulative Energy per Client")
    plt.xlabel("Client ID")
    plt.ylabel("Total Energy (J)")
    plt.xticks(x)
    plt.legend()
    plt.grid(True)
    
    # 8. Energy proportion comparison
    # plt.subplot(338)
    # algo_proportions = [e/sum(algo_energy) for e in algo_energy]
    # random_proportions = [e/sum(random_energy) for e in random_energy]
    # plt.bar(x - width/2, algo_proportions, width, label='Algorithm')
    # plt.bar(x + width/2, random_proportions, width, label='Random')
    # plt.title("Energy Distribution Proportion")
    # plt.xlabel("Client ID")
    # plt.ylabel("Proportion of Total Energy")
    # plt.xticks(x)
    # plt.legend()
    # plt.grid(True)
    
    # 9. Energy fairness comparison
    plt.subplot(337)
    # Calculate Jain's fairness index
    def jains_fairness(values):
        return sum(values)**2 / (len(values) * sum(v**2 for v in values))
    
    fairness_algo = jains_fairness(algo_energy)
    fairness_random = jains_fairness(random_energy)
    
    plt.bar(['Algorithm', 'Random'], [fairness_algo, fairness_random], color=['blue', 'orange'])
    plt.ylim(0, 1.1)
    plt.title("Energy Fairness Comparison")
    plt.ylabel("Jain's Fairness Index")
    plt.grid(True)
    
    plt.tight_layout(pad=3.0)
    plt.savefig("fl_comparison_results.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()