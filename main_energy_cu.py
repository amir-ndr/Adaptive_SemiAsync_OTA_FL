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
import pandas as pd  # Added for CSV export

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
    NUM_ROUNDS = 300
    BATCH_SIZE = 32
    LOCAL_EPOCHS = 1
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {DEVICE}")
    
    # Load data
    train_dataset, test_dataset = load_mnist()
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    client_data_map = partition_mnist_noniid(train_dataset, NUM_CLIENTS)
    
    # Initialize clients
    clients = []
    for cid in range(NUM_CLIENTS):
        # Ensure client has at least 1 sample
        indices = client_data_map[cid]
        if len(indices) == 0:
            indices = [0]  # Add dummy index to prevent errors
            logger.warning(f"Client {cid} has no data! Adding dummy sample")
        
        client = Client(
            client_id=cid,
            data_indices=indices,
            model=CNNMnist(),
            fk=np.random.uniform(1e9, 2e9),  # 1-2 GHz CPU
            mu_k=1e-27,                      # Energy coefficient
            P_max=2.0 + np.random.rand(),     # Max transmit power
            C=1e6,                           # FLOPs per sample
            Ak=BATCH_SIZE,                   # Batch size
            train_dataset=train_dataset,
            device=DEVICE,
            local_epochs=LOCAL_EPOCHS
        )
        clients.append(client)
        print(f"Client {cid}: {len(client_data_map[cid])} samples | "
              f"Comp time: {client.dt_k:.4f}s")
    
    E_max_dict = {cid: np.random.uniform(0.01, 0.025) for cid in range(NUM_CLIENTS)}
    print("Client Energy Budgets:")
    for cid, budget in E_max_dict.items():
        print(f"  Client {cid}: {budget:.2f} J")

    # Initialize server
    global_model = CNNMnist().to(DEVICE)
    server = Server(
        global_model=global_model,
        clients=clients,
        V=10,               # Lyapunov parameter
        sigma_n=0.5,          # Noise std
        tau_cm=0.01,           # Comm latency
        T_max=50,             # Time budget (s)
        E_max=E_max_dict,      # Energy budget
        T_total_rounds=NUM_ROUNDS,
        device=DEVICE
    )
    
    # Training loop
    accuracies = []
    round_durations = []
    energy_queues = []
    avg_staleness_per_round = []
    selected_counts = []  # Track number of selected clients per round
    client_selection_counts = {cid: 0 for cid in range(NUM_CLIENTS)}  # Track per-client selection count
    
    # Energy tracking variables
    cumulative_energy = 0.0
    cumulative_energy_per_round = []
    energy_breakdown = []  # Store energy consumption per round

    for round_idx in range(NUM_ROUNDS):
        round_start = time.time()
        print(f"\n=== Round {round_idx+1}/{NUM_ROUNDS} ===")
        
        # 1. Select clients and broadcast current model
        selected, power_alloc = server.select_clients()
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
        round_energy = 0.0
        energy_per_client = {}
        
        for client in selected:
            start_comp = time.time()
            client.compute_gradient()
            comp_time = time.time() - start_comp
            comp_times.append(comp_time)
            
            # Calculate energy consumption for this client
            comp_energy = client.mu_k * (client.fk ** 3) * comp_time
            comm_energy = power_alloc[client.client_id] * server.tau_cm
            client_energy = comp_energy + comm_energy
            round_energy += client_energy
            
            # Record per-client energy
            energy_per_client[client.client_id] = client_energy
            
            print(f"  Client {client.client_id}: "
                  f"Grad norm={client.gradient_norm:.4f}, "
                  f"Actual comp={comp_time:.4f}s, "
                  f"Energy={client_energy:.6f}J")
        
        # 3. Reset staleness AFTER computation
        for client in selected:
            client.reset_staleness()
        
        # 4. Calculate round duration and aggregate
        max_comp_time = max(comp_times) if selected else 0
        D_t = max_comp_time + server.tau_cm
        
        if selected:
            aggregated = server.aggregate(selected, power_alloc)
            server.update_model(aggregated, round_idx)
        else:
            print("No clients selected - communication only round")
            round_energy = 0.0
        
        # 5. Update cumulative energy
        cumulative_energy += round_energy
        cumulative_energy_per_round.append(cumulative_energy)
        
        # Store energy breakdown for this round
        energy_breakdown.append({
            'round': round_idx + 1,
            'round_energy': round_energy,
            'cumulative_energy': cumulative_energy,
            **{f'client_{cid}_energy': energy_per_client.get(cid, 0.0) 
               for cid in range(NUM_CLIENTS)}
        })
        
        # 6. Update queues
        server.update_queues(selected, power_alloc, D_t)
        
        # 7. Update computation time for ALL clients
        for client in clients:
            if client in selected:
                client.reset_computation()
            else:
                client.dt_k = max(0, client.dt_k - D_t)
                client.increment_staleness()
        
        # 8. Record metrics and evaluate
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
              f"Avg staleness: {current_avg_staleness:.2f} | "
              f"Round energy: {round_energy:.6f}J | "
              f"Cumulative energy: {cumulative_energy:.6f}J")

    # Final evaluation
    final_acc = evaluate_model(server.global_model, test_loader, DEVICE)
    accuracies.append(final_acc)
    print(f"\n=== Training Complete ===")
    print(f"Final accuracy: {final_acc:.2f}%")
    print(f"Average round duration: {np.mean(round_durations):.2f}s")
    print(f"Max energy queue: {max(energy_queues):.2f}")
    print(f"Total cumulative energy: {cumulative_energy:.6f}J")
    
    # Print client selection statistics
    print("\nClient Selection Statistics:")
    sorted_counts = sorted(client_selection_counts.items(), key=lambda x: x[1], reverse=True)
    for cid, count in sorted_counts:
        print(f"Client {cid}: Selected {count} times ({count/NUM_ROUNDS:.1%} of rounds)")
    
    # Save energy data to CSV
    energy_df = pd.DataFrame(energy_breakdown)
    energy_df.to_csv('energy_consumption.csv', index=False)
    print("Saved energy data to 'energy_consumption.csv'")
    
    # Plot cumulative energy
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, NUM_ROUNDS+1), cumulative_energy_per_round, 'b-', linewidth=2)
    plt.title('Cumulative Energy Consumption')
    plt.xlabel('Round')
    plt.ylabel('Cumulative Energy (Joules)')
    plt.grid(True)
    plt.savefig('cumulative_energy.png')
    print("Saved cumulative energy plot to 'cumulative_energy.png'")
    
    # Optional: Plot per-round energy too
    per_round_energy = [e['round_energy'] for e in energy_breakdown]
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, NUM_ROUNDS+1), per_round_energy, 'r-', linewidth=1)
    plt.title('Per-Round Energy Consumption')
    plt.xlabel('Round')
    plt.ylabel('Energy per Round (Joules)')
    plt.grid(True)
    plt.savefig('per_round_energy.png')
    print("Saved per-round energy plot to 'per_round_energy.png'")

if __name__ == "__main__":
    main()