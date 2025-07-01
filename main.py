import torch
import numpy as np
import time
from torch.utils.data import DataLoader
from client import Client
from server import Server
# from server2 import Server
from model import CNNMnist
from dataloader import load_mnist, partition_mnist_noniid
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
    NUM_ROUNDS = 100
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
        client = Client(
            client_id=cid,
            data_indices=client_data_map[cid],
            model=CNNMnist(),
            fk=np.random.uniform(1e9, 2e9),  # 1-2 GHz CPU
            mu_k=1e-27,                      # Energy coefficient
            P_max=2.0 + np.random.rand(),                       # Max transmit power
            C=1e6,                           # FLOPs per sample
            Ak=BATCH_SIZE,                   # Batch size
            train_dataset=train_dataset,
            device=DEVICE,
            local_epochs=LOCAL_EPOCHS
        )
        clients.append(client)
        print(f"Client {cid}: {len(client_data_map[cid])} samples | "
              f"Comp time: {client.dt_k:.4f}s")
    
    E_max_dict = {cid: np.random.uniform(30, 70) for cid in range(NUM_CLIENTS)}
    print("Client Energy Budgets:")
    for cid, budget in E_max_dict.items():
        print(f"  Client {cid}: {budget:.2f} J")

    # Initialize server
    global_model = CNNMnist().to(DEVICE)
    server = Server(
        global_model=global_model,
        clients=clients,
        V=25.0,               # Lyapunov parameter
        sigma_n=0.01,           # Noise std
        tau_cm=0.01,           # Comm latency
        T_max=200,             # Time budget (s)
        E_max=E_max_dict,            # Energy budget
        T_total_rounds=NUM_ROUNDS,
        device=DEVICE
    )
    
    # Training loop
    accuracies = []
    round_durations = []
    energy_queues = []
    avg_staleness_per_round = []
    # Remove D_t_prev since we no longer need it

    for round_idx in range(NUM_ROUNDS):
        round_start = time.time()
        print(f"\n=== Round {round_idx+1}/{NUM_ROUNDS} ===")
        
        # REMOVED: Initial computation time update (not needed)
        
        # 1. Select clients and broadcast current model
        selected, power_alloc = server.select_clients()
        print(f"Selected {len(selected)} clients: {[c.client_id for c in selected]}")
        
        # 2. Reset staleness for selected clients (computation time reset happens later)
        server.broadcast_model(selected)  # Update model first
        for client in selected:
            client.reset_staleness()  # Update model
        
        # 3. Compute gradients on selected clients
        comp_times = []
        for client in selected:
            start_comp = time.time()
            client.compute_gradient()
            comp_time = time.time() - start_comp
            comp_times.append(comp_time)
            print(f"  Client {client.client_id}: Grad norm={client.gradient_norm:.4f}, Comp time={comp_time:.2f}s")
        
        # 4. Calculate round duration and aggregate
        if selected:
            # Get max computation time from actual execution
            max_comp_time = max(comp_times)
            D_t = max_comp_time + server.tau_cm
            aggregated = server.aggregate(selected, power_alloc)
            server.update_model(aggregated)
        else:
            D_t = server.tau_cm
            print("No clients selected - communication only round")
        
        # 5. Update queues (energy/time)
        server.update_queues(selected, power_alloc, D_t)
        
        # 6. Update computation time and staleness for ALL clients
        for client in clients:
            if client in selected:
                # Selected clients: reset to FULL computation time for NEXT round
                client.dt_k = client._full_computation_time()
            else:
                # Non-selected clients: subtract current round duration
                client.dt_k = max(0, client.dt_k - D_t)
                client.increment_staleness()
        
        # 7. Record metrics
        current_avg_staleness = np.mean([client.tau_k for client in clients])
        avg_staleness_per_round.append(current_avg_staleness)
        print(f"Avg Staleness={current_avg_staleness:.2f}")
        round_durations.append(D_t)
            
        total_energy = 0
        for client in selected:
            comp_energy = client.mu_k * client.fk**2 * client.C * client.Ak
            # comp_energy = 0.1
            comm_energy = (power_alloc[client.client_id] * client.gradient_norm / abs(client.h_t_k))**2
            total_energy += comp_energy + comm_energy
        print(f"Round energy: {total_energy:.2f} J")

        # 6. Evaluate every 5 rounds
        if (round_idx + 1) % 5 == 0 or round_idx == 0:
            acc = evaluate_model(server.global_model, test_loader, DEVICE)
            accuracies.append(acc)
            print(f"Global model accuracy: {acc:.2f}%")
        
        # 7. Log metrics
        round_time = time.time() - round_start
        max_energy_q = max(server.Q_e.values()) if server.Q_e else 0
        energy_queues.append(max_energy_q)
        
        print(f"Round duration: {D_t:.4f}s | "
              f"Wall time: {round_time:.2f}s | "
              f"Max energy queue: {max_energy_q:.2f} | "
              f"Time queue: {server.Q_time:.2f}")

    # Final evaluation
    final_acc = evaluate_model(server.global_model, test_loader, DEVICE)
    print(f"\n=== Training Complete ===")
    print(f"Final accuracy: {final_acc:.2f}%")
    print(f"Average round duration: {np.mean(round_durations):.2f}s")
    print(f"Max energy queue: {max(energy_queues):.2f}")

    # Plot results
    plt.figure(figsize=(15, 12))  # Increase figure size for 6 plots

# Accuracy plot
    plt.subplot(321)
    eval_rounds = [5*i for i in range(len(accuracies))]
    plt.plot(eval_rounds, accuracies, 'o-')
    plt.title("Test Accuracy")
    plt.xlabel("Communication Rounds")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)

    # Client selection
    plt.subplot(322)
    selected_counts = [len(s) for s in server.selected_history]
    plt.plot(selected_counts)
    plt.title("Selected Clients per Round")
    plt.xlabel("Rounds")
    plt.ylabel("Number of Clients")
    plt.grid(True)

    # Energy queues
    plt.subplot(323)
    plt.plot(energy_queues)
    plt.title("Max Energy Queue Value")
    plt.xlabel("Rounds")
    plt.ylabel("Queue Value")
    plt.grid(True)

    # Round durations
    plt.subplot(324)
    plt.plot(round_durations)
    plt.title("Round Duration")
    plt.xlabel("Rounds")
    plt.ylabel("Time (s)")
    plt.grid(True)

    # Average staleness
    plt.subplot(325)
    plt.plot(avg_staleness_per_round, 'b-')
    plt.title("Average Client Staleness")
    plt.xlabel("Rounds")
    plt.ylabel("Staleness (rounds)")
    plt.grid(True)

    plt.tight_layout(pad=3.0)  # Add padding between subplots
    plt.savefig("semi_async_ota_fl_results.png", dpi=300)  # Higher resolution
    plt.show()

if __name__ == "__main__":
    main()