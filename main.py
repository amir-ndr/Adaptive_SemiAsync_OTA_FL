import torch
import numpy as np
import time
from torch.utils.data import DataLoader
from client import Client
from server import Server
from model import CNNMnist
from dataloader import load_mnist, partition_mnist_noniid
import matplotlib.pyplot as plt

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
            P_max=1.0,                       # Max transmit power
            C=1e6,                           # FLOPs per sample
            Ak=BATCH_SIZE,                   # Batch size
            train_dataset=train_dataset,
            device=DEVICE,
            local_epochs=LOCAL_EPOCHS
        )
        clients.append(client)
        print(f"Client {cid}: {len(client_data_map[cid])} samples | "
              f"Comp time: {client.dt_k:.4f}s")
    
    # Initialize server
    global_model = CNNMnist().to(DEVICE)
    server = Server(
        global_model=global_model,
        clients=clients,
        V=10.0,               # Lyapunov parameter
        sigma_n=0.1,           # Noise std
        tau_cm=0.05,           # Comm latency
        T_max=300,             # Time budget (s)
        E_max=10.0,            # Energy budget
        T_total_rounds=NUM_ROUNDS,
        device=DEVICE
    )
    
    # Training loop
    accuracies = []
    round_durations = []
    energy_queues = []
    D_t_prev = 0  # Previous round duration

    for round_idx in range(NUM_ROUNDS):
        round_start = time.time()
        print(f"\n=== Round {round_idx+1}/{NUM_ROUNDS} ===")
        
        # 1. Update computation states with previous round duration
        for client in clients:
            client.update_computation_time(D_t_prev)
        
        # 2. Select clients and broadcast current model
        selected, power_alloc = server.select_clients()
        print(f"Selected {len(selected)} clients: {[c.client_id for c in selected]}")
        server.broadcast_model(selected)
        
        # 3. Compute gradients on selected clients
        for client in selected:
            start_comp = time.time()
            client.compute_gradient()
            comp_time = time.time() - start_comp
            print(f"  Client {client.client_id}: "
                  f"Grad norm={client.gradient_norm:.4f}, "
                  f"Comp time={comp_time:.2f}s")
        
        # 4. Aggregate and update global model
        if selected:
            D_t = max(c.dt_k for c in selected) + server.tau_cm
            aggregated = server.aggregate(selected, power_alloc)
            server.update_model(aggregated)
        else:
            D_t = server.tau_cm
            print("No clients selected - communication only round")
        
        # 5. Update queues and client states
        server.update_queues(selected, power_alloc, D_t)
        D_t_prev = D_t  # Store for next round
        round_durations.append(D_t)
        
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
    plt.figure(figsize=(12, 8))
    
    # Accuracy plot
    plt.subplot(221)
    eval_rounds = [5*i for i in range(len(accuracies))]
    plt.plot(eval_rounds, accuracies, 'o-')
    plt.title("Test Accuracy")
    plt.xlabel("Communication Rounds")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)
    
    # Client selection
    plt.subplot(222)
    selected_counts = [len(s) for s in server.selected_history]
    plt.plot(selected_counts)
    plt.title("Selected Clients per Round")
    plt.xlabel("Rounds")
    plt.ylabel("Number of Clients")
    plt.grid(True)
    
    # Energy queues
    plt.subplot(223)
    plt.plot(energy_queues)
    plt.title("Max Energy Queue Value")
    plt.xlabel("Rounds")
    plt.ylabel("Queue Value")
    plt.grid(True)
    
    # Round durations
    plt.subplot(224)
    plt.plot(round_durations)
    plt.title("Round Duration")
    plt.xlabel("Rounds")
    plt.ylabel("Time (s)")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("semi_async_ota_fl_results.png")
    plt.show()

if __name__ == "__main__":
    main()