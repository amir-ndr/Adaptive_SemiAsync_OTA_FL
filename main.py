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
            device=DEVICE
        )
        # Initialize as ready
        client.ready = True
        client.dt_k = 0
        clients.append(client)
    
    # Initialize server
    global_model = CNNMnist().to(DEVICE)
    server = Server(
        global_model=global_model,
        clients=clients,
        V=10.0,               # Lyapunov parameter
        sigma_n=0.1,           # Noise std
        tau_cm=0.05,           # Comm latency
        T_max=300,             # Time budget (s)
        E_max=1.0,            # Energy budget
        T_total_rounds=NUM_ROUNDS,
        device=DEVICE
    )
    
    # Training loop
    accuracies = []
    start_time = time.time()
    
    # Initialize previous round duration
    # Initialize previous round duration
    D_t_prev = 0  # For first round initialization

    for round_idx in range(NUM_ROUNDS):
        round_start = time.time()
        print(f"\n=== Round {round_idx+1}/{NUM_ROUNDS} ===")
        
        # 1. Update client states with previous round's duration
        for client in clients:
            # Update computation time (even if not finished)
            # This reduces dt_k and may mark client as ready if computation completes
            client.update_computation(D_t_prev)
        
        # 2. Compute scores and sort clients based on remaining computation time
        for client in clients:
            # Set channel gain for this round
            client.set_channel_gain()
            
            # Compute score ρ_k = |h_k| / √(Q_e * ||g||²)
            # Use last gradient norm if available, else use a default value
            grad_norm = client.gradient_norm if client.gradient_norm > 0 else 1.0
            score = abs(client.h_t_k) / np.sqrt(max(1e-8, server.Q_e[client.client_id]) * max(1e-8, grad_norm))
            client.score = score
        
        # 3. Sort clients by ascending dt_k (lowest remaining time first)
        # and then descending score (highest score first)
        sorted_clients = sorted(clients, key=lambda c: (c.dt_k, -c.score))
        
        # 4. Greedy client selection (Algorithm 1)
        selected = []
        best_cost = float('inf')
        best_power = {}
        
        for client in sorted_clients:
            candidate_set = selected + [client]
            
            # Compute power allocation for candidate set
            power_alloc, c_values = server._compute_power(candidate_set)
            if not power_alloc:
                continue
                
            # Compute round duration (max remaining time in candidate set + communication latency)
            D_temp = max(c.dt_k for c in candidate_set) + server.tau_cm
            
            # Compute drift-plus-penalty cost
            cost = server._compute_cost(candidate_set, power_alloc, D_temp)
            
            # Keep client if cost decreases
            if cost < best_cost:
                selected = candidate_set
                best_cost = cost
                best_power = power_alloc
            else:
                # Stop when cost starts increasing
                break
        
        print(f"Selected {len(selected)} clients: {[c.client_id for c in selected]}")
        
        # 5. Compute gradients for selected clients
        for client in selected:
            # Use current model to compute gradient (even if computation not complete)
            client.compute_gradient(global_model)
        
        # 6. Determine round duration
        if selected:
            D_t = max(c.dt_k for c in selected) + server.tau_cm
            aggregated = server.aggregate(selected, best_power)
        else:
            D_t = server.tau_cm  # Minimal time if no selection
        
        # 7. Update queues and client states
        server.update_queues(selected, best_power, D_t)
        
        # 8. Store for next round's client update
        D_t_prev = D_t
        
        # 9. Evaluate periodically
        if (round_idx + 1) % 5 == 0 or round_idx == 0:
            acc = evaluate_model(global_model, test_loader, DEVICE)
            accuracies.append(acc)
            print(f"Accuracy: {acc:.2f}%")
        
        # 10. Log
        round_time = time.time() - round_start
        print(f"Round Duration: {D_t:.4f}s | "
            f"Wall Clock: {round_time:.2f}s | "
            f"Max Energy Queue: {max(server.Q_e.values()):.2f} | "
            f"Time Queue: {server.Q_time:.2f}")
    
    # Plot results
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.plot(accuracies)
    plt.title("Test Accuracy")
    
    plt.subplot(132)
    selected_counts = [len(s) for s in server.selected_history]
    plt.plot(selected_counts)
    plt.title("Selected Clients per Round")
    
    plt.subplot(133)
    queue_values = [max(q.values()) for q in server.queue_history]
    plt.plot(queue_values)
    plt.title("Max Energy Queue")
    
    plt.tight_layout()
    plt.savefig("results.png")
    plt.show()

if __name__ == "__main__":
    main()