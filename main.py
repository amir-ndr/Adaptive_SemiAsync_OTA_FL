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
    LOCAL_EPOCHS = 3
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
        # Initialize with full computation time
        clients.append(client)
        print(f"Client {cid}: Full computation time = {client.dt_k:.4f}s")
    
    # Initialize server
    global_model = CNNMnist().to(DEVICE)
    server = Server(
        global_model=global_model,
        clients=clients,
        V=10.0,               # Lyapunov parameter
        sigma_n=0.1,           # Noise std
        tau_cm=0.05,           # Comm latency (small but non-zero)
        T_max=300,             # Time budget (s)
        E_max=10.0,            # Energy budget
        T_total_rounds=NUM_ROUNDS,
        device=DEVICE
    )
    
    # Training loop
    accuracies = []
    D_t_prev = 0  # Previous round duration

    for round_idx in range(NUM_ROUNDS):
        round_start = time.time()
        print(f"\n=== Round {round_idx+1}/{NUM_ROUNDS} ===")
        
        # 1. Update client computation states with previous round's duration
        for client in clients:
            client.update_computation_time(D_t_prev)
            # Print client state for debugging
            print(f"  Client {client.client_id}: dt_k={client.dt_k:.4f}s, ready={client.ready}")
        
        # 2. Select clients and allocate power
        selected, power_alloc = server.select_clients()
        print(f"Selected {len(selected)} clients: {[c.client_id for c in selected]}")
        
        # 3. Compute gradients for selected clients
        for client in selected:
            # Compute gradient with specified epochs
            client.compute_gradient()
            print(f"  Client {client.client_id} computed gradient, norm={client.gradient_norm:.4f}")
        
        # 4. Determine round duration
        if selected:
            D_t = max(c.dt_k for c in selected) + server.tau_cm
            aggregated = server.aggregate(selected, power_alloc)
            server.update_model(aggregated)
        else:
            D_t = server.tau_cm
        
        print(f"Round duration: {D_t:.4f}s")
        
        # 5. Update queues and client states
        server.update_queues(selected, power_alloc, D_t)
        
        # 6. Store for next round
        D_t_prev = D_t
        
        # 7. Evaluate periodically
        if (round_idx + 1) % 5 == 0 or round_idx == 0:
            acc = evaluate_model(server.global_model, test_loader, DEVICE)
            accuracies.append(acc)
            print(f"Accuracy: {acc:.2f}%")
        
        # 8. Log
        round_time = time.time() - round_start
        print(f"Wall Clock Time: {round_time:.2f}s | "
              f"Max Energy Queue: {max(server.Q_e.values()):.2f} | "
              f"Time Queue: {server.Q_time:.2f}")

    # Plot results (same as before)
    # ... [rest of your plotting code]

    
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