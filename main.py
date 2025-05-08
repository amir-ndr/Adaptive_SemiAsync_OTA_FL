import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from model import CNNMnist
from dataloader import load_mnist, partition_mnist_noniid
from client import Client
from server import Server

def plot_results(acc_history, energy_history):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(acc_history, 'b-o')
    plt.title("Test Accuracy vs Training Rounds")
    plt.xlabel("Round")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(np.cumsum(energy_history), 'r-s')
    plt.title("Cumulative Energy Consumption")
    plt.xlabel("Round")
    plt.ylabel("Energy (Joules)")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            outputs = model(x)
            predicted = outputs.argmax(dim=1)
            correct += (predicted == y).sum().item()
            total += y.size(0)
    return 100.0 * correct / total

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLIENTS = 10
SHARDS_PER_CLIENT = 2
TOTAL_ROUNDS = 20
EMAX = 1e5
LYAPUNOV_V = 1.0
GLOBAL_LR = 0.01
LOCAL_LR = 0.01
BATCH_SIZE = 32

def main():
    # Load and partition data
    train_dataset, test_dataset = load_mnist()
    client_data_map = partition_mnist_noniid(train_dataset, NUM_CLIENTS, SHARDS_PER_CLIENT)

    # Create client dataloaders
    def get_client_dataloader(indices):
        return DataLoader(
            Subset(train_dataset, indices),
            batch_size=BATCH_SIZE,
            shuffle=True,
            pin_memory=True
        )

    # Initialize clients
    clients = []
    for cid in range(NUM_CLIENTS):
        dataloader = get_client_dataloader(client_data_map[cid])
        clients.append(Client(
            client_id=cid,
            model_fn=CNNMnist,
            dataset=train_dataset,
            dataloader=dataloader,
            device=DEVICE,
            local_lr=LOCAL_LR
        ))

    # Initialize server
    server = Server(
        model=CNNMnist().to(DEVICE),
        num_clients=NUM_CLIENTS,
        total_rounds=TOTAL_ROUNDS,
        E_max=EMAX,
        V=LYAPUNOV_V,
        global_lr=GLOBAL_LR,
        communication_latency=1.0,
        device=DEVICE
    )
    server.register_clients(clients)

    # Start client threads
    for client in clients:
        client.start()

    # Prepare evaluation
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, pin_memory=True)
    acc_history = []
    energy_history = []

    # Training loop
    try:
        for t in range(TOTAL_ROUNDS):
            print(f"\n=== Round {t+1}/{TOTAL_ROUNDS} ===")
            server.run_round(t)
            
            # Evaluate and track metrics
            acc = evaluate_model(server.model, test_loader=test_loader)
            acc_history.append(acc)
            energy_history.append(server.energy_consumption[-1])
            
            print(f"Round {t} | Accuracy: {acc:.2f}% | "
                  f"Cumulative Energy: {sum(energy_history):.2f} J")
            
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    finally:
        # Clean shutdown
        for client in clients:
            client.stop()
        for client in clients:
            client.join()

        # Plot results
        plot_results(acc_history, energy_history)

if __name__ == "__main__":
    main()