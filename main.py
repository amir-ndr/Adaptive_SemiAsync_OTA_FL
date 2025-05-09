import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from model import CNNMnist
from dataloader import load_mnist, partition_mnist_noniid
from client import Client
from server import Server
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def plot_results(acc_history, energy_history):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(acc_history, 'b-o')
    plt.title("Test Accuracy")
    plt.xlabel("Round"), plt.ylabel("Accuracy (%)")
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(np.cumsum(energy_history), 'r-s')
    plt.title("Cumulative Energy")
    plt.xlabel("Round"), plt.ylabel("Energy (J)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    # Configuration
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_CLIENTS = 10
    SHARDS_PER_CLIENT = 2
    TOTAL_ROUNDS = 20
    EMAX = 1e5
    LYAPUNOV_V = 1.0
    GLOBAL_LR = 0.01
    NOISE_POWER = 0.1
    BATCH_SIZE = 32

    # Load data
    train_dataset, test_dataset = load_mnist()
    client_data_map = partition_mnist_noniid(train_dataset, NUM_CLIENTS, SHARDS_PER_CLIENT)

    # Initialize clients
    clients = []
    for cid in range(NUM_CLIENTS):
        dataloader = DataLoader(
            Subset(train_dataset, client_data_map[cid]),
            batch_size=BATCH_SIZE,
            shuffle=True,
            pin_memory=True
        )
        clients.append(Client(
            client_id=cid,
            model_fn=CNNMnist,
            dataset=train_dataset,
            dataloader=dataloader,
            device=DEVICE,
            local_lr=GLOBAL_LR
        ))

    # Initialize server
    server = Server(
        model=CNNMnist().to(DEVICE),
        num_clients=NUM_CLIENTS,
        total_rounds=TOTAL_ROUNDS,
        E_max=EMAX,
        V=LYAPUNOV_V,
        global_lr=GLOBAL_LR,
        noise_power=NOISE_POWER,
        device=DEVICE
    )
    server.register_clients(clients)

    # Start clients
    for client in clients:
        client.start()
    time.sleep(2)  # Warmup

    # Prepare evaluation
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    acc_history = []
    energy_history = []

    # Training loop
    try:
        for t in range(TOTAL_ROUNDS):
            server.run_round(t)
            
            # Evaluate and track metrics
            acc = server.evaluate(test_loader)
            energy_consumed = (server.energy_per_round * (t+1)) + server.energy_queue
            
            acc_history.append(acc)
            energy_history.append(energy_consumed)
            
            staleness = list(server.client_staleness.values())
            logging.info(
                f"Round {t+1} | "
                f"Accuracy: {acc:.2f}% | "
                f"Energy: {energy_consumed:.2f}J | "
                f"Avg Staleness: {np.mean(staleness):.2f}"
            )

    except KeyboardInterrupt:
        logging.warning("Training interrupted!")
    finally:
        server.shutdown()
        if acc_history and energy_history:
            plot_results(acc_history, energy_history)

if __name__ == "__main__":
    main()