import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from model import CNNMnist
from dataloader import load_mnist, partition_mnist_noniid
from client import Client
from server import Server

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_clients = 10
shards_per_client = 2
T = 20  # total rounds
E = 5   # local steps
V = 1.0
Emax = 1e5

# Load and partition data
train_dataset, test_dataset = load_mnist()
client_data_map = partition_mnist_noniid(train_dataset, num_clients, shards_per_client)

def get_client_dataloader(indices):
    subset = Subset(train_dataset, indices)
    return DataLoader(subset, batch_size=32, shuffle=True)

# Initialize clients
clients = []
for cid in range(num_clients):
    dataloader = get_client_dataloader(client_data_map[cid])
    client = Client(cid, CNNMnist, train_dataset, dataloader, device)
    clients.append(client)

# Initialize server
server = Server(CNNMnist(), num_clients=num_clients, total_rounds=T, E_max=Emax, V=V)

server.register_clients(clients)

# Start client threads
for client in clients:
    client.start()

# Run training
acc_history = []
energy_history = []

for t in range(T):
    print(f"\n[Round {t}] Starting round...")
    server.run_round(t)
    acc = server.evaluate_global_model()
    acc_history.append(acc)
    energy_history.append(server.total_energy)

# Terminate clients
for client in clients:
    client.join(timeout=1)

# Plotting results
rounds = list(range(1, T + 1))
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(rounds, acc_history, marker='o')
plt.title("Test Accuracy vs Rounds")
plt.xlabel("Round")
plt.ylabel("Accuracy (%)")

plt.subplot(1, 2, 2)
plt.plot(rounds, energy_history, marker='x', color='orange')
plt.title("Cumulative Energy vs Rounds")
plt.xlabel("Round")
plt.ylabel("Total Energy Consumption")

plt.tight_layout()
plt.show()
