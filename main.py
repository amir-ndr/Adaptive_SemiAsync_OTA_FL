import torch
import numpy as np
import random
import time
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from client import Client
from server import Server
from dataloader import load_mnist, partition_mnist_noniid
from model import CNNMnist
import threading

# ======= Settings =======
num_clients = 10
shards_per_client = 2
E_local = 1
T_global = 200  # Total global rounds
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

V = 1e4             # Lyapunov control parameter
P_max = 1e6         # Total energy budget
tau_cm = 0.01       # OTA communication latency
eta_local = 0.01    # Local SGD learning rate
eta_global = 0.1    # Global model update learning rate

# ======= Step 1: Load MNIST and partition non-IID =======
train_dataset, test_dataset = load_mnist()
client_data_map = partition_mnist_noniid(train_dataset, num_clients, shards_per_client)

# ======= Step 2: Initialize Clients =======
clients = []
for client_id in range(num_clients):
    model = CNNMnist()
    data_indices = client_data_map[client_id]
    train_subset = torch.utils.data.Subset(train_dataset, data_indices)
    compute_power = np.random.uniform(5e7, 1e8)
    channel = torch.randn(1, dtype=torch.cfloat, device=device)
    client = Client(
        client_id=client_id,
        data_indices=data_indices,
        train_dataset=train_subset,
        test_dataset=test_dataset,
        model=model,
        device=device,
        compute_power=compute_power,
        channel=channel,
        eta=eta_local,
        E_local=E_local,
    )
    client.start()
    clients.append(client)

# ======= Step 3: Initialize Server =======
global_model = CNNMnist()
server = Server(global_model, device, num_clients, V, P_max, T_global)

# ======= Step 4: Tracking =======
accuracy_list = []
loss_list = []
q_queue_list = []
best_accuracy = 0.0
best_model_weights = None

print("Broadcasting initial global model to all clients...")
for client in clients:
    client.receive_global_model(server.global_model_flat.clone())
print("Initial global model broadcast done.")

# ======= Helper: Evaluation =======
def evaluate_global_model(server_model, test_loader, device):
    server_model.eval()
    correct = 0
    total = 0
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0.0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            outputs = server_model(x)
            loss = criterion(outputs, y)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    accuracy = 100 * correct / total
    avg_loss = total_loss / len(test_loader)
    return accuracy, avg_loss

# ======= Step 5: Training Loop =======
for t in range(T_global):
    print(f"\n=== Global Round {t} ===")

    # Wait until at least 2 clients are ready
    while True:
        ready_clients = [c for c in clients if c.bk == 1]
        if len(ready_clients) >= 2:
            break
        time.sleep(0.1)

    print(f"[Server] {len(ready_clients)} clients are ready.")

    # Select optimal subset of clients
    selected_clients = server.select_clients(ready_clients, tau_cm)
    print(f"[Server] Selected {len(selected_clients)} clients for aggregation.")

    # Assign events and trigger selected clients
    event_map = server.assign_events(selected_clients)

    # Wait for all selected clients to finish
    server.wait_for_clients(event_map)

    # Receive OTA aggregated signal
    normalized_gradient = server.receive_ota_signal(selected_clients)

    # Update global model
    server.global_update(normalized_gradient, eta_global)

    # Update energy virtual queue
    server.update_virtual_queue(selected_clients)

    # Broadcast updated model to selected clients only
    for client in selected_clients:
        client.receive_global_model(server.global_model_flat.clone())

    # Update staleness for non-selected clients
    for client in clients:
        if client not in selected_clients:
            client.increment_staleness()

    # Evaluate
    if t % 10 == 0:
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
        acc, loss = evaluate_global_model(server.model, test_loader, device)
        accuracy_list.append(acc)
        loss_list.append(loss)
        print(f"[Evaluation] Round {t}: Accuracy = {acc:.2f}%, Loss = {loss:.4f}")
        if acc > best_accuracy:
            best_accuracy = acc
            best_model_weights = server.flatten_model().clone()

    q_queue_list.append(server.Q)

# ======= Step 6: After Training =======
print("\nTraining complete.")

for client in clients:
    client.stop()
for client in clients:
    client.join()

plt.figure(figsize=(8,5))
plt.plot(range(0, T_global, 10), accuracy_list, marker='o')
plt.xlabel('Global Round')
plt.ylabel('Test Accuracy (%)')
plt.title('Global Test Accuracy over Rounds')
plt.grid(True)
plt.show()

plt.figure(figsize=(8,5))
plt.plot(range(len(q_queue_list)), q_queue_list)
plt.xlabel('Global Round')
plt.ylabel('Virtual Queue Q(t)')
plt.title('Virtual Energy Queue over Rounds')
plt.grid(True)
plt.show()

if best_model_weights is not None:
    torch.save(best_model_weights, 'best_global_model.pth')
    print(f"Best model saved with {best_accuracy:.2f}% accuracy as 'best_global_model.pth'.")
