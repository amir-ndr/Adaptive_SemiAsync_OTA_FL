import torch
import numpy as np
import time
import copy
import logging
from torch.utils.data import DataLoader
from client_sync import SyncClient
from server_sync import SyncServer
from model import CNNMnist
from dataloader import load_mnist, partition_mnist_noniid
import matplotlib.pyplot as plt
import collections

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
            logging.FileHandler("sync_ota_federated_learning.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting Synchronous OTA-FEEL Simulation")

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
            indices = [0]  # Add dummy index
            logger.warning(f"Client {cid} has no data! Adding dummy sample")
        
        client = SyncClient(
            client_id=cid,
            data_indices=indices,
            model=CNNMnist(),
            fk=np.random.uniform(1e9, 3e9),    # 1-3 GHz CPU
            mu_k=1e-27,                        # Energy coefficient
            P_max=3.0,                         # Max transmit power (not used directly in sync)
            C=1e6,                             # FLOPs per sample
            Ak=BATCH_SIZE,                     # Batch size
            train_dataset=train_dataset,
            device=DEVICE,
            local_epochs=LOCAL_EPOCHS
        )
        clients.append(client)
        print(f"Client {cid}: {len(client_data_map[cid])} samples | "
              f"CPU: {client.fk/1e9:.2f} GHz")
    
    # Set energy budgets (Joules)
    E_max_dict = {cid: np.random.uniform(30, 50) for cid in range(NUM_CLIENTS)}
    print("\nClient Energy Budgets:")
    for cid, budget in E_max_dict.items():
        print(f"  Client {cid}: {budget:.2f} J")

    # Initialize server
    global_model = CNNMnist().to(DEVICE)
    server = SyncServer(
    global_model=global_model,
    clients=clients,
    gamma0=5.0,           # Reduced target SNR
    G=1.0,                # Realistic gradient bound
    l=0.01,               # Tighter smoothness constant
    V=100.0,              # Increased Lyapunov parameter
    sigma_n=0.01,         # Reduced noise std
    T_total_rounds=NUM_ROUNDS,
    E_max=E_max_dict,
    device=DEVICE
)

    
    # Training loop
    accuracies = []
    round_durations = []
    client_selection_counts = {cid: 0 for cid in range(NUM_CLIENTS)}
    avg_grad_norms = []
    energy_consumption = {cid: [] for cid in range(NUM_CLIENTS)}
    queue_history = []

    for round_idx in range(NUM_ROUNDS):
        round_start = time.time()
        logger.info(f"\n=== Round {round_idx+1}/{NUM_ROUNDS} ===")
        
        # 1. Select devices using energy-aware scheduling
        selected, sigma_t = server.schedule_devices(round_idx)
        selected_ids = [c.client_id for c in selected]
        for cid in selected_ids:
            client_selection_counts[cid] += 1
        
        logger.info(f"Selected {len(selected)} clients: {selected_ids}")
        
        # 2. Broadcast model to selected devices
        server.broadcast_model(selected)
        
        # 3. Compute gradients on selected clients
        comp_times = []
        grad_norms = []
        for client in selected:
            start_comp = time.time()
            client.compute_gradient()
            comp_time = time.time() - start_comp
            comp_times.append(comp_time)
            grad_norms.append(client.gradient_norm)
            logger.info(f"  Client {client.client_id}: "
                        f"Grad norm={client.gradient_norm:.4f}, "
                        f"Comp time={comp_time:.4f}s")
        
        # 4. Aggregate gradients via OTA
        aggregated = server.aggregate(selected, sigma_t)
        
        # 5. Update global model
        if round_idx < 100:
            lr = 0.1
        elif round_idx < 250:
            lr = 0.05
        else:
            lr = 0.01
        with torch.no_grad():
            params = torch.nn.utils.parameters_to_vector(server.global_model.parameters())
            params -= lr * aggregated
            torch.nn.utils.vector_to_parameters(params, server.global_model.parameters())
            logger.info(f"Global model updated with LR={lr:.4f}")
        
        # 6. Collect actual energy consumption and update queues
        actual_energy = {}
        for client in selected:
            energy = client.report_energy(sigma_t)
            actual_energy[client.client_id] = energy
            energy_consumption[client.client_id].append(energy)
            logger.info(f"  Client {client.client_id}: "
                        f"Energy used={energy:.6f} J")
        server.update_queues(selected, actual_energy)
        queue_history.append(copy.deepcopy(server.Q_e))
        
        # 7. Track metrics
        round_time = time.time() - round_start
        round_durations.append(round_time)
        avg_grad_norms.append(np.mean(grad_norms))
        
        # Evaluate every 5 rounds
        if (round_idx + 1) % 5 == 0 or round_idx == 0:
            acc = evaluate_model(server.global_model, test_loader, DEVICE)
            accuracies.append(acc)
            logger.info(f"Global model accuracy: {acc:.2f}%")
        
        logger.info(f"Round duration: {round_time:.2f}s | "
                    f"Avg grad norm: {np.mean(grad_norms):.4f}")

    # Final evaluation
    final_acc = evaluate_model(server.global_model, test_loader, DEVICE)
    accuracies.append(final_acc)
    logger.info(f"\n=== Training Complete ===")
    logger.info(f"Final accuracy: {final_acc:.2f}%")
    
    # Print client selection and energy statistics
    logger.info("\nClient Selection and Energy Statistics:")
    for cid in range(NUM_CLIENTS):
        total_energy = sum(energy_consumption[cid])
        budget = E_max_dict[cid]
        logger.info(f"Client {cid}: Selected {client_selection_counts[cid]} times | "
                    f"Energy: {total_energy:.2f}/{budget:.2f} J | "
                    f"Remaining: {budget - total_energy:.2f} J")

    # Plot results
    plt.figure(figsize=(15, 12))
    
    # Accuracy plot
    plt.subplot(321)
    eval_rounds = [5*i for i in range(len(accuracies))]
    plt.plot(eval_rounds, accuracies, 'o-')
    plt.title("Test Accuracy")
    plt.xlabel("Communication Rounds")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)

    # Client selection distribution
    plt.subplot(322)
    plt.bar(range(NUM_CLIENTS), [client_selection_counts[cid] for cid in range(NUM_CLIENTS)])
    plt.title("Client Selection Distribution")
    plt.xlabel("Client ID")
    plt.ylabel("Times Selected")
    plt.xticks(range(NUM_CLIENTS))
    plt.grid(True)
    
    # Energy consumption per client
    plt.subplot(323)
    for cid in range(NUM_CLIENTS):
        if energy_consumption[cid]:
            plt.plot(energy_consumption[cid], label=f'Client {cid}')
    plt.title("Per-Round Energy Consumption")
    plt.xlabel("Rounds")
    plt.ylabel("Energy (J)")
    plt.legend(loc='upper right', fontsize=8)
    plt.grid(True)
    
    # Virtual queues
    plt.subplot(324)
    for cid in range(NUM_CLIENTS):
        queue_vals = [q[cid] for q in queue_history]
        plt.plot(queue_vals, label=f'Client {cid}')
    plt.title("Virtual Energy Queues")
    plt.xlabel("Rounds")
    plt.ylabel("Queue Value")
    plt.grid(True)
    
    # Average gradient norms
    plt.subplot(325)
    plt.plot(avg_grad_norms)
    plt.title("Average Gradient Norms")
    plt.xlabel("Rounds")
    plt.ylabel("L2 Norm")
    plt.grid(True)
    
    # Round durations
    plt.subplot(326)
    plt.plot(round_durations)
    plt.title("Round Duration")
    plt.xlabel("Rounds")
    plt.ylabel("Time (s)")
    plt.grid(True)

    plt.tight_layout(pad=3.0)
    plt.savefig("sync_ota_federated_learning_results.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()