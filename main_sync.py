# main.py
import torch
import numpy as np
import time
from torch.utils.data import DataLoader
from client_sync import Client
from server_sync import Server
from model import CNNMnist
from dataloader import load_mnist, partition_mnist_noniid
import matplotlib.pyplot as plt
import logging
import collections
import math

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
            logging.FileHandler("sync_ota_fl.log", encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting Synchronous OTA-FL simulation")

    # Parameters
    NUM_CLIENTS = 10
    NUM_ROUNDS = 300
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
        indices = client_data_map[cid]
        if len(indices) == 0:
            indices = [0]  # Add dummy index to prevent errors
            logger.warning(f"Client {cid} has no data! Adding dummy sample")
        
        # Compute per-sample energy constant
        fk = np.random.uniform(1e9, 2e9)  # 1-2 GHz CPU
        mu_k = 1e-27  # Energy coefficient
        C = 1e6  # FLOPs per sample
        
        client = Client(
            client_id=cid,
            data_indices=indices,
            model=CNNMnist(),
            fk=fk,
            mu_k=mu_k,
            P_max=2.0,
            C=C,
            Ak=BATCH_SIZE,
            train_dataset=train_dataset,
            device=DEVICE
        )
        clients.append(client)
        logger.info(f"Client {cid} | Samples: {len(indices)} | CPU: {fk/1e9:.2f} GHz")
    
    # Energy budgets (Joules)
    E_max_dict = {cid: np.random.uniform(25, 38) for cid in range(NUM_CLIENTS)}
    print("Client Energy Budgets:")
    for cid, budget in E_max_dict.items():
        print(f"  Client {cid}: {budget:.2f} J")

    # Initialize server with corrected parameters
    global_model = CNNMnist().to(DEVICE)
    server = Server(
    global_model=global_model,
    clients=clients,
    V=50.0,            # Reduced from 5000
    sigma_n=1.0,       # Increased noise (realistic)
    gamma0=1e-6,       # Reduced SNR target
    G=0.001,           # Reduced gradient variance
    l=0.001,           # Reduced smoothness
    T_total_rounds=NUM_ROUNDS,
    E_max=E_max_dict,
    device=DEVICE
)
    
    # Training metrics
    accuracies = []
    round_durations = []
    max_energy_queues = []
    selected_counts = []
    client_selection_counts = {cid: 0 for cid in range(NUM_CLIENTS)}
    cumulative_energy = {cid: 0.0 for cid in range(NUM_CLIENTS)}
    per_round_energy = []
    evaluation_rounds = []  # Track rounds when evaluation occurred

    logger.info(f"Starting training for {NUM_ROUNDS} rounds")
    
    for round_idx in range(NUM_ROUNDS):
        round_start = time.time()
        logger.info(f"\n=== Round {round_idx+1}/{NUM_ROUNDS} ===")
        
        # 1. Device selection
        selected = server.select_devices()
        selected_ids = [c.client_id for c in selected]
        selected_counts.append(len(selected))
        
        # Update selection counts
        for cid in selected_ids:
            client_selection_counts[cid] += 1
        
        logger.info(f"Selected {len(selected)} clients: {selected_ids}")
        
        # 2. Broadcast model and σₜ to selected clients
        server.broadcast_model(selected, server.sigma_t)
        logger.info(f"Broadcast model with σₜ={server.sigma_t:.4f}")
        
        # 3. Compute gradients and aggregate
        comp_start = time.time()
        aggregated_update = server.aggregate_gradients(selected)
        comp_time = time.time() - comp_start
        
        # 4. Update global model
        server.update_model(aggregated_update)
        
        # 5. Update virtual queues
        round_energy = server.update_queues(selected)
        per_round_energy.append(round_energy)
        
        # Update cumulative energy
        for client in selected:
            cid = client.client_id
            actual_energy = client.compute_actual_energy(client.sigma_t)
            cumulative_energy[cid] += actual_energy
            
            # Check energy constraint violation
            if cumulative_energy[cid] > E_max_dict[cid]:
                logger.warning(f"Client {cid} energy violation: "
                              f"{cumulative_energy[cid]:.2f}/{E_max_dict[cid]:.2f} J")
        
        # 6. Record round duration
        round_duration = comp_time + server.tau_cm
        round_durations.append(round_duration)
        
        # 7. Track max energy queue
        max_q = max(server.Q_e.values()) if server.Q_e else 0
        max_energy_queues.append(max_q)
        
        # 8. Evaluate model every 5 rounds
        if (round_idx + 1) % 5 == 0 or round_idx == 0:
            acc = evaluate_model(server.global_model, test_loader, DEVICE)
            accuracies.append(acc)
            evaluation_rounds.append(round_idx)
            logger.info(f"Global accuracy: {acc:.2f}%")
        
        # Log metrics
        round_time = time.time() - round_start
        logger.info(f"Round duration: {round_duration:.2f}s | "
                    f"Wall time: {round_time:.2f}s | "
                    f"Max energy queue: {max_q:.2f} | "
                    f"Round energy: {round_energy:.4e} J")

    # Final evaluation
    final_acc = evaluate_model(server.global_model, test_loader, DEVICE)
    accuracies.append(final_acc)
    evaluation_rounds.append(NUM_ROUNDS)
    logger.info(f"\n=== Training Complete ===")
    logger.info(f"Final accuracy: {final_acc:.2f}%")
    
    # Print energy usage
    logger.info("\nEnergy Usage Summary:")
    for cid in range(NUM_CLIENTS):
        budget = E_max_dict[cid]
        consumed = cumulative_energy[cid]
        logger.info(f"Client {cid}: {consumed:.2f}/{budget:.2f} J "
                    f"({consumed/budget:.1%}) | "
                    f"Selected {client_selection_counts[cid]} times")
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Accuracy plot
    plt.subplot(221)
    plt.plot(evaluation_rounds, accuracies, 'o-')
    plt.title("Test Accuracy")
    plt.xlabel("Communication Rounds")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)

    # Client selection
    plt.subplot(222)
    plt.plot(selected_counts)
    plt.title("Selected Clients per Round")
    plt.xlabel("Rounds")
    plt.ylabel("Number of Clients")
    plt.grid(True)

    # Energy queues
    plt.subplot(223)
    plt.plot(max_energy_queues)
    plt.title("Max Energy Queue Value")
    plt.xlabel("Rounds")
    plt.ylabel("Queue Value")
    plt.grid(True)

    # Energy consumption
    plt.subplot(224)
    plt.plot(per_round_energy)
    plt.title("Per-Round Energy Consumption")
    plt.xlabel("Rounds")
    plt.ylabel("Energy (J)")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("sync_ota_fl_results.png", dpi=300)
    plt.show()
    
    # Additional plots
    plt.figure(figsize=(12, 8))
    
    # Client participation heatmap
    participation_matrix = np.zeros((NUM_CLIENTS, NUM_ROUNDS))
    for round_idx, selected in enumerate(server.selected_history):
        for cid in selected:
            participation_matrix[cid, round_idx] = 1
    
    plt.subplot(211)
    plt.imshow(participation_matrix, cmap='Blues', aspect='auto')
    plt.title("Client Participation Over Rounds")
    plt.ylabel("Client ID")
    plt.xlabel("Round")
    plt.colorbar(label="Participation (1=selected)")
    
    # Cumulative energy usage
    plt.subplot(212)
    for cid in range(NUM_CLIENTS):
        energy_per_round = np.zeros(NUM_ROUNDS)
        for round_idx in range(NUM_ROUNDS):
            if cid in server.selected_history[round_idx]:
                client = next(c for c in clients if c.client_id == cid)
                energy_per_round[round_idx] = client.compute_actual_energy(
                    client.sigma_t if hasattr(client, 'sigma_t') else 0
                )
        cumulative = np.cumsum(energy_per_round)
        plt.plot(cumulative, label=f'Client {cid}')
    
    plt.title("Cumulative Energy Consumption")
    plt.xlabel("Rounds")
    plt.ylabel("Energy (J)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("client_energy_usage.png", dpi=300)
    plt.show()
    
    # Save model
    torch.save(server.global_model.state_dict(), "sync_ota_fl_model.pth")

if __name__ == "__main__":
    main()