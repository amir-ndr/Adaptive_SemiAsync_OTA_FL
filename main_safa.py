import torch
import numpy as np
import time
from torch.utils.data import DataLoader
from safa_client import SAFAClient
from safa_server import SAFAServer
from model import CNNMnist
from dataloader import load_mnist, partition_mnist_noniid
import matplotlib.pyplot as plt
import logging
from collections import defaultdict

def main():
    # ===== Enhanced Logging Configuration =====
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("safa_fl_system.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting SAFA Federated Learning Simulation")

    # ========== Experiment Configuration ==========
    NUM_CLIENTS = 10
    NUM_ROUNDS = 300
    BATCH_SIZE = 32
    LOCAL_EPOCHS = 1
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    CRASH_PROB = 0.15
    LAG_TOLERANCE = 3
    SELECT_FRAC = 0.5
    EVAL_INTERVAL = 5
    
    logger.info(f"\n=== Configuration ==="
                f"\nClients: {NUM_CLIENTS}"
                f"\nRounds: {NUM_ROUNDS}"
                f"\nBatch Size: {BATCH_SIZE}"
                f"\nLocal Epochs: {LOCAL_EPOCHS}"
                f"\nDevice: {DEVICE}"
                f"\nCrash Probability: {CRASH_PROB:.0%}"
                f"\nStaleness Tolerance: {LAG_TOLERANCE}"
                f"\nSelection Fraction: {SELECT_FRAC:.0%}"
                f"\nEvaluation Interval: {EVAL_INTERVAL} rounds")

    # ========== Data Preparation ==========
    train_dataset, test_dataset = load_mnist()
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    client_data_map = partition_mnist_noniid(train_dataset, NUM_CLIENTS)

    # ========== Client Initialization ==========
    clients = []
    for cid in range(NUM_CLIENTS):
        indices = client_data_map[cid]
        if len(indices) == 0:
            indices = [0]  # Ensure at least one sample
            logger.warning(f"Client {cid} received dummy data")

        # Hardware diversity parameters
        cpu_freq = np.random.uniform(1e9, 2e9)  # 1-2 GHz
        crash_prob = CRASH_PROB * np.random.uniform(0.8, 1.2)
        
        clients.append(
            SAFAClient(
                client_id=cid,
                data_indices=indices,
                model=CNNMnist(),
                fk=cpu_freq,
                mu_k=1e-27,                      # Realistic energy coefficient
                P_max=0.2,                       
                C=1e6,                           # FLOPs per sample
                Ak=BATCH_SIZE,                   
                train_dataset=train_dataset,
                device=DEVICE,
                local_epochs=LOCAL_EPOCHS,
                crash_prob=crash_prob
            )
        )
        logger.info(f"Client {cid} initialized | "
                   f"Samples: {len(indices)} | "
                   f"CPU: {cpu_freq/1e9:.2f}GHz | "
                   f"Crash Prob: {crash_prob:.1%}")

    # ========== Server Initialization ==========
    global_model = CNNMnist().to(DEVICE)
    server = SAFAServer(
        global_model=global_model,
        clients=clients,
        lag_tolerance=LAG_TOLERANCE,
        select_frac=SELECT_FRAC,
        learning_rate=0.01,  # ADDED: Critical for model updates
        device=DEVICE
    )

    # ========== Enhanced Metrics Tracking ==========
    metrics = {
        'round_durations': [],
        'energy_consumption': [],
        'crashed_counts': [],
        'effective_updates': [],
        'selected_counts': [],
        'client_selections': defaultdict(int),
        'accuracies': [],
        'staleness': [],
        'comm_ratio': [],
        'rounds': list(range(NUM_ROUNDS))
    }

    # ========== Training Loop ==========
    logger.info("\n=== Starting Training ===")
    for round_idx in range(NUM_ROUNDS):
        round_start = time.time()
        logger.info(f"\n--- Round {round_idx+1}/{NUM_ROUNDS} ---")
        
        # Run federated round
        eval_this_round = (round_idx % EVAL_INTERVAL == 0) or (round_idx == NUM_ROUNDS - 1)
        round_duration = server.run_round(
            test_loader=test_loader if eval_this_round else None
        )

        # Record metrics
        metrics['round_durations'].append(round_duration)
        
        # Energy metrics - FIXED indexing
        if server.energy_history and round_idx < len(server.energy_history):
            round_energy = server.energy_history[round_idx]
            total_energy = sum(e['total'] for e in round_energy) if round_energy else 0
            comm_energy = sum(e['communication'] for e in round_energy) if round_energy else 0
            
            metrics['energy_consumption'].append(total_energy)
            metrics['comm_ratio'].append(comm_energy / total_energy if total_energy > 0 else 0)
        else:
            metrics['energy_consumption'].append(0)
            metrics['comm_ratio'].append(0)
            
        # Client status metrics - FIXED
        metrics['crashed_counts'].append(
            sum(1 for c in clients if c.state == "crashed")
        )
        
        # Selection metrics - FIXED indexing
        if server.selection_history and round_idx < len(server.selection_history):
            selected_ids = server.selection_history[round_idx]
            metrics['selected_counts'].append(len(selected_ids))
            
            # Update client selection counts
            for cid in selected_ids:
                metrics['client_selections'][cid] += 1
        else:
            metrics['selected_counts'].append(0)
        
        # Effective updates - FIXED
        metrics['effective_updates'].append(
            len(server.energy_history[round_idx]) if server.energy_history and round_idx < len(server.energy_history) else 0
        )
        
        # Staleness - FIXED indexing
        if server.staleness_history and round_idx < len(server.staleness_history):
            metrics['staleness'].append(server.staleness_history[round_idx])
        else:
            metrics['staleness'].append(0)

        # Accuracy tracking - FIXED
        if eval_this_round and server.accuracy_history:
            metrics['accuracies'].append(server.accuracy_history[-1])

        # Periodic logging
        if (round_idx + 1) % 10 == 0 or round_idx == 0:
            logger.info(
                f"Round {round_idx+1:03d}/{NUM_ROUNDS} | "
                f"Duration: {round_duration:.2f}s | "
                f"Selected: {metrics['selected_counts'][-1]} | "
                f"Crashed: {metrics['crashed_counts'][-1]} | "
                f"Effective: {metrics['effective_updates'][-1]} | "
                f"Energy: {metrics['energy_consumption'][-1]:.2f}J | "
                f"Staleness: {metrics['staleness'][-1]:.2f}"
            )

    # ========== Final Evaluation ==========
    final_acc = server.evaluate(test_loader)
    metrics['accuracies'].append(final_acc)
    
    logger.info(f"\n=== Training Complete ==="
                f"\nFinal Accuracy: {final_acc:.2f}%"
                f"\nTotal Energy: {sum(metrics['energy_consumption']):.2f}J"
                f"\nAvg Round Time: {np.mean(metrics['round_durations']):.2f}s"
                f"\nAvg Staleness: {np.mean(metrics['staleness']):.2f}")

    # ========== Enhanced Visualization ==========
    plt.figure(figsize=(18, 15))
    
    # 1. Accuracy Progress
    plt.subplot(321)
    eval_rounds = [EVAL_INTERVAL*i for i in range(len(metrics['accuracies'])-1)] + [NUM_ROUNDS]
    plt.plot(eval_rounds, metrics['accuracies'], 'o-')
    plt.title("Test Accuracy Progress")
    plt.xlabel("Communication Rounds")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)

    # 2. Client Participation
    plt.subplot(322)
    plt.plot(metrics['rounds'], metrics['selected_counts'], label='Selected')
    plt.plot(metrics['rounds'], metrics['effective_updates'], label='Effective')
    plt.title("Client Participation per Round")
    plt.xlabel("Rounds")
    plt.ylabel("Number of Clients")
    plt.legend()
    plt.grid(True)

    # 3. Energy Consumption
    plt.subplot(323)
    plt.plot(metrics['rounds'], metrics['energy_consumption'])
    plt.title("Energy Consumption per Round")
    plt.xlabel("Rounds")
    plt.ylabel("Energy (Joules)")
    plt.grid(True)

    # 4. Round Duration
    plt.subplot(324)
    plt.plot(metrics['rounds'], metrics['round_durations'])
    plt.title("Round Duration")
    plt.xlabel("Rounds")
    plt.ylabel("Time (seconds)")
    plt.grid(True)

    # 5. System Health
    plt.subplot(325)
    plt.plot(metrics['rounds'], metrics['staleness'], 'b-', label='Avg Staleness')
    plt.plot(metrics['rounds'], metrics['crashed_counts'], 'r-', label='Crashed Clients')
    plt.title("System Health Metrics")
    plt.xlabel("Rounds")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True)

    # 6. Client Selection Distribution
    plt.subplot(326)
    plt.bar(range(NUM_CLIENTS), [metrics['client_selections'][cid] for cid in range(NUM_CLIENTS)])
    plt.title("Client Selection Distribution")
    plt.xlabel("Client ID")
    plt.ylabel("Times Selected")
    plt.xticks(range(NUM_CLIENTS))
    plt.grid(True)

    plt.suptitle("SAFA Federated Learning Performance Metrics", fontsize=16, y=0.99)
    plt.tight_layout(pad=3.0)
    plt.savefig("safa_fl_results.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Energy Breakdown
    plt.figure(figsize=(12, 6))
    plt.plot(metrics['rounds'], metrics['comm_ratio'])
    plt.title("Communication Energy Ratio per Round")
    plt.xlabel("Rounds")
    plt.ylabel("Comm/Total Energy Ratio")
    plt.grid(True)
    plt.savefig("safa_energy_breakdown.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Client Statistics
    client_stats = server.get_client_stats()
    print("\n=== Client Statistics ===")
    print(f"{'ID':<5}{'Selected':<10}{'Missed':<10}{'Staleness':<12}{'Version':<10}{'State':<12}")
    for cid in range(NUM_CLIENTS):
        stats = client_stats.get(cid, {})
        print(f"{cid:<5}"
              f"{stats.get('selected_count',0):<10}"
              f"{stats.get('rounds_missed',0):<10}"
              f"{stats.get('staleness',0):<12}"
              f"{stats.get('version',0):<10}"
              f"{stats.get('state','unknown'):<12}")

if __name__ == "__main__":
    main()