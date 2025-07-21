import numpy as np
import time
import torch
import copy
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from model import CNNMnist
from server_cotaf import COTAFServer
from client_cotaf import COTAFClient
from dataloader import load_mnist, partition_mnist_noniid
import logging
import csv
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("cotaf_noise_experiment.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def model_param_stats(model, name="Model"):
    """Log statistics about model parameters"""
    stats = {}
    for name, param in model.named_parameters():
        stats[name] = {
            'mean': param.data.mean().item(),
            'std': param.data.std().item(),
            'min': param.data.min().item(),
            'max': param.data.max().item()
        }
    return stats

def calculate_cnnmnist_flops():
    """Calculate FLOPs for one forward+backward pass of CNNMnist"""
    # Conv1: 1x32x5x5 kernel, input (1,28,28), output (32,28,28)
    conv1_flops = 28 * 28 * 32 * (5 * 5 * 1)  # 28*28*32*25
    
    # Conv2: 32x64x5x5 kernel, input (32,14,14), output (64,14,14)
    conv2_flops = 14 * 14 * 64 * (5 * 5 * 32)  # 14*14*64*800
    
    # FC1: 7*7*64 = 3136 → 512
    fc1_flops = 3136 * 512 * 2  # Multiply-add operations
    
    # FC2: 512 → 10
    fc2_flops = 512 * 10 * 2
    
    # ReLU layers (approximately 1 FLOP per element)
    relu1_flops = 28 * 28 * 32
    relu2_flops = 14 * 14 * 64
    relu3_flops = 512
    
    total_flops = (conv1_flops + conv2_flops + fc1_flops + fc2_flops +
                   relu1_flops + relu2_flops + relu3_flops)
    
    # Backward pass requires ~2x forward FLOPs
    return total_flops * 3  # Approximate total FLOPs per sample

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

def run_single_trial(noise_var, num_rounds=100):
    """Run a single COTAF trial with specified noise variance"""
    # Parameters
    NUM_CLIENTS = 10
    BATCH_SIZE = 32
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load and partition data
    train_dataset, test_data = load_mnist()
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False)
    
    # Precompute FLOPs for CNNMnist model
    C = calculate_cnnmnist_flops()
    
    # Partition data
    client_data_map = partition_mnist_noniid(train_dataset, NUM_CLIENTS)
    
    # Create clients
    clients = [
        COTAFClient(
            client_id=cid,
            data_indices=client_data_map[cid],
            model=CNNMnist(),
            fk=np.random.uniform(1e9, 3e9),
            mu_k=1e-27,
            P_max=0.5 + np.random.rand() * 0.5,
            C=C,
            Ak=BATCH_SIZE,
            train_dataset=train_dataset,
            device=DEVICE,
            local_epochs=1
        )
        for cid in range(NUM_CLIENTS)
    ]
    
    # Initialize optimizers
    for client in clients:
        client.optimizer = torch.optim.Adam(
            client.local_model.parameters(), 
            lr=0.001,
            weight_decay=1e-4
        )
    
    # Initialize global model
    global_model = CNNMnist().to(DEVICE)
    
    # Initialize COTAF server with specified noise variance
    server = COTAFServer(
        global_model=global_model,
        clients=clients,
        P_max=0.5,          # Transmission power constraint
        noise_var=noise_var,  # Use specified noise variance
        H_local=1,          # Local steps per round
        device=DEVICE
    )

    # Train for specified rounds
    for round_idx in range(num_rounds):
        # 1. Broadcast global model
        server.broadcast_model()
        
        # 2. Local training
        for client in clients:
            client.local_train()
        
        # 3. COTAF aggregation
        new_state = server.aggregate()
        server.update_model(new_state)
    
    # Final evaluation
    return evaluate_model(server.global_model, test_loader, DEVICE)

def run_noise_experiment():
    """Run noise impact experiment for COTAF"""
    noise_levels = [0.01, 0.05, 0.1, 0.5]
    num_trials = 1
    num_rounds = 300
    results = []
    
    # Create results directory
    os.makedirs("noise_results", exist_ok=True)
    
    for noise in noise_levels:
        logger.info(f"\n=== Starting noise level: {noise} ===")
        accuracies = []
        
        for trial in range(num_trials):
            logger.info(f"Trial {trial+1}/{num_trials}")
            try:
                acc = run_single_trial(noise_var=noise, num_rounds=num_rounds)
                accuracies.append(acc)
                logger.info(f"Trial {trial+1} accuracy: {acc:.2f}%")
            except Exception as e:
                logger.error(f"Trial {trial+1} failed: {str(e)}")
                accuracies.append(0.0)
        
        avg_acc = np.mean(accuracies)
        results.append((noise, avg_acc))
        logger.info(f"Noise {noise}: Average Accuracy = {avg_acc:.2f}%")
    
    # Save results to CSV
    with open("noise_results/cotaf_noise_results.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Noise Level", "Average Accuracy"])
        for noise, acc in results:
            writer.writerow([noise, acc])
    
    # Plot results
    plt.figure(figsize=(10, 6))
    noise_values, acc_values = zip(*results)
    plt.semilogx(noise_values, acc_values, 'o-', linewidth=2)
    plt.title("COTAF: Accuracy vs Noise Level")
    plt.xlabel("Noise Level (σ²)")
    plt.ylabel("Final Accuracy (%)")
    plt.grid(True, which="both", ls="--")
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig("noise_results/cotaf_noise_vs_accuracy.png", dpi=300)
    
    # Print summary
    print("\n=== COTAF Noise Experiment Results ===")
    print("Noise Level | Avg Accuracy")
    for noise, acc in results:
        print(f"{noise:.4f}     | {acc:.2f}%")
    
    return results

if __name__ == "__main__":
    # Run the noise impact experiment
    noise_results = run_noise_experiment()