import numpy as np
import time
import torch
import copy
import matplotlib.pyplot as plt
import pandas as pd
import os
from torch.utils.data import DataLoader, Subset
from model import CNNMnist
from server_cotaf import COTAFServer
from client_cotaf import COTAFClient
from main_cotaf import run_cotaf_experiment, evaluate_model, calculate_cnnmnist_flops
from dataloader import load_mnist, partition_mnist_dirichlet
import logging
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("cotaf_debug.log"),
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

def run_multiple_experiments(num_runs=10, min_accuracy=75):
    """Run multiple COTAF experiments and aggregate results"""
    # Configuration
    NUM_CLIENTS = 10
    NUM_ROUNDS = 300
    BATCH_SIZE = 32
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Prepare for multiple runs
    successful_runs = []
    run_count = 0
    total_runs = 0
    
    # Create results directory
    os.makedirs("cotaf_results100", exist_ok=True)
    
    while len(successful_runs) < num_runs:
        total_runs += 1
        logger.info(f"Starting experiment run {total_runs}")
        
        # Load and partition data
        train_dataset, test_data = load_mnist()
        client_data_map = partition_mnist_dirichlet(train_dataset, NUM_CLIENTS, alpha=0.2)
        
        # Precompute FLOPs for CNNMnist model
        C = calculate_cnnmnist_flops()
        
        # Create clients
        cotaf_clients = [
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
        for client in cotaf_clients:
            client.optimizer = torch.optim.Adam(
                client.local_model.parameters(), 
                lr=0.001,
                weight_decay=1e-4
            )
        
        # Run experiment
        try:
            results = run_cotaf_experiment(
                clients=cotaf_clients,
                NUM_ROUNDS=NUM_ROUNDS,
                BATCH_SIZE=BATCH_SIZE,
                DEVICE=DEVICE
            )
            
            # Check accuracy
            final_acc = results['final_acc']
            if final_acc >= min_accuracy:
                successful_runs.append(results)
                logger.info(f"Run {total_runs} successful! Accuracy: {final_acc:.2f}%")
            else:
                logger.info(f"Run {total_runs} discarded. Accuracy: {final_acc:.2f}% < {min_accuracy}%")
                
        except Exception as e:
            logger.error(f"Experiment failed: {str(e)}")
            continue
    
    logger.info(f"Completed {total_runs} runs, {len(successful_runs)} successful")
    
    # Aggregate results
    aggregated = {
        'accuracies': np.mean([run['accuracies'] for run in successful_runs], axis=0),
        'evaluation_rounds': successful_runs[0]['evaluation_rounds'],
        'losses': np.mean([run['losses'] for run in successful_runs], axis=0),
        'total_energy': np.mean([run['total_energy'] for run in successful_runs], axis=0),
        'alpha_values': np.mean([run['alpha_values'] for run in successful_runs], axis=0),
        'final_accuracy': np.mean([run['final_acc'] for run in successful_runs]),
        'per_client_energy': {}
    }
    
    # Save aggregated results to CSV
    # 1. Accuracy results
    accuracy_df = pd.DataFrame({
        'round': aggregated['evaluation_rounds'],
        'accuracy': aggregated['accuracies']
    })
    accuracy_df.to_csv("cotaf_results100/accuracy_results.csv", index=False)
    
    # 2. Energy per round
    energy_df = pd.DataFrame({
        'round': np.arange(1, NUM_ROUNDS+1),
        'energy': aggregated['total_energy']
    })
    energy_df.to_csv("cotaf_results100/energy_per_round.csv", index=False)
    
    # 3. Alpha values
    alpha_df = pd.DataFrame({
        'round': np.arange(1, NUM_ROUNDS+1),
        'alpha': aggregated['alpha_values']
    })
    alpha_df.to_csv("cotaf_results100/alpha_values.csv", index=False)
    
    # 4. Loss values
    loss_df = pd.DataFrame({
        'round': np.arange(1, NUM_ROUNDS+1),
        'loss': aggregated['losses']
    })
    loss_df.to_csv("cotaf_results100/loss_values.csv", index=False)
    
    # 5. Summary report
    with open("cotaf_results100/summary_report.txt", "w") as f:
        f.write(f"COTAF Experiment Summary ({len(successful_runs)} successful runs)\n")
        f.write(f"Total runs executed: {total_runs}\n")
        f.write(f"Average final accuracy: {aggregated['final_accuracy']:.2f}%\n")
        f.write(f"Average total energy: {np.mean(aggregated['total_energy']):.2f} J\n")
    
    return aggregated, successful_runs

def plot_aggregated_results(aggregated):
    """Plot aggregated results from multiple runs"""
    plt.figure(figsize=(15, 12))
    
    # Accuracy plot
    plt.subplot(2, 2, 1)
    plt.plot(aggregated['evaluation_rounds'], aggregated['accuracies'], 'o-')
    plt.title("Model Accuracy (Averaged)")
    plt.xlabel("Communication Rounds")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)
    
    # Loss plot
    plt.subplot(2, 2, 2)
    plt.plot(aggregated['losses'])
    plt.title("Training Loss (Averaged)")
    plt.xlabel("Communication Rounds")
    plt.ylabel("Average Client Loss")
    plt.grid(True)
    
    # Energy consumption
    plt.subplot(2, 2, 3)
    plt.plot(aggregated['total_energy'])
    plt.title("Cumulative Energy Consumption (Averaged)")
    plt.xlabel("Communication Rounds")
    plt.ylabel("Total Energy (J)")
    plt.grid(True)
    
    # Alpha values
    plt.subplot(2, 2, 4)
    plt.plot(aggregated['alpha_values'])
    plt.title("Precoding Factor (αₜ) (Averaged)")
    plt.xlabel("Communication Rounds")
    plt.ylabel("αₜ Value")
    plt.yscale('log')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('cotaf_results100/aggregated_results.png')
    plt.show()

def plot_individual_results(successful_runs):
    """Plot results from individual runs for comparison"""
    plt.figure(figsize=(15, 10))
    
    # Accuracy comparison
    plt.subplot(2, 2, 1)
    for i, run in enumerate(successful_runs):
        plt.plot(run['evaluation_rounds'], run['accuracies'], label=f'Run {i+1}')
    plt.title("Accuracy Comparison")
    plt.xlabel("Communication Rounds")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)
    
    # Energy comparison
    plt.subplot(2, 2, 2)
    for i, run in enumerate(successful_runs):
        plt.plot(run['total_energy'], label=f'Run {i+1}')
    plt.title("Energy Consumption Comparison")
    plt.xlabel("Communication Rounds")
    plt.ylabel("Total Energy (J)")
    plt.legend()
    plt.grid(True)
    
    # Final accuracy distribution
    plt.subplot(2, 2, 3)
    final_accs = [run['final_acc'] for run in successful_runs]
    plt.hist(final_accs, bins=10)
    plt.title("Final Accuracy Distribution")
    plt.xlabel("Accuracy (%)")
    plt.ylabel("Frequency")
    plt.grid(True)
    
    # Alpha value comparison
    plt.subplot(2, 2, 4)
    for i, run in enumerate(successful_runs):
        plt.plot(run['alpha_values'], label=f'Run {i+1}')
    plt.title("Alpha Value Comparison")
    plt.xlabel("Communication Rounds")
    plt.ylabel("αₜ Value")
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('cotaf_results100/individual_comparison.png')
    plt.show()

if __name__ == "__main__":
    # Run multiple experiments and aggregate results
    aggregated, successful_runs = run_multiple_experiments(num_runs=10, min_accuracy=80)
    
    # Plot results
    plot_aggregated_results(aggregated)
    plot_individual_results(successful_runs)
    
    # Print summary
    logger.info("\n=== COTAF Experiment Summary ===")
    logger.info(f"Total successful runs: {len(successful_runs)}")
    logger.info(f"Average final accuracy: {aggregated['final_accuracy']:.2f}%")
    logger.info(f"Average total energy: {np.mean(aggregated['total_energy']):.2f} J")
    
    # Save all results
    with open("cotaf_results100/all_runs.pkl", "wb") as f:
        import pickle
        pickle.dump({
            'aggregated': aggregated,
            'individual_runs': successful_runs
        }, f)