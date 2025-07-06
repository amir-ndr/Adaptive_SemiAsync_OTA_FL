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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("cotaf_debug.log"),
        logging.StreamHandler()
    ]
)

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

def log_model_stats(model, round_idx, prefix=""):
    """Log detailed model statistics"""
    logging.info(f"{prefix} Model Stats (Round {round_idx}):")
    for name, param in model.named_parameters():
        logging.info(f"  {name}: mean={param.data.mean().item():.6f}, "
                     f"std={param.data.std().item():.6f}, "
                     f"min={param.data.min().item():.6f}, "
                     f"max={param.data.max().item():.6f}")

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

def run_cotaf_experiment(clients, NUM_ROUNDS, BATCH_SIZE, DEVICE):
    # Initialize global model
    train_dataset, test_data = load_mnist()
    global_model = CNNMnist().to(DEVICE)
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False)
    
    # Log initial model stats
    log_model_stats(global_model, 0, "Initial Global")
    
    # Initialize COTAF server
    server = COTAFServer(
        global_model=global_model,
        clients=clients,
        P_max=0.5,          # Transmission power constraint
        noise_var=0.001,    # Channel noise variance
        H_local=1,          # Local steps per round
        device=DEVICE
    )

    # Initialize metrics
    metrics = {
        'accuracies': [],
        'evaluation_rounds': [],
        'losses': [],
        'round_times': [],
        'total_energy': [],
        'alpha_values': [],
        'divergence': [],
        'param_stats': []  # Track parameter statistics
    }
    
    # Initial evaluation
    initial_acc = evaluate_model(global_model, test_loader, DEVICE)
    metrics['accuracies'].append(initial_acc)
    metrics['evaluation_rounds'].append(0)
    logging.info(f"Initial model accuracy: {initial_acc:.2f}%")
    
    logging.info("\n=== Starting COTAF Experiment ===")
    cumulative_time = 0
    learning_rate = 0.01  # Reduced from 0.05
    
    for round_idx in range(NUM_ROUNDS):
        round_start = time.time()
        logging.info(f"\n=== Round {round_idx+1}/{NUM_ROUNDS} ===")
        
        # Log global model stats before broadcast
        log_model_stats(server.global_model, round_idx+1, "Pre-Broadcast Global")
        
        # 1. Broadcast global model
        server.broadcast_model()
        
        # 2. Simulate fading channel (optional)
        use_fading = False  # Disable fading for stability debugging
        if use_fading:
            fading_coeffs = [complex(np.random.randn(), np.random.randn()) 
                            for _ in range(len(clients))]
            server.apply_fading(fading_coeffs)
            logging.info("Applied fading coefficients")
        
        # 3. Local training with gradient monitoring
        client_losses = []
        max_gradients = []
        for client_idx, client in enumerate(clients):
            # Log client model stats before training
            if client_idx == 0:  # Only log first client for brevity
                log_model_stats(client.local_model, round_idx+1, f"Client {client_idx} Pre-Training")
            
            loss = client.local_train()
            client_losses.append(loss)
            
            # Log gradients after training
            max_grad = 0.0
            for param in client.local_model.parameters():
                if param.grad is not None:
                    grad_norm = param.grad.data.norm(2).item()
                    if grad_norm > max_grad:
                        max_grad = grad_norm
            max_gradients.append(max_grad)
            logging.info(f"Client {client_idx}: Max gradient: {max_grad:.6f}")
        
        avg_loss = np.mean(client_losses)
        metrics['losses'].append(avg_loss)
        logging.info(f"Average client loss: {avg_loss:.4f}, Max gradients: {max(max_gradients):.6f}")
        
        # 4. COTAF aggregation
        logging.info("Starting aggregation")
        new_state = server.aggregate()
        
        # Log aggregated state before updating
        temp_model = CNNMnist().to(DEVICE)
        temp_model.load_state_dict(new_state)
        log_model_stats(temp_model, round_idx+1, "Aggregated")
        
        server.update_model(new_state)
        
        # 5. Record metrics
        round_time = time.time() - round_start
        cumulative_time += round_time
        metrics['round_times'].append(round_time)
        
        # Track alpha value
        try:
            # Get updates for alpha calculation
            updates = [client.get_update(0.1) for client in clients]
            valid_updates = [u for u in updates if u is not None]
            logging.info(f"Valid updates: {len(valid_updates)}/{len(updates)}")
            
            if valid_updates:
                alpha_t = server.compute_alpha_t(valid_updates)
                metrics['alpha_values'].append(alpha_t)
            else:
                logging.warning("No valid updates for alpha calculation")
                metrics['alpha_values'].append(0.1)
        except Exception as e:
            logging.error(f"Error computing alpha_t: {str(e)}")
            metrics['alpha_values'].append(0.1)
        
        # Track divergence if available
        if 'divergence_history' in server.energy_tracker and server.energy_tracker['divergence_history']:
            metrics['divergence'].append(server.energy_tracker['divergence_history'][-1])
            logging.info(f"Model divergence: {server.energy_tracker['divergence_history'][-1]:.4f}")
        
        # Track total energy
        if 'per_round_total' in server.energy_tracker and server.energy_tracker['per_round_total']:
            metrics['total_energy'].append(sum(server.energy_tracker['per_round_total']))
            logging.info(f"Round energy: {sum(server.energy_tracker['per_round_total']):.2f}J")
        else:
            metrics['total_energy'].append(0)
            logging.warning("No energy data recorded")
        
        # Record parameter statistics
        metrics['param_stats'].append(model_param_stats(server.global_model))
        
        # Evaluate every 5 rounds
        if (round_idx + 1) % 5 == 0 or round_idx == NUM_ROUNDS - 1:
            acc = evaluate_model(server.global_model, test_loader, DEVICE)
            metrics['accuracies'].append(acc)
            metrics['evaluation_rounds'].append(round_idx+1)
            logging.info(f"Global model accuracy: {acc:.2f}%")
            
            # Log final model stats after evaluation
            log_model_stats(server.global_model, round_idx+1, "Post-Aggregation Global")
        
        logging.info(f"Round duration: {round_time:.4f}s | Cumulative time: {cumulative_time:.2f}s")
    
    # Final metrics
    metrics['final_acc'] = metrics['accuracies'][-1]
    metrics['total_time'] = cumulative_time
    metrics['energy_metrics'] = server.energy_tracker
    return metrics

def plot_cotaf_results(results):
    plt.figure(figsize=(15, 10))
    
    # Accuracy plot
    plt.subplot(2, 2, 1)
    plt.plot(results['evaluation_rounds'], results['accuracies'], 'o-')
    plt.title("Model Accuracy")
    plt.xlabel("Communication Rounds")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)
    
    # Loss plot
    plt.subplot(2, 2, 2)
    plt.plot(results['losses'])
    plt.title("Training Loss")
    plt.xlabel("Communication Rounds")
    plt.ylabel("Average Client Loss")
    plt.grid(True)
    
    # Energy consumption
    if results['total_energy']:
        plt.subplot(2, 2, 3)
        plt.plot(results['total_energy'])
        plt.title("Cumulative Energy Consumption")
        plt.xlabel("Communication Rounds")
        plt.ylabel("Total Energy (J)")
        plt.grid(True)
    
    # Alpha values
    if results['alpha_values']:
        plt.subplot(2, 2, 4)
        plt.plot(results['alpha_values'])
        plt.title("Precoding Factor (αₜ)")
        plt.xlabel("Communication Rounds")
        plt.ylabel("αₜ Value")
        plt.yscale('log')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('cotaf_results.png')
    plt.show()

def plot_energy_results(results):
    if 'energy_metrics' not in results or not results['energy_metrics']:
        logging.warning("No energy metrics available for plotting")
        return
        
    energy = results['energy_metrics']
    plt.figure(figsize=(12, 5))
    
    # Per-round energy
    plt.subplot(1, 2, 1)
    if 'per_round_total' in energy:
        plt.plot(energy['per_round_total'])
        plt.title("Per-Round Energy Consumption")
        plt.xlabel("Round Index")
        plt.ylabel("Energy (J)")
        plt.grid(True)
    else:
        plt.text(0.5, 0.5, "No per-round energy data", ha='center')
    
    # Cumulative per-client energy
    plt.subplot(1, 2, 2)
    if 'cumulative_per_client' in energy:
        client_ids = list(energy['cumulative_per_client'].keys())
        cumulative_energy = [energy['cumulative_per_client'][cid] for cid in client_ids]
        plt.bar(client_ids, cumulative_energy)
        plt.title("Cumulative Energy per Client")
        plt.xlabel("Client ID")
        plt.ylabel("Total Energy (J)")
        plt.grid(True)
    else:
        plt.text(0.5, 0.5, "No per-client energy data", ha='center')
    
    plt.tight_layout()
    plt.savefig('energy_results.png')
    plt.show()

if __name__ == "__main__":
    # Parameters
    NUM_CLIENTS = 10
    NUM_ROUNDS = 50  # Reduced from 100 for debugging
    BATCH_SIZE = 64  # Increased from 32
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    logging.info(f"Using device: {DEVICE}")
    
    # Load and partition data
    train_dataset, test_data = load_mnist()
    
    # Precompute FLOPs for CNNMnist model
    C = calculate_cnnmnist_flops()
    logging.info(f"Precomputed FLOPs per sample: {C / 1e6:.2f}M")
    
    # Partition with 10 shards per client to use all data
    client_data_map = partition_mnist_noniid(
        train_dataset, 
        num_clients=NUM_CLIENTS, 
        # shards_per_client=10
    )
    
    # Create clients with realistic parameters
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
    
    # Initialize optimizers with lower learning rate
    for client in cotaf_clients:
        client.optimizer = torch.optim.Adam(  # Switch to Adam for stability
            client.local_model.parameters(), 
            lr=0.001,
            weight_decay=1e-4
        )
        logging.info(f"Client {client.client_id}: Optimizer: {client.optimizer}")
    
    # Run COTAF experiment
    try:
        results = run_cotaf_experiment(
            clients=cotaf_clients,
            NUM_ROUNDS=NUM_ROUNDS,
            BATCH_SIZE=BATCH_SIZE,
            DEVICE=DEVICE
        )
    except Exception as e:
        logging.critical(f"Experiment failed: {str(e)}", exc_info=True)
        raise
    
    # Plot results
    plot_cotaf_results(results)
    plot_energy_results(results)
    
    # Print final results
    logging.info("\n=== COTAF Final Results ===")
    logging.info(f"Final accuracy: {results['final_acc']:.2f}%")
    logging.info(f"Total training time: {results['total_time']:.2f} seconds")
    logging.info(f"Average round duration: {np.mean(results['round_times']):.4f} seconds")
    logging.info(f"Total rounds: {NUM_ROUNDS}")

    if 'energy_metrics' in results:
        logging.info("\nEnergy Consumption Summary:")
        if 'per_round_total' in results['energy_metrics']:
            total_energy = sum(results['energy_metrics']['per_round_total'])
            logging.info(f"Total system energy: {total_energy:.2f} J")
            logging.info(f"Average per-round energy: {np.mean(results['energy_metrics']['per_round_total']):.2f} J")
        
        if 'cumulative_per_client' in results['energy_metrics']:
            logging.info("\nPer-client cumulative energy:")
            for cid, energy in results['energy_metrics']['cumulative_per_client'].items():
                logging.info(f"Client {cid}: {energy:.2f} J")
    
    # Save parameter stats for analysis
    with open("param_stats.json", "w") as f:
        import json
        json.dump(results['param_stats'], f)