import imp
import torch
import numpy as np
from torch.utils.data import DataLoader
from model import CNNMnist
from dataloader import load_mnist, partition_mnist_noniid
import matplotlib.pyplot as plt
import time
from client import Client

class Server:
    def __init__(self, global_model, clients, V=1.0, sigma_n=0.1, 
                 tau_cm=0.1, T_max=100, E_max=None, device='cpu'):
        self.global_model = global_model.to(device)
        self.clients = clients
        self.device = device
        self.V = V
        self.sigma_n = sigma_n
        self.tau_cm = tau_cm
        self.T_max = T_max
        self.d = self._get_model_dimension()
        
        # Virtual queues
        self.Q_e = {client.client_id: 0.0 for client in clients}
        self.Q_time = 0.0
        self.E_max = E_max or {client.client_id: 1.0 for client in clients}
        
        # History tracking
        self.selected_history = []
        self.queue_history = []
        self.aggregation_times = []
        
    def _get_model_dimension(self):
        return sum(p.numel() for p in self.global_model.parameters())
    
    def select_clients(self, current_time):
        """Select clients using greedy algorithm"""
        candidates = [c for c in self.clients if c.last_gradient is not None]
        
        if not candidates:
            return [], {}
            
        # Compute scores
        for client in candidates:
            client.set_channel_gain()
            score = abs(client.h_t_k) / np.sqrt(max(1e-8, self.Q_e[client.client_id]) * 
                                              max(1e-8, client.gradient_norm))
            client.score = score
            
        # Sort by dt_k (ascending) and score (descending)
        candidates.sort(key=lambda c: (c.dt_k, -c.score))
        
        # Greedy selection
        selected = []
        best_cost = float('inf')
        best_power = {}
        
        for client in candidates:
            temp_selected = selected + [client]
            
            # Compute power allocation
            power_alloc, c_values = self._compute_power(temp_selected)
            if not power_alloc:
                continue
                
            # Compute round duration
            comp_times = [c.dt_k for c in temp_selected]
            D_temp = max(comp_times) + self.tau_cm
            
            # Compute cost
            cost = self._compute_cost(temp_selected, power_alloc, D_temp)
            
            if cost < best_cost:
                selected = temp_selected
                best_cost = cost
                best_power = power_alloc
            else:
                break
                
        return selected, best_power

    def _compute_power(self, selected):
        """Power allocation for selected clients"""
        c_values = {}
        for client in selected:
            c = self.Q_e[client.client_id] * client.gradient_norm**2 / abs(client.h_t_k)**2
            c_values[client.client_id] = c
            
        inv_sqrt_sum = sum(1/np.sqrt(max(1e-8, c)) for c in c_values.values())
        total_power = ((self.V * self.d * self.sigma_n**2) / 
                      (len(selected) * inv_sqrt_sum**2)) ** 0.25
        
        power_alloc = {}
        for client in selected:
            c = c_values[client.client_id]
            power = (1/np.sqrt(c) / inv_sqrt_sum) * total_power
            power_alloc[client.client_id] = min(power, client.P_max)
            
        return power_alloc, c_values

    def _compute_cost(self, selected, power_alloc, D_temp):
        """Compute drift-plus-penalty cost"""
        total_power = sum(power_alloc.values())
        alpha_sum = 0
        for client in selected:
            alpha = power_alloc[client.client_id] / total_power
            alpha_sum += alpha**2
            
        convergence_term = 1.0 * alpha_sum + self.d * self.sigma_n**2 / total_power**2
        energy_cost = sum(
            self.Q_e[c.client_id] * (power_alloc[c.client_id]**2) * 
            (c.gradient_norm**2 / abs(c.h_t_k)**2) for c in selected
        )
        
        return self.V * convergence_term + energy_cost + self.Q_time * D_temp

    def aggregate(self, selected, power_alloc):
        """Perform OTA aggregation"""
        total_power = sum(power_alloc.values())
        aggregated = torch.zeros(self.d, device=self.device)
        
        for client in selected:
            p = power_alloc[client.client_id]
            h = client.h_t_k
            scaled_grad = client.last_gradient * (p * h.conjugate().real / abs(h))
            aggregated += scaled_grad
            
        # Add noise and normalize
        noise = torch.randn(self.d, device=self.device) * self.sigma_n
        return (aggregated + noise) / total_power

    def update_model(self, update):
        """Update global model"""
        with torch.no_grad():
            params = torch.nn.utils.parameters_to_vector(self.global_model.parameters())
            params -= 0.01 * update
            torch.nn.utils.vector_to_parameters(params, self.global_model.parameters())

    def update_queues(self, selected, power_alloc, D_t):
        """Update virtual queues"""
        # Update energy queues
        for client in selected:
            cid = client.client_id
            E_com = power_alloc[cid]**2 * client.gradient_norm**2 / abs(client.h_t_k)**2
            
            # Use self.E_max[cid] instead of self.E_max
            self.Q_e[cid] = max(0, self.Q_e[cid] + E_com - self.E_max[cid]/100)
        
        # Update time queue
        avg_round_time = self.T_max / 100
        self.Q_time = max(0, self.Q_time + D_t - avg_round_time)
        
        # Store history
        self.queue_history.append(self.Q_e.copy())
        self.selected_history.append([c.client_id for c in selected])

# Evaluation function
def evaluate_model(model, test_loader, device='cpu'):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# Main simulation
def main():
    # Parameters
    NUM_CLIENTS = 10
    SIM_TIME = 300  # Simulate for 300 seconds
    EVAL_INTERVAL = 30  # Evaluate every 30 seconds
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {DEVICE}")
    
    # Load data
    train_dataset, test_dataset = load_mnist()
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    client_data_map = partition_mnist_noniid(train_dataset, NUM_CLIENTS)
    
    # Create clients
    clients = []
    for cid in range(NUM_CLIENTS):
        client = Client(
            client_id=cid,
            data_indices=client_data_map[cid],
            model=CNNMnist(),
            fk=np.random.uniform(1e9, 2e9),
            mu_k=1e-27,
            P_max=1.0,
            C=1000,
            Ak=32,
            train_dataset=train_dataset,
            device=DEVICE
        )
        clients.append(client)
    
    # Create server
    global_model = CNNMnist().to(DEVICE)
    server = Server(
        global_model=global_model,
        clients=clients,
        V=10.0,
        sigma_n=0.1,
        tau_cm=0.05,
        T_max=100,
        E_max=1.0,
        device=DEVICE
    )
    
    # Simulation loop
    start_time = time.time()
    current_time = 0
    last_eval = 0
    accuracies = []
    eval_times = []
    
    while current_time < SIM_TIME:
        # Update client computations
        for client in clients:
            client.compute_gradient(current_time, global_model)
        
        # Attempt aggregation
        selected, power_alloc = server.select_clients(current_time)
        if selected:
            # Calculate round duration
            comp_times = [c.dt_k for c in selected]
            D_t = max(comp_times) + server.tau_cm
            
            # Perform aggregation
            aggregated = server.aggregate(selected, power_alloc)
            server.update_model(aggregated)
            server.update_queues(selected, power_alloc, D_t)
            
            # Update time
            current_time += D_t
            server.aggregation_times.append(current_time)
            print(f"Aggregation at {current_time:.2f}s: Selected {len(selected)} clients")
        else:
            # No clients ready - advance time
            next_avail = min(c.next_available for c in clients)
            current_time = max(current_time + 1, next_avail)
        
        # Periodic evaluation
        if current_time - last_eval > EVAL_INTERVAL:
            acc = evaluate_model(global_model, test_loader, DEVICE)
            accuracies.append(acc)
            eval_times.append(current_time)
            last_eval = current_time
            print(f"[{current_time:.2f}s] Accuracy: {acc:.2f}%")
    
    # Plot results
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.plot(eval_times, accuracies)
    plt.title("Test Accuracy")
    plt.xlabel("Simulation Time (s)")
    
    plt.subplot(132)
    selected_counts = [len(s) for s in server.selected_history]
    plt.plot(server.aggregation_times, selected_counts, 'o-')
    plt.title("Selected Clients per Aggregation")
    plt.xlabel("Simulation Time (s)")
    
    plt.subplot(133)
    queue_values = [max(q.values()) for q in server.queue_history]
    plt.plot(server.aggregation_times, queue_values)
    plt.title("Max Energy Queue")
    plt.xlabel("Simulation Time (s)")
    
    plt.tight_layout()
    plt.savefig("training_results.png")
    plt.show()

if __name__ == "__main__":
    main()
