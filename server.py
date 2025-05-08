import threading
import numpy as np
import torch
from queue import Queue
from collections import defaultdict

class Server:
    def __init__(self, model, num_clients, total_rounds, E_max, V, 
                 global_lr=0.01, communication_latency=1.0, device="cpu"):
        self.model = model.to(device)
        self.num_clients = num_clients
        self.total_rounds = total_rounds
        self.E_max = E_max
        self.V = V
        self.global_lr = global_lr
        self.communication_latency = communication_latency
        self.device = device

        # Client management
        self.clients = []
        self.client_staleness = defaultdict(int)
        self.selected_clients = []
        self.ready_clients = set()
        
        # Gradient aggregation storage
        self.client_signals = {}
        self.client_energies = {}
        
        # Synchronization
        self.lock = threading.Lock()
        self.round_complete = threading.Event()
        self.round = 0

        # Lyapunov optimization
        self.energy_queue = 0.0
        self.energy_per_round = E_max / total_rounds

        # Metrics
        self.training_times = []
        self.energy_consumption = []

    def register_clients(self, clients):
        self.clients = clients
        for client in clients:
            client.set_callbacks(self.notify_client_completion, self.receive_client_signal)
        self.client_staleness = {k:0 for k in range(len(clients))}

    def notify_client_completion(self, client_id):
        with self.lock:
            self.ready_clients.add(client_id)
            if len(self.ready_clients) == len(self.selected_clients):
                self.round_complete.set()

    def receive_client_signal(self, client_id, signal, energy):
        with self.lock:
            self.client_signals[client_id] = signal.to(self.device)
            self.client_energies[client_id] = energy

    def calculate_gamma(self, client_id):
        tau = max(self.client_staleness[client_id], 1)
        return 1.0 / (tau + 1e-6)

    def select_clients(self):
        candidates = sorted(
            self.clients,
            key=lambda c: c.estimated_training_time(),
            reverse=False
        )

        best_set = []
        min_cost = float('inf')
        
        current_set = []
        for client in candidates:
            current_set.append(client)
            
            # Calculate round duration
            D_t = max(c.estimated_training_time() for c in current_set) + self.communication_latency
            
            # Calculate total energy cost
            E_t = sum(c.estimated_energy() for c in current_set)
            
            # Lyapunov objective function
            objective = self.V * D_t + self.energy_queue * E_t
            
            if objective < min_cost:
                best_set = current_set.copy()
                min_cost = objective
            else:
                break  # Stop when objective starts increasing
        
        return [c.client_id for c in best_set]

    def aggregate_signals(self):
        if not self.selected_clients:
            return None
        valid_clients = [cid for cid in self.selected_clients if cid in self.client_signals]
        if not valid_clients:
            return None

        # Calculate total effective power (denominator for normalization)
        total_weight = 0.0
        for cid in self.selected_clients:
            client = self.clients[cid]
            gamma = self.calculate_gamma(cid)
            total_weight += client.p_k * gamma * torch.norm(client.h_k).item()

        if total_weight < 1e-9:
            raise ValueError("Aggregation weights too small - check client selection")

        # Aggregate complex signals
        first_signal = self.client_signals[valid_clients[0]]
        aggregated = torch.zeros_like(first_signal, device=self.device)
        for cid in self.selected_clients:
            client = self.clients[cid]
            h_k = client.h_k
            signal = self.client_signals[cid]
            
            # Match dimensions if needed
            if signal.shape != h_k.shape:
                signal = signal.expand_as(h_k)
                
            aggregated += signal * h_k

        # Normalize and convert to real-valued gradient
        aggregated_gradient = (aggregated / total_weight).real
        return aggregated_gradient

    def update_global_model(self, aggregated_gradient):
        if aggregated_gradient is None:
            return

        current_params = torch.nn.utils.parameters_to_vector(self.model.parameters())
        updated_params = current_params - self.global_lr * aggregated_gradient
        torch.nn.utils.vector_to_parameters(updated_params, self.model.parameters())

    def update_virtual_queue(self):
        round_energy = sum(self.client_energies.values())
        self.energy_queue = max(self.energy_queue + round_energy - self.energy_per_round, 0)
        self.energy_consumption.append(round_energy)

    def update_staleness(self):
        for cid in range(len(self.clients)):
            if cid not in self.selected_clients:
                self.client_staleness[cid] += 1
            else:
                self.client_staleness[cid] = 0

    def run_round(self, t):
        self.round = t
        self.selected_clients = self.select_clients()
        self.ready_clients.clear()
        self.client_signals.clear()
        self.client_energies.clear()
        self.round_complete.clear()

        # Notify selected clients
        for cid in self.selected_clients:
            staleness = self.client_staleness[cid]
            self.clients[cid].receive_global_model(self.model, staleness)

        # Wait for all selected clients to complete
        self.round_complete.wait(timeout=60.0)  # Add timeout for safety

        # Process aggregation
        agg_grad = self.aggregate_signals()
        if agg_grad is not None:
            self.update_global_model(agg_grad)
            self.update_virtual_queue()
            self.update_staleness()

        print(f"[Round {t}] Selected: {self.selected_clients} | "
              f"Queue: {self.energy_queue:.2f} | "
              f"Energy: {sum(self.client_energies.values()):.2f} J")

    def evaluate_model(self, test_loader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return correct / total

    def shutdown_clients(self):
        for client in self.clients:
            client.stop()