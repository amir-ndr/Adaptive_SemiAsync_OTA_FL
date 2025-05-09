import threading
import numpy as np
import torch
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s:%(message)s')

class Server:
    def __init__(self, model, num_clients, total_rounds, E_max, V,
                 global_lr=0.01, noise_power=0.1, device="cpu"):
        self.model = model.to(device)
        self.num_clients = num_clients
        self.total_rounds = total_rounds
        self.E_max = E_max
        self.V = V
        self.global_lr = global_lr
        self.noise_power = noise_power
        self.device = device

        # Client management
        self.clients = []
        self.client_staleness = defaultdict(int)
        self.selected_clients = []
        self.ready_clients = set()
        
        # Communication buffers
        self.client_signals = {}
        self.client_energies = {}
        
        # Lyapunov optimization
        self.energy_queue = 0.0
        self.energy_per_round = E_max / total_rounds
        
        # Synchronization
        self.lock = threading.Lock()
        self.round_complete = threading.Event()
        self.current_round = 0

    def register_clients(self, clients):
        self.clients = clients
        for client in clients:
            client.set_callbacks(self._notify_client_completion, self._receive_client_signal)
        self.client_staleness = {k:0 for k in range(len(clients))}

    def run_round(self, round_idx):
        self.current_round = round_idx
        logging.info(f"\n=== Starting Round {round_idx}/{self.total_rounds} ===")
        
        try:
            # Phase 1: Client selection
            self.selected_clients = self._select_clients()
            if not self.selected_clients:
                logging.warning("No clients selected for this round")
                return

            # Phase 2: Model broadcast
            self._broadcast_global_model()
            
            # Phase 3: Wait for client responses
            if not self._wait_for_clients(timeout=300):
                logging.warning("Round timed out waiting for clients")
                return

            # Phase 4: OTA aggregation and model update
            aggregated_grad = self._aggregate_signals()
            if aggregated_grad is not None:
                self._update_global_model(aggregated_grad)
            
            # Phase 5: System state updates
            self._update_virtual_queue()
            self._update_staleness()

        except Exception as e:
            logging.error(f"Round failed: {str(e)}")

    def _select_clients(self):
        """Implements Algorithm 1 from the paper"""
        candidates = sorted([c for c in self.clients if c.ready],
                           key=lambda x: x.estimated_training_time())
        
        best_set = []
        min_cost = float('inf')
        current_set = []

        for client in candidates:
            current_set.append(client)
            D_t = max(c.estimated_training_time() for c in current_set)
            E_t = sum(c.estimated_energy() for c in current_set)
            objective = self.V * D_t + self.energy_queue * E_t

            if objective < min_cost:
                best_set = current_set.copy()
                min_cost = objective
            else:
                break

        # Fallback to fastest client if no valid selection
        if not best_set and candidates:
            best_set = [candidates[0]]
            logging.warning("Forced selection of fastest client")

        selected_ids = [c.client_id for c in best_set]
        logging.info(f"Selected clients: {selected_ids}")
        return selected_ids

    def _broadcast_global_model(self):
        """Distribute current global model to selected clients"""
        model_state = self.model.state_dict()
        for cid in self.selected_clients:
            client = self.clients[cid]
            staleness = self.client_staleness[cid]
            logging.info(f"Broadcasting to Client {cid} | Staleness: {staleness}")
            client.receive_global_model(model_state, staleness)

    def _aggregate_signals(self):
        """Implements OTA aggregation from Eq. 9-12"""
        valid_clients = [cid for cid in self.selected_clients 
                        if cid in self.client_signals]
        if not valid_clients:
            return None

        # Calculate total effective power (Eq. 10)
        total_weight = sum(
            self.clients[cid].p_k * self._gamma(cid)
            for cid in valid_clients
        )
        if total_weight < 1e-9:
            raise ValueError("Aggregation weights too small")

        # Sum signals and add noise (Eq. 9,11)
        aggregated = sum(self.client_signals[cid] for cid in valid_clients)
        noise = torch.randn_like(aggregated) * self.noise_power
        normalized_grad = (aggregated + noise) / total_weight

        logging.info(f"Aggregated {len(valid_clients)} gradients | "
                    f"Effective SNR: {torch.norm(aggregated)/self.noise_power:.2f}")
        return normalized_grad.real

    def _update_global_model(self, aggregated_grad):
        """Update model parameters with aggregated gradient"""
        current_params = torch.nn.utils.parameters_to_vector(self.model.parameters())
        updated_params = current_params - self.global_lr * aggregated_grad
        torch.nn.utils.vector_to_parameters(updated_params, self.model.parameters())
        logging.info("Global model updated")

    def _update_virtual_queue(self):
        """Lyapunov queue update from Eq. 16"""
        round_energy = sum(self.client_energies.get(cid, 0) 
                          for cid in self.selected_clients)
        self.energy_queue = max(self.energy_queue + round_energy - self.energy_per_round, 0)
        logging.info(f"Energy update | Used: {round_energy:.2f}J | "
                    f"Queue: {self.energy_queue:.2f}/{self.E_max}")

    def _update_staleness(self):
        """Update staleness counters based on participation"""
        for cid in range(len(self.clients)):
            if cid in self.selected_clients and cid in self.ready_clients:
                self.client_staleness[cid] = 0
            else:
                self.client_staleness[cid] += 1

    def _gamma(self, client_id):
        """Staleness-aware weight from Eq. 7"""
        return 1 / (1 + self.client_staleness[client_id])

    def _wait_for_clients(self, timeout=300):
        """Synchronize with selected clients"""
        self.ready_clients.clear()
        self.round_complete.clear()
        return self.round_complete.wait(timeout=timeout)

    def _notify_client_completion(self, client_id):
        """Callback for client training completion"""
        with self.lock:
            self.ready_clients.add(client_id)
            if self.ready_clients.issuperset(self.selected_clients):
                self.round_complete.set()

    def _receive_client_signal(self, client_id, signal, energy):
        """Callback for OTA signal reception"""
        with self.lock:
            self.client_signals[client_id] = signal.to(self.device)
            self.client_energies[client_id] = energy
            logging.debug(f"Received signal from Client {client_id} | "
                         f"Energy: {energy:.2f}J")

    def evaluate_model(self, test_loader):
        """Evaluate global model accuracy"""
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
        accuracy = 100 * correct / total
        logging.info(f"Model accuracy: {accuracy:.2f}%")
        return accuracy

    def shutdown(self):
        """Graceful termination"""
        for client in self.clients:
            client.stop()
        logging.info("Server shutdown completed")