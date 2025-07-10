import torch
import numpy as np
import time
import copy
import logging
import math
from collections import defaultdict

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class SyncOTAServer:
    def __init__(self, global_model, clients, E_bars, T, V, gamma0, sigma0, 
                 l, G, Lb, eta_base=0.1, decay_rate=0.95, device='cpu'):
        """
        Synchronous OTA FEEL server implementation
        
        Args:
            global_model: Initial global model
            clients: List of SyncOTAClient objects
            E_bars: Dict {client_id: total_energy_budget}
            T: Total training rounds
            V: Lyapunov trade-off parameter
            gamma0: Target SNR
            sigma0: Noise standard deviation
            l: Smoothness constant (Lipschitz smoothness)
            G: Gradient norm bound
            Lb: Batch size (fixed for all clients)
            eta_base: Base learning rate
            decay_rate: Learning rate decay rate
            device: Computation device
        """
        self.global_model = global_model.to(device)
        self.clients = clients
        self.E_bars = E_bars
        self.T = T
        self.V = V
        self.gamma0 = gamma0
        self.sigma0 = sigma0
        self.l = l
        self.G = G
        self.Lb = Lb
        self.eta_base = eta_base
        self.decay_rate = decay_rate
        self.device = device
        
        # Model dimension
        self.s = sum(p.numel() for p in self.global_model.parameters())
        
        # Virtual queues (Eq 25)
        self.qn = {client.client_id: 0.0 for client in clients}
        
        # History tracking
        self.selected_history = []
        self.energy_consumption = {cid: 0.0 for cid in self.qn.keys()}
        self.queue_history = []
        self.accuracy_history = []
        self.effective_updates = []  # Track update magnitudes
        
        logger.info(f"Server initialized | "
                    f"Model dim: {self.s} | "
                    f"Rounds: {T} | "
                    f"V: {V} | "
                    f"Gamma0: {gamma0} | "
                    f"Noise std: {sigma0}")

    def initialize_clients(self):
        """Initial round to compute gradient norms (Algorithm 2 line 1)"""
        logger.info("=== Initialization Round ===")
        # Broadcast initial model to all clients
        global_state = self.global_model.state_dict()
        for client in self.clients:
            client.update_model(global_state)
        
        # Compute initial gradients and get norms
        for client in self.clients:
            client.compute_gradient()
            logger.info(f"Client {client.client_id}: "
                        f"Initial grad norm: {client.last_grad_norm:.4f}")
        
        logger.info("Initialization complete")

    def set_learning_rate(self, round_idx):
        """Decaying learning rate schedule"""
        self.eta_t = self.eta_base * (self.decay_rate ** (round_idx // 10))
        logger.debug(f"Round {round_idx}: Learning rate set to {self.eta_t:.4f}")
        return self.eta_t

    def calculate_power_scalar(self, grad_norm_estimates):
        """
        Calculate power scalar σt (Eq 29)
        
        σt = √(γ₀σ₀²√s / minₙ(‖g̃ₙₜ‖₂))
        """
        min_grad_norm = min(grad_norm_estimates.values())
        # Avoid division by zero and large gradients
        if min_grad_norm < 1e-4:
            min_grad_norm = 1e-4
        elif min_grad_norm > 100:
            min_grad_norm = 100
            
        # Calculate with numerical stability
        numerator = self.gamma0 * (self.sigma0**2) * math.sqrt(self.s)
        self.sigma_t = math.sqrt(numerator / min_grad_norm)
        
        logger.info(f"Power scalar set: σ_t = {self.sigma_t:.4f} | "
                    f"Min grad norm: {min_grad_norm:.4f}")
        return self.sigma_t

    def select_devices(self):
        """
        Select devices using Algorithm 1 (solving P6)
        
        Steps:
        1. Get channel estimates and EST-P gradient norms
        2. Estimate energy consumption for all clients
        3. Calculate costs: Cₙₜ = qₙₜ * Ẽₙₜ
        4. Sort clients by cost (ascending)
        5. Evaluate all possible subset sizes k=1..N
        6. Select k* that minimizes objective:
            V * Uₜ + Σₙ βₙₜ qₙₜ Ẽₙₜ
        """
        # Get current estimates
        grad_norm_estimates = {}
        h_estimates = {}
        energy_estimates = {}
        costs = {}
        
        for client in self.clients:
            cid = client.client_id
            # Get channel estimate
            h_mag = client.set_channel_gain()
            h_estimates[cid] = h_mag
            
            # EST-P: Use last gradient norm
            grad_norm_estimates[cid] = client.last_grad_norm
            
            # Estimate energy (Eq 26)
            energy_estimates[cid] = client.estimate_energy(self.sigma_t)
            
            # Calculate cost: qₙₜ * Ẽₙₜ
            costs[cid] = self.qn[cid] * energy_estimates[cid]
        
        # Sort clients by cost (ascending)
        sorted_clients = sorted(self.clients, key=lambda c: costs[c.client_id])
        sorted_cids = [c.client_id for c in sorted_clients]
        
        # Initialize best selection
        best_set = []
        best_objective = float('inf')
        
        # Evaluate all possible subset sizes k=1..N
        for k in range(1, len(self.clients) + 1):
            candidate_cids = sorted_cids[:k]
            Bt_size = len(candidate_cids)
            
            # Calculate Uₜ (Eq 19)
            Ut = (self.l * self.eta_t**2 / 2) * (
                self.G**2 / (self.Lb * Bt_size) + 
                (self.sigma0**2 * self.s) / (self.sigma_t**2 * Bt_size**2)
            )
            
            # Sum of costs for candidate set
            sum_costs = sum(costs[cid] for cid in candidate_cids)
            
            # Objective: V*Uₜ + Σ costs
            objective = self.V * Ut + sum_costs
            
            # Track best solution
            if objective < best_objective:
                best_objective = objective
                best_set = candidate_cids
        
        selected_clients = [c for c in self.clients if c.client_id in best_set]
        logger.info(f"Selected {len(selected_clients)} clients: {best_set} | "
                    f"Objective: {best_objective:.4e}")
        return selected_clients

    def aggregate_gradients(self, selected_clients):
        """
        Aggregate gradients via OTA transmission (Eq 8, 10)
        """
        if not selected_clients:
            logger.warning("No clients selected, skipping aggregation")
            return None, {}  # Return None for update and empty energy dict
        
        # Initialize aggregated signal
        aggregated = torch.zeros(self.s, device=self.device)
        actual_energies = {}
        
        # Collect transmissions and actual energies
        for client in selected_clients:
            signal, energy, _ = client.compute_and_transmit(self.sigma_t)
            
            # Direct summation (NO staleness discount)
            aggregated += signal
            
            actual_energies[client.client_id] = energy
            self.energy_consumption[client.client_id] += energy
        
        # Add scaled Gaussian noise
        signal_norm = torch.norm(aggregated).item()
        noise_scale = max(0.01, signal_norm / 10)  # Prevent underflow
        noise = torch.randn(self.s, device=self.device) * self.sigma0 * noise_scale
        noisy_aggregated = aggregated + noise
        
        # Normalize and create update
        normalization = self.sigma_t * len(selected_clients)
        if abs(normalization) < 1e-8:
            logger.error("Normalization near zero! Using safe fallback")
            normalization = 1e-8
            
        update = noisy_aggregated / normalization
        
        # Track effective update magnitude
        update_norm = torch.norm(update).item()
        self.effective_updates.append(update_norm)
        
        logger.info(f"Aggregation complete | "
                    f"Clients: {len(selected_clients)} | "
                    f"Noise std: {self.sigma0} | "
                    f"Update norm: {update_norm:.4f}")
        
        return update, actual_energies

    def update_global_model(self, update, round_idx):
        """Update global model with learning rate decay"""
        if update is None:
            logger.warning("No update received, model not updated")
            return
        
        # Get current model parameters as vector
        params_vector = torch.nn.utils.parameters_to_vector(
            self.global_model.parameters()
        ).detach()
        
        # Gradient clipping to prevent explosions
        max_norm = 10.0
        update_norm = torch.norm(update).item()
        if update_norm > max_norm:
            update = update * (max_norm / update_norm)
            logger.warning(f"Gradient clipped: {update_norm:.2f} -> {max_norm:.2f}")
        
        # Apply update
        new_params = params_vector - self.eta_t * update
        
        # Convert back to parameters
        torch.nn.utils.vector_to_parameters(
            new_params, 
            self.global_model.parameters()
        )
        
        logger.info(f"Model updated | "
                    f"Learning rate: {self.eta_t:.4f} | "
                    f"Update norm: {update_norm:.4f}")

    def update_virtual_queues(self, actual_energies):
        """
        Update virtual queues (Eq 25)
        
        qₙₜ₊₁ = max(0, qₙₜ + βₙₜEₙₜ - Ēₙ/T)
        """
        queue_updates = {}
        
        for client in self.clients:
            cid = client.client_id
            energy_inc = actual_energies.get(cid, 0.0)
            per_round_budget = self.E_bars[cid] / self.T
            
            # Update queue
            prev_q = self.qn[cid]
            self.qn[cid] = max(0, prev_q + energy_inc - per_round_budget)
            
            # Track changes
            queue_updates[cid] = {
                'prev': prev_q,
                'new': self.qn[cid],
                'energy_inc': energy_inc,
                'budget': per_round_budget
            }
        
        # Log queue updates
        for cid, update in queue_updates.items():
            logger.info(f"Queue update client {cid}: "
                         f"{update['prev']:.2f} → {update['new']:.2f} | "
                         f"Energy: {update['energy_inc']:.4e} | "
                         f"Budget/round: {update['budget']:.4e}")
        
        return queue_updates

    def run_round(self, round_idx):
        """Execute one training round"""
        logger.info(f"\n=== Round {round_idx+1}/{self.T} ===")
        
        # Step 0: Set learning rate
        self.set_learning_rate(round_idx)
        
        # Step 1: Calculate power scalar σt using ALL clients
        grad_norms = {c.client_id: c.last_grad_norm for c in self.clients}
        self.calculate_power_scalar(grad_norms)
        
        # Step 2: Select devices (Algorithm 1)
        selected = self.select_devices()
        self.selected_history.append([c.client_id for c in selected])
        
        # Step 3: Broadcast model to selected clients
        global_state = self.global_model.state_dict()
        for client in selected:
            client.update_model(global_state)
        
        # Step 4: Compute and aggregate gradients
        update, actual_energies = self.aggregate_gradients(selected)
        
        # Step 5: Update global model
        self.update_global_model(update, round_idx)
        
        # Step 6: Update virtual queues
        self.update_virtual_queues(actual_energies)
        self.queue_history.append(copy.deepcopy(self.qn))
        
        # Return metrics
        round_metrics = {
            'selected': [c.client_id for c in selected],
            'sigma_t': self.sigma_t,
            'update_norm': torch.norm(update).item() if update is not None else 0,
            'actual_energies': actual_energies,
            'queues': copy.deepcopy(self.qn)
        }
        
        return round_metrics

    def train(self, test_loader, eval_every=5):
        """Full training loop"""
        self.initialize_clients()
        
        for round_idx in range(self.T):
            start_time = time.time()
            round_metrics = self.run_round(round_idx)
            duration = time.time() - start_time
            
            # Periodic evaluation
            if (round_idx + 1) % eval_every == 0 or round_idx == 0:
                accuracy = self.evaluate(test_loader)
                self.accuracy_history.append(accuracy)
                logger.info(f"Round {round_idx+1} accuracy: {accuracy:.2f}% | "
                            f"Duration: {duration:.2f}s")
        
        # Final evaluation
        final_acc = self.evaluate(test_loader)
        self.accuracy_history.append(final_acc)
        logger.info(f"Final accuracy: {final_acc:.2f}%")
        
        return {
            'accuracy_history': self.accuracy_history,
            'energy_consumption': self.energy_consumption,
            'final_queues': self.qn,
            'selection_counts': self.calculate_selection_counts(),
            'effective_updates': self.effective_updates
        }

    def evaluate(self, test_loader):
        """Evaluate global model on test set"""
        self.global_model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                # Add channel dimension if needed (for MNIST)
                if images.dim() == 3:
                    images = images.unsqueeze(1)
                outputs = self.global_model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        return accuracy

    def calculate_selection_counts(self):
        """Calculate how many times each client was selected"""
        counts = defaultdict(int)
        for round_selection in self.selected_history:
            for cid in round_selection:
                counts[cid] += 1
        return dict(counts)