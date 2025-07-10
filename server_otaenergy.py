import torch
import numpy as np
import logging
import math
import time
from collections import defaultdict
import copy

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class SyncOTAServer:
    def __init__(self, global_model, clients, E_bars, T, V, gamma0, sigma0,
                 l, G, Lb, eta_base=0.15, decay_rate=0.95, device='cpu'):
        """
        Synchronous OTA FEEL server implementation (Fixed to match paper)
       
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
        self.effective_updates = []
       
        # Performance metrics
        self.round_times = []
        self.cumulative_energy = {cid: 0.0 for cid in self.qn.keys()}
        self.grad_norm_history = []
       
        logger.info(f"Server initialized | "
                    f"Model dim: {self.s} | "
                    f"Rounds: {T} | "
                    f"V: {V} | "
                    f"Gamma0: {gamma0} | "
                    f"Noise std: {sigma0}")

    def initialize_clients(self):
        """Initial round to compute gradient norms"""
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
        self.eta_t = self.eta_base * (self.decay_rate ** (round_idx // 10))  # Less frequent decay
        logger.info(f"Round {round_idx}: Learning rate set to {self.eta_t:.4f}")
        return self.eta_t

    def calculate_power_scalar(self, grad_norm_estimates):
        """
        Calculate power scalar σt (Eq 29)
       
        σt = √(γ₀σ₀²√s / minₙ(‖g̃ₙₜ‖₂))
        """
        min_grad_norm = min(grad_norm_estimates.values())
        # Safe clipping for stability
        min_grad_norm = max(min_grad_norm, 0.01)   # Prevent division by zero
        min_grad_norm = min(min_grad_norm, 10.0)   # Prevent excessive scaling
           
        # Calculate with numerical stability
        numerator = self.gamma0 * (self.sigma0**2) * math.sqrt(self.s)
        self.sigma_t = math.sqrt(numerator / min_grad_norm)
       
        logger.info(f"Power scalar set: σ_t = {self.sigma_t:.4f} | "
                    f"Min grad norm: {min_grad_norm:.4f}")
        return self.sigma_t

    def select_devices(self):
        """
        CORRECTED device selection using Algorithm 1 from paper
        """
        # Calculate Ẽn,t for all clients
        E_estimates = {}
        for client in self.clients:
            E_estimates[client.client_id] = client.estimate_energy(self.sigma_t)
        
        # Compute C_t = qn,t * Ẽn,t
        C_t = {client.client_id: self.qn[client.client_id] * E_estimates[client.client_id] 
               for client in self.clients}
        
        # Sort clients by C_t (ascending)
        sorted_clients = sorted(self.clients, key=lambda c: C_t[c.client_id])
        sorted_C = [C_t[c.client_id] for c in sorted_clients]
        
        # Initialize best selection
        best_set = []
        best_cost = float('inf')
        
        # Precompute constants for Ut
        U_const = (self.l * self.eta_t**2) / 2
        G_term = self.G**2 / self.Lb
        noise_term = (self.sigma0**2 * self.s) / (self.sigma_t**2)
        
        # Try all k values (1 to N)
        for k in range(1, len(self.clients) + 1):
            # Calculate Ut(k) (Eq 19)
            Ut = U_const * (G_term * k + noise_term / (k**2))
            
            # Sum of costs for first k clients
            sum_costs = sum(sorted_C[:k])
            
            # Objective function: V*Ut + sum_costs
            objective = self.V * Ut + sum_costs
            
            if objective < best_cost:
                best_cost = objective
                best_set = sorted_clients[:k]
        
        selected_clients = best_set
        logger.info(f"Selected {len(selected_clients)} clients | "
                    f"Objective: {best_cost:.4e} | "
                    f"IDs: {[c.client_id for c in selected_clients]}")
        return selected_clients

    def aggregate_gradients(self, selected_clients):
        """
        CORRECTED gradient aggregation without staleness
        """
        if not selected_clients:
            logger.warning("No clients selected, skipping aggregation")
            return None, {}
       
        # Initialize aggregated signal
        aggregated = torch.zeros(self.s, device=self.device)
        actual_energies = {}
        grad_norms = []
       
        # Collect transmissions
        for client in selected_clients:
            # Compute gradient and transmit (will update last_grad_norm)
            signal, energy, grad_norm = client.compute_and_transmit(self.sigma_t)
            aggregated += signal
            actual_energies[client.client_id] = energy
            self.energy_consumption[client.client_id] += energy
            grad_norms.append(grad_norm)
       
        # Add pure Gaussian noise (no scaling)
        noise = torch.randn(self.s, device=self.device) * self.sigma0
        noisy_aggregated = aggregated + noise
       
        # Normalize and create update
        k = len(selected_clients)
        normalization = self.sigma_t * k
        if abs(normalization) < 1e-8:
            logger.error("Normalization near zero! Using safe fallback")
            normalization = 1e-8
           
        update = noisy_aggregated / normalization
       
        # Track metrics
        update_norm = torch.norm(update).item()
        self.effective_updates.append(update_norm)
        self.grad_norm_history.append(np.mean(grad_norms))
       
        logger.info(f"Aggregation complete | "
                    f"Clients: {k} | "
                    f"Update norm: {update_norm:.4f} | "
                    f"Avg grad norm: {np.mean(grad_norms):.4f}")
       
        return update, actual_energies

    def update_global_model(self, update):
        """CORRECTED model update with plain SGD (no momentum)"""
        if update is None:
            logger.warning("No update received, model not updated")
            return
       
        # Get current model parameters as vector
        params_vector = torch.nn.utils.parameters_to_vector(
            self.global_model.parameters()
        ).detach()
       
        # Gradient clipping to prevent explosions
        max_norm = 2.0  # Tighter clipping
        update_norm = torch.norm(update).item()
        if update_norm > max_norm:
            update = update * (max_norm / update_norm)
            logger.warning(f"Gradient clipped: {update_norm:.2f} -> {max_norm:.2f}")
       
        # Apply SGD update
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
        """
        queue_updates = {}
       
        for client in self.clients:
            cid = client.client_id
            energy_inc = actual_energies.get(cid, 0.0)
            per_round_budget = self.E_bars[cid] / self.T
           
            # Update queue
            prev_q = self.qn[cid]
            self.qn[cid] = max(0, prev_q + energy_inc - per_round_budget)
           
            # Track cumulative energy
            self.cumulative_energy[cid] += energy_inc
           
            # Track changes
            queue_updates[cid] = {
                'prev': prev_q,
                'new': self.qn[cid],
                'energy_inc': energy_inc,
                'budget': per_round_budget
            }
       
        # Log queue updates
        for cid, update in queue_updates.items():
            if cid in actual_energies:  # Only log selected clients
                logger.info(f"Queue update client {cid}: "
                            f"{update['prev']:.2f} → {update['new']:.2f} | "
                            f"Energy: {update['energy_inc']:.4e} | "
                            f"Budget/round: {update['budget']:.4e}")
       
        return queue_updates

    def run_round(self, round_idx):
        """Execute one training round with correct sequence"""
        logger.info(f"\n=== Round {round_idx+1}/{self.T} ===")
        start_time = time.time()
       
        # Step 0: Set learning rate
        self.set_learning_rate(round_idx)
        
        # Step 1: Pre-round setup for ALL clients
        grad_norm_estimates = {}
        channel_gains = np.random.exponential(scale=1.0, size=len(self.clients))
        
        for i, client in enumerate(self.clients):
            # Set channel gains from server-side generation
            client.set_channel_gain(channel_gains[i])
            # Use last known gradient norm (EST-P)
            grad_norm_estimates[client.client_id] = client.last_grad_norm
        
        # Step 2: Calculate power scalar using ALL clients
        self.calculate_power_scalar(grad_norm_estimates)
        
        # Step 3: Select devices using corrected algorithm
        selected = self.select_devices()
        self.selected_history.append([c.client_id for c in selected])
       
        # Step 4: Broadcast model to selected clients
        global_state = self.global_model.state_dict()
        for client in selected:
            client.update_model(global_state)
       
        # Step 5: Compute and aggregate gradients
        update, actual_energies = self.aggregate_gradients(selected)
       
        # Step 6: Update global model
        self.update_global_model(update)
       
        # Step 7: Update virtual queues
        self.update_virtual_queues(actual_energies)
        self.queue_history.append(copy.deepcopy(self.qn))
       
        # Record round time
        self.round_times.append(time.time() - start_time)
        avg_time = np.mean(self.round_times[-5:]) if self.round_times else 0
        logger.info(f"Round completed in {self.round_times[-1]:.2f}s | "
                    f"Avg last 5: {avg_time:.2f}s")
       
        # Return metrics
        round_metrics = {
            'selected': [c.client_id for c in selected],
            'sigma_t': self.sigma_t,
            'update_norm': torch.norm(update).item() if update is not None else 0,
            'actual_energies': actual_energies,
            'queues': copy.deepcopy(self.qn)
        }
       
        return round_metrics

    def train(self, test_loader, eval_every=10):
        """Full training loop with stability checks"""
        self.initialize_clients()
        best_accuracy = 0.0
        early_stop_counter = 0
        max_early_stop = 10  # Stop if no improvement in 10 evaluations
       
        for round_idx in range(self.T):
            try:
                round_metrics = self.run_round(round_idx)
            except Exception as e:
                logger.error(f"Round {round_idx} failed: {str(e)}")
                if round_idx > 10:  # Don't break on early failures
                    break
                continue
           
            # Periodic evaluation
            if (round_idx + 1) % eval_every == 0 or round_idx == 0:
                accuracy = self.evaluate(test_loader)
                self.accuracy_history.append(accuracy)
                
                # Early stopping check
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
                
                logger.info(f"Round {round_idx+1} accuracy: {accuracy:.2f}% | "
                            f"Best: {best_accuracy:.2f}% | "
                            f"Early stop counter: {early_stop_counter}/{max_early_stop}")
                
                if early_stop_counter >= max_early_stop:
                    logger.info(f"Early stopping at round {round_idx+1}")
                    break
       
        # Final evaluation
        final_acc = self.evaluate(test_loader)
        self.accuracy_history.append(final_acc)
        logger.info(f"Final accuracy: {final_acc:.2f}%")
       
        return {
            'accuracy_history': self.accuracy_history,
            'energy_consumption': self.cumulative_energy,
            'final_queues': self.qn,
            'selection_counts': self.calculate_selection_counts(),
            'effective_updates': self.effective_updates,
            'round_times': self.round_times,
            'grad_norms': self.grad_norm_history
        }

    def evaluate(self, test_loader):
        """Evaluate global model on test set"""
        self.global_model.eval()
        correct = 0
        total = 0
       
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
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