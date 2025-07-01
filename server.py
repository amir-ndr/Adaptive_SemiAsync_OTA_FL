import torch
import numpy as np
import copy
import logging

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Server:
    def __init__(self, global_model, clients, V=1.0, sigma_n=0.1, 
                 tau_cm=0.1, T_max=100, E_max=1.0, T_total_rounds=50, 
                 device='cpu'):
        self.global_model = global_model
        self.clients = clients
        self.device = device
        self.V = V
        self.sigma_n = sigma_n
        self.tau_cm = tau_cm
        self.T_max = T_max
        self.T_total_rounds = T_total_rounds
        self.d = self._get_model_dimension()
        
        # Virtual queues
        self.Q_e = {client.client_id: 0.0 for client in clients}
        self.Q_time = 0.0
        
        # Energy budgets
        self.E_max = E_max if isinstance(E_max, dict) else \
                    {c.client_id: E_max for c in clients}
        
        # History
        self.selected_history = []
        self.queue_history = []

        logger.info(f"Server initialized | "
                    f"Model dim: {self.d} | "
                    f"V: {V} | "
                    f"Noise: {sigma_n} | "
                    f"Rounds: {T_total_rounds}")

    def _get_model_dimension(self):
        return sum(p.numel() for p in self.global_model.parameters())
    
    def broadcast_model(self, selected_clients):
        global_state = self.global_model.state_dict()
        for client in selected_clients:
            client.update_stale_model(global_state)
        logger.info(f"Broadcast model to {len(selected_clients)} clients")
    
    def select_clients(self):
        epsilon = 1e-8
        
        # Log queue status before selection
        queue_status = ", ".join([f"Client {cid}: {q:.2f}" 
                                 for cid, q in self.Q_e.items()])
        logger.info(f"Pre-selection queues | "
                    f"Time Q: {self.Q_time:.2f} | "
                    f"Energy Qs: {queue_status}")
        
        # Compute scores
        for client in self.clients:
            client.set_channel_gain()
            numerator = abs(client.h_t_k)**2
            denominator = (self.Q_e[client.client_id] + epsilon) * \
                         (client.gradient_norm**2 + epsilon) * \
                         (client.dt_k + epsilon)
            client.score = numerator / denominator
        
        # Log client status
        for client in self.clients:
            logger.debug(f"Client {client.client_id} status | "
                         f"Score: {client.score:.4e} | "
                         f"dt_k: {client.dt_k:.4f}s | "
                         f"Q_e: {self.Q_e[client.client_id]:.2f} | "
                         f"|h|: {abs(client.h_t_k):.4f} | "
                         f"Grad norm: {client.gradient_norm:.4f}")
        
        # Sort and select
        sorted_clients = sorted(self.clients, key=lambda c: c.score, reverse=True)
        selected = []
        best_cost = float('inf')
        
        for client in sorted_clients:
            cost_k = self._approx_cost(client, selected)
            if cost_k < best_cost:
                selected.append(client)
                best_cost = cost_k
                logger.debug(f"  Added client {client.client_id} | "
                             f"New cost: {cost_k:.4e}")
            else:
                logger.debug(f"  Stopping selection | "
                             f"Client {client.client_id} would increase cost to {cost_k:.4e}")
                break
        
        # Compute power allocation
        power_alloc, c_values = self._compute_power(selected)
        
        logger.info(f"Selected {len(selected)} clients: {[c.client_id for c in selected]}")
        for client in selected:
            logger.info(f"  Client {client.client_id} | "
                        f"Power: {power_alloc[client.client_id]:.4f} | "
                        f"c_k: {c_values[client.client_id]:.4e} | "
                        f"Score: {client.score:.4e}")
        
        return selected, power_alloc

    def _approx_cost(self, candidate, current_set):
        candidate_set = current_set + [candidate]
        n = len(candidate_set)
        
        # Convergence penalty
        conv_penalty = 1.0 / n
        avg_pmax = sum(c.P_max for c in candidate_set) / n
        conv_penalty += self.d * self.sigma_n**2 / (avg_pmax**2 * n**2 + 1e-8)
        
        # Energy cost
        energy_cost = 0
        for client in candidate_set:
            E_comp = client.mu_k * client.fk**2 * client.C * client.Ak * client.local_epochs
            # E_comp = 0.05
            E_comm = (client.P_max * client.gradient_norm / abs(client.h_t_k))**2
            energy_cost += self.Q_e[client.client_id] * (E_comp + E_comm)
        
        # Time penalty
        D_temp = max(c.dt_k for c in candidate_set) + self.tau_cm
        
        total_cost = self.V * conv_penalty + energy_cost + self.Q_time * D_temp
        return total_cost

    def _compute_power(self, selected):
        if not selected:
            return {}, {}

        n = len(selected)
        c_values = {}
        
        for client in selected:
            normalized_grad_norm = client.gradient_norm**2 / self.d
            ck = max(1e-8, self.Q_e[client.client_id] * normalized_grad_norm / abs(client.h_t_k)**2)
            c_values[client.client_id] = ck
        
        weights = [1/np.sqrt(c) for c in c_values.values()]
        total_weight = sum(weights)
        
        # Handle near-zero weights - ADDED FALLBACK
        if total_weight < 1e-8:
            # Equal power allocation fallback
            power_alloc = {client.client_id: min(1.0, client.P_max) for client in selected}
            logger.warning("Total weight near zero - using equal power allocation")
            return power_alloc, c_values
        
        # Optimize total power
        S_t = (self.V * self.sigma_n**2 * total_weight**2 / n) ** 0.25
        
        # REMOVED SOFTMAX SCALING - CAUSES OVER-AMPLIFICATION
        power_alloc = {}
        for i, client in enumerate(selected):
            pk = weights[i] / total_weight * S_t
            power_alloc[client.client_id] = min(pk, client.P_max)
        
        logger.debug(f"Power allocation | "
                    f"Total S_t: {S_t:.4f} | "
                    f"Total weight: {total_weight:.4e} | "
                    f"n: {n}")
        return power_alloc, c_values

    def aggregate(self, selected, power_alloc):
        total_power = sum(power_alloc.values())
        if total_power < 1e-8:
            logger.warning("Aggregation failed: total power near zero!")
            return torch.zeros(self.d, device=self.device)
        
        aggregated = torch.zeros(self.d, device=self.device)
        for client in selected:
            p = power_alloc[client.client_id]
            h = client.h_t_k
            staleness_factor = 0.9 ** client.tau_k
            compensation = p * np.conj(h) / (abs(h)**2)
            aggregated += client.last_gradient * compensation.real #* staleness_factor
        
        # Add noise
        noise = torch.randn(self.d, device=self.device) * self.sigma_n
        result = (aggregated + noise) / total_power
        
        logger.info(f"Aggregation complete | "
                    f"Total power: {total_power:.4f} | "
                    f"Noise std: {self.sigma_n} | "
                    f"Update norm: {torch.norm(result).item():.4f}")
        return result

    def update_model(self, update, lr=0.3):
        with torch.no_grad():
            params = torch.nn.utils.parameters_to_vector(self.global_model.parameters())
            prev_norm = torch.norm(params).item()
            params -= lr * update
            new_norm = torch.norm(params).item()
            torch.nn.utils.vector_to_parameters(params, self.global_model.parameters())
            
            logger.info(f"Model updated | "
                        f"Param change: {prev_norm - new_norm:.4e} | "
                        f"New norm: {new_norm:.4f}")

    def update_queues(self, selected, power_alloc, D_t):
        logger.info(f"Updating queues | "
                    f"Round duration: {D_t:.4f}s | "
                    f"Time Q before: {self.Q_time:.2f}")
        
        # Update energy queues
        for client in selected:
            cid = client.client_id
            E_comm = (power_alloc[cid] * client.gradient_norm / abs(client.h_t_k))**2
            E_comp = client.mu_k * client.fk**2 * client.C * client.Ak * client.local_epochs
            # E_comp = 0.05
            energy_increment = E_comp + E_comm - self.E_max[cid]/self.T_total_rounds
            
            prev_q = self.Q_e[cid]
            self.Q_e[cid] = max(0, self.Q_e[cid] + energy_increment)
            
            logger.info(f"  Client {cid} energy update | "
                         f"Comp: {E_comp:.4e} J | "
                         f"Comm: {E_comm:.4e} J | "
                         f"ΔQ: {energy_increment:.4e} | "
                         f"Q_e: {prev_q:.2f} → {self.Q_e[cid]:.2f}")
        
        # Update time queue
        time_increment = D_t - self.T_max/self.T_total_rounds
        prev_time_q = self.Q_time
        self.Q_time = max(0, self.Q_time + time_increment)
        
        logger.info(f"Time queue update | "
                    f"Δ: {time_increment:.4e} | "
                    f"Q_time: {prev_time_q:.2f} → {self.Q_time:.2f}")
        
        # Record history
        self.selected_history.append([c.client_id for c in selected])
        self.queue_history.append(copy.deepcopy(self.Q_e))