import torch
import numpy as np
import copy

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

    def _get_model_dimension(self):
        return sum(p.numel() for p in self.global_model.parameters())
    
    def broadcast_model(self, selected_clients):
        """Broadcast current model to selected clients (Sec III.B)"""
        global_state = self.global_model.state_dict()
        for client in selected_clients:
            client.update_stale_model(global_state)
    
    def select_clients(self):
        """Greedy client selection with staleness awareness"""
        for client in self.clients:
            client.set_channel_gain()
            grad_norm = client.gradient_norm if hasattr(client, 'gradient_norm') else 1.0
            client.score = abs(client.h_t_k) / np.sqrt(
                max(1e-8, self.Q_e[client.client_id]) * max(1e-8, grad_norm))
        
        # Sort by computation time and score
        sorted_clients = sorted(self.clients, key=lambda c: (c.dt_k, -c.score))
        
        # Greedy selection
        selected = []
        best_cost = float('inf')
        best_power = {}
        best_c_values = {}  # Store best c_values for cost calculation
        MAX_STALENESS = 7

        for client in sorted_clients:
            if client.tau_k > MAX_STALENESS:
                continue  # Skip overly stale clients
            candidate_set = selected + [client]
            power_alloc, c_values = self._compute_power(candidate_set)  # Get c_values
            if not power_alloc: 
                continue
                
            D_temp = max(c.dt_k for c in candidate_set) + self.tau_cm
            # Pass c_values to cost function
            cost = self._compute_cost(candidate_set, power_alloc, c_values, D_temp)
            
            if cost < best_cost:
                selected = candidate_set
                best_cost = cost
                best_power = power_alloc
                best_c_values = c_values  # Store corresponding c_values
            else:
                break
                
        return selected, best_power

    def _compute_power(self, selected):
        c_values = {}
        for client in selected:
            ck = max(1e-8, self.Q_e[client.client_id] * client.gradient_norm**2 / abs(client.h_t_k)**2)
            c_values[client.client_id] = ck
        
        inv_sqrt_c = [1/np.sqrt(c) for c in c_values.values()]
        total_inv_sqrt = sum(inv_sqrt_c)
        
        if total_inv_sqrt < 1e-8:
            return {c.client_id: c.P_max for c in selected}, c_values
        
        # CORRECTED power calculation (Eq V.A)
        S_t = (self.V * self.d * self.sigma_n**2 / (len(selected) * (total_inv_sqrt)**2)) ** 0.25
        
        power_alloc = {}
        for i, client in enumerate(selected):
            ck = c_values[client.client_id]
            pk = (1/np.sqrt(ck) / total_inv_sqrt * S_t)
            # print('power: ', pk)
            power_alloc[client.client_id] = min(pk, client.P_max)
        
        return power_alloc, c_values

    def _compute_cost(self, selected, power_alloc, c_values, D_temp):
        total_power = sum(power_alloc.values())
        
        # Handle case where total_power is near zero
        if total_power < 1e-8:
            alpha_sum = float('inf')
        else:
            alpha_sum = sum((p/(total_power + 1e-8))**2 for p in power_alloc.values())
        
        # Convergence penalty terms
        convergence_penalty = 1.0 * alpha_sum
        if total_power > 1e-8:
            convergence_penalty += self.d * self.sigma_n**2 / (total_power**2)
        
        # Energy cost - FIXED
        energy_cost = 0
        for client in selected:
            # 1. Computation energy (fixed cost when selected)
            E_comp = client.mu_k * client.fk**2 * client.C * client.Ak * client.local_epochs
            # 2. Communication energy (||s_t^k||^2)
            E_comm = (power_alloc[client.client_id] * client.gradient_norm / abs(client.h_t_k))**2
            # 3. Total energy cost with queue weighting
            energy_cost += self.Q_e[client.client_id] * (E_comp + E_comm)
        
        # Time penalty
        time_penalty = self.Q_time * D_temp
        
        return self.V * convergence_penalty + energy_cost + time_penalty

    def aggregate(self, selected, power_alloc):
        total_power = sum(power_alloc.values())
        if total_power < 1e-8:
            return torch.zeros(self.d, device=self.device)
        
        aggregated = torch.zeros(self.d, device=self.device)
        for client in selected:
            p = power_alloc[client.client_id]
            h = client.h_t_k
            compensation = p * np.conj(h) / (abs(h)**2)
            aggregated += client.last_gradient * compensation.real
        
        # Complex Gaussian noise
        noise = torch.randn(self.d, device=self.device) * (self.sigma_n / np.sqrt(2))
        return (aggregated + noise) / total_power

    def update_model(self, update, lr=0.1):
        """Update global model (Eq. 9)"""
        with torch.no_grad():
            params = torch.nn.utils.parameters_to_vector(self.global_model.parameters())
            params -= lr * update
            torch.nn.utils.vector_to_parameters(params, self.global_model.parameters())

    def update_queues(self, selected, power_alloc, D_t):
        """Update queues with proper energy calculation"""
        for client in selected:
            cid = client.client_id
            E_comp = client.mu_k * client.fk**2 * client.C * client.Ak * client.local_epochs
            E_comm = (power_alloc[cid] * client.gradient_norm / abs(client.h_t_k))**2
            # SAME calculation as in _compute_cost
            energy_increment = E_comp + E_comm - self.E_max[cid]/self.T_total_rounds
            self.Q_e[cid] = max(0, self.Q_e[cid] + energy_increment)

        # Time queue update
        self.Q_time = max(0, self.Q_time + D_t - self.T_max/self.T_total_rounds)

        # Record history
        self.selected_history.append([c.client_id for c in selected])
        self.queue_history.append(copy.deepcopy(self.Q_e))