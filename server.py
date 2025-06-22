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
        
        # Virtual queues (paper: Sec IV.C)
        self.Q_e = {client.client_id: 0.0 for client in clients}  # Energy queues
        self.Q_time = 0.0  # Time queue
        
        # Energy budgets
        if isinstance(E_max, dict):
            self.E_max = E_max
        else:
            self.E_max = {client.client_id: E_max for client in clients}
        
        # History
        self.selected_history = []
        self.aggregation_times = []
        self.queue_history = []

    def _get_model_dimension(self):
        return sum(p.numel() for p in self.global_model.parameters())
    
    def select_clients(self):
        """Greedy client selection (Algorithm 1)"""
        # Compute scores for all clients
        for client in self.clients:
            client.set_channel_gain()
            # Compute score ρ_k = |h_k|/√(Q_e * ||g||²)
            # Use last gradient norm if available, else use 1.0
            grad_norm = client.gradient_norm if hasattr(client, 'gradient_norm') else 1.0
            score = abs(client.h_t_k) / np.sqrt(
                max(1e-8, self.Q_e[client.client_id]) * 
                max(1e-8, grad_norm))
            client.score = score
        
        # Sort by ascending dt_k (lowest remaining time first)
        # and then descending score (highest ρ_k score first)
        sorted_clients = sorted(self.clients, key=lambda c: (c.dt_k, -c.score))
        
        # Greedy selection (Algorithm 1)
        selected = []
        best_cost = float('inf')
        best_power = {}
        
        for client in sorted_clients:
            candidate_set = selected + [client]
            
            # Compute power allocation for candidate set
            power_alloc, c_values = self._compute_power(candidate_set)
            if not power_alloc:
                continue
                
            # Compute round duration (paper: Sec III.B)
            D_temp = max(c.dt_k for c in candidate_set) + self.tau_cm
            
            # Compute drift-plus-penalty cost (paper: Eq. 25)
            cost = self._compute_cost(candidate_set, power_alloc, D_temp)
            
            # Keep client if cost decreases
            if cost < best_cost:
                selected = candidate_set
                best_cost = cost
                best_power = power_alloc
            else:
                # Stop when cost starts increasing
                break
                
        return selected, best_power

    def _compute_power(self, selected):
        """Closed-form power allocation (paper: Sec V.A)"""
        c_values = {}
        for client in selected:
            # c_k = Q_e^k(t) * ∥g_t^k∥^2 / |h_t^k|^2
            c = self.Q_e[client.client_id] * client.gradient_norm**2 / abs(client.h_t_k)**2
            c_values[client.client_id] = c
            
        inv_sqrt_c_sum = sum(1/np.sqrt(max(1e-8, c)) for c in c_values.values())
        num_clients = len(selected)
        
        # Exact formula from paper
        total_power = (
            (self.V * self.d * self.sigma_n**2) / 
            (num_clients * (inv_sqrt_c_sum)**2)
        )**0.25
        
        power_alloc = {}
        for client in selected:
            c = c_values[client.client_id]
            power = (1/np.sqrt(c) / inv_sqrt_c_sum) * total_power
            # Apply power constraint
            power_alloc[client.client_id] = min(power, client.P_max)
            
        return power_alloc, c_values

    def _compute_cost(self, selected, power_alloc, D_temp):
        """Drift-plus-penalty cost (paper: Eq. 25)"""
        total_power = sum(power_alloc.values())
        alpha_sum = 0
        for client in selected:
            alpha = power_alloc[client.client_id] / total_power
            alpha_sum += alpha**2
            
        # Convergence penalty (G^2 = 1.0 as in paper implementation)
        convergence_penalty = 1.0 * alpha_sum + self.d * self.sigma_n**2 / total_power**2
        
        # Energy cost
        energy_cost = 0
        for client in selected:
            # c_k * (p_k)^2
            c = self.Q_e[client.client_id] * client.gradient_norm**2 / abs(client.h_t_k)**2
            energy_cost += c * power_alloc[client.client_id]**2
        
        # Time penalty
        time_penalty = self.Q_time * D_temp
        
        return self.V * convergence_penalty + energy_cost + time_penalty

    def aggregate(self, selected, power_alloc):
        """OTA aggregation (paper: Eq. 5-7)"""
        # Compute total effective power
        total_power = sum(power_alloc.values())
        aggregated = torch.zeros(self.d, device=self.device)
        
        for client in selected:
            p = power_alloc[client.client_id]
            h = client.h_t_k
            # Apply power and channel compensation
            scaled_grad = client.last_gradient * (p * h.conjugate().real / abs(h))
            aggregated += scaled_grad
        
        # Add noise
        noise = torch.randn(self.d, device=self.device) * self.sigma_n
        return (aggregated + noise) / total_power

    def update_model(self, update, lr=0.01):
        """Update global model (paper: Eq. 9)"""
        with torch.no_grad():
            params = torch.nn.utils.parameters_to_vector(self.global_model.parameters())
            params -= lr * update
            torch.nn.utils.vector_to_parameters(params, self.global_model.parameters())

    def update_queues(self, selected, power_alloc, D_t):
        """Update virtual queues and client states (paper: Eq. 21,22)"""
        # 1. Update energy queues
        for client in selected:
            cid = client.client_id
            # Total energy = computation + communication
            E_comp = client.mu_k * client.fk**2 * client.C * client.Ak
            E_com = power_alloc[cid]**2 * client.gradient_norm**2 / abs(client.h_t_k)**2
            self.Q_e[cid] = max(0, self.Q_e[cid] + E_comp + E_com - 
                                self.E_max[cid]/self.T_total_rounds)
        
        # 2. Update time queue
        self.Q_time = max(0, self.Q_time + D_t - self.T_max/self.T_total_rounds)
        
        # 3. Update client states (paper: Eq. 21c)
        for client in self.clients:
            if client in selected:
                # Selected clients start new computation
                client.reset_computation()
                client.reset_staleness()
            else:
                # Unselected clients continue computation
                client.update_computation_time(D_t)
                client.increment_staleness()
        
        # Record history
        self.queue_history.append(copy.deepcopy(self.Q_e))
        self.selected_history.append([c.client_id for c in selected])