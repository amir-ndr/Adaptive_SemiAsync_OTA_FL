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
        self.E_max = {client.client_id: E_max for client in clients}
        
        # History
        self.selected_history = []
        self.aggregation_times = []
        self.queue_history = []

    def _get_model_dimension(self):
        return sum(p.numel() for p in self.global_model.parameters())
    
    def select_clients(self):
        """Greedy client selection (Algorithm 1)"""
        # Step 1: Prepare clients
        ready_clients = [c for c in self.clients if c.ready]
        if not ready_clients:
            return [], {}
        
        # Step 2: Compute scores ρ_k = |h_k|/√(Q_e·∥g∥²)
        for client in ready_clients:
            client.set_channel_gain()
            score = abs(client.h_t_k) / np.sqrt(
                max(1e-8, self.Q_e[client.client_id]) * 
                max(1e-8, client.gradient_norm)
            )
            client.score = score
        
        # Step 3: Sort by dt_k (ascending) and ρ_k (descending)
        ready_clients.sort(key=lambda c: (c.dt_k, -c.score))
        
        # Step 4: Greedy selection
        selected = []
        best_cost = float('inf')
        best_power = {}
        
        for client in ready_clients:
            temp_selected = selected + [client]
            
            # Compute power allocation for candidate set
            power_alloc, c_values = self._compute_power(temp_selected)
            if not power_alloc:
                continue
                
            # Compute round duration (paper: Sec III.B)
            D_temp = max(c.dt_k for c in temp_selected) + self.tau_cm
            
            # Compute cost (paper: Eq. 25)
            cost = self._compute_cost(temp_selected, power_alloc, D_temp)
            
            if cost < best_cost:
                selected = temp_selected
                best_cost = cost
                best_power = power_alloc
            else:
                break  # Stop when cost increases
                
        return selected, best_power

    def _compute_power(self, selected):
        """Closed-form power allocation (paper: Sec V.A)"""
        c_values = {}
        for client in selected:
            c = self.Q_e[client.client_id] * client.gradient_norm**2 / abs(client.h_t_k)**2
            c_values[client.client_id] = c
            
        inv_sqrt_c_sum = sum(1/np.sqrt(max(1e-8, c)) for c in c_values.values())
        num_clients = len(selected)
        
        # Exact formula from your paper
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
        """Drift-plus-penalty cost (paper: Eq. 24)"""
        total_power = sum(power_alloc.values())
        alpha_sum = 0
        for client in selected:
            alpha = power_alloc[client.client_id] / total_power
            alpha_sum += alpha**2
            
        # Convergence penalty
        convergence_penalty = 1.0 * alpha_sum + self.d * self.sigma_n**2 / total_power**2
        
        # Energy cost
        energy_cost = 0
        for client in selected:
            c = self.Q_e[client.client_id] * client.gradient_norm**2 / abs(client.h_t_k)**2
            energy_cost += c * power_alloc[client.client_id]**2
        
        # Time penalty
        time_penalty = self.Q_time * D_temp
        
        return self.V * convergence_penalty + energy_cost + time_penalty

    def aggregate(self, selected, power_alloc):
        """OTA aggregation (paper: Eq. 5-7)"""
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
        aggregated_grad = (aggregated + noise) / total_power
        
        # Update global model
        self.update_model(aggregated_grad)
        
        return aggregated_grad

    def update_model(self, update, lr=0.01):
        """Update global model (paper: Eq. 9)"""
        with torch.no_grad():
            params = torch.nn.utils.parameters_to_vector(self.global_model.parameters())
            params -= lr * update
            torch.nn.utils.vector_to_parameters(params, self.global_model.parameters())

    def update_queues(self, selected, power_alloc, D_t):
        """Update virtual queues (paper: Eq. 21, 22)"""
        # 1. Update energy queues
        for client in selected:
            cid = client.client_id
            # Total energy = computation + communication
            E_comp = client.mu_k * client.fk**2 * client.C * client.Ak
            E_com = power_alloc[cid]**2 * client.gradient_norm**2 / abs(client.h_t_k)**2
            self.Q_e[cid] = max(0, self.Q_e[cid] + E_comp + E_com - self.E_max[cid]/self.T_total_rounds)
        
        # 2. Update time queue
        self.Q_time = max(0, self.Q_time + D_t - self.T_max/self.T_total_rounds)
        
        # 3. Update computation times (paper: Eq. 21c)
        for client in self.clients:
            if client in selected:
                # Selected clients start new computation
                client.dt_k = client.C * client.Ak / client.fk
                client.ready = False
                client.reset_staleness()
            else:
                # Unselected clients continue computation
                client.update_computation(D_t)
                client.increment_staleness()
        
        # Record history
        self.queue_history.append(self.Q_e.copy())
    def update_client_states(self, D_t):
        """Update all clients' computation states"""
        for client in self.clients:
            client.update_computation(D_t)