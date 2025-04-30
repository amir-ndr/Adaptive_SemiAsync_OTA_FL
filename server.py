import torch
import numpy as np
import threading

class Server:
    def __init__(self, model, device, num_clients, V, P_max, T, noise_std=1e-3):
        self.device = device
        self.model = model.to(self.device)
        self.global_model_flat = self.flatten_model()
        
        # Lyapunov optimization parameters
        self.V = V
        self.P_max = P_max
        self.T = T
        self.Q = 0.0  # Virtual energy queue

        self.noise_std = noise_std  # OTA noise std
        self.num_clients = num_clients

    def flatten_model(self):
        return torch.cat([param.data.view(-1) for param in self.model.parameters()])

    def set_model_from_flat(self, flat_model):
        pointer = 0
        for param in self.model.parameters():
            num_param = param.numel()
            param.data = flat_model[pointer:pointer + num_param].view_as(param).clone()
            pointer += num_param

    def select_clients(self, ready_clients, tau_cm):
        """
        Algorithm 1: Adaptive client selection based on drift-plus-penalty.
        """
        if not ready_clients:
            print("[Server] No ready clients to select from.")
            return []

        print(f"[Server] Running selection from {len(ready_clients)} ready clients.")

        # Step 6: Compute remaining training times
        dtk = {c.id: c.get_computation_time() for c in ready_clients}

        # Step 7: Sort clients ascending order of dt_k
        sorted_clients = sorted(ready_clients, key=lambda c: dtk[c.id])

        # Step 5: Initialize
        selected_St = []
        Jmin = float('inf')

        # Step 8-17: Add clients greedily
        for client in sorted_clients:
            tentative_St = selected_St + [client]
            Dt = max(dtk[c.id] for c in tentative_St) + tau_cm

            # Drift-plus-penalty objective
            drift_penalty = self.V * Dt + self.Q * sum(c.gtk.norm().item()**2 for c in tentative_St)

            print(f"[Server] Tentatively adding Client {client.id}: DriftPenalty = {drift_penalty:.2f}")

            if drift_penalty < Jmin:
                Jmin = drift_penalty
                selected_St = tentative_St
            else:
                print(f"[Server] Adding Client {client.id} did not improve objective. Stopping selection.")
                break

        print(f"[Server] Final selected clients: {[c.id for c in selected_St]}")
        return selected_St

    def assign_events(self, selected_clients):
        """
        Assign a new threading.Event to each selected client to signal readiness.
        """
        event_map = {}
        for client in selected_clients:
            ready_event = threading.Event()
            event_map[client.id] = ready_event
            client.receive_global_model(self.global_model_flat.clone(), server_ready_event=ready_event)
            print(f"[Server] Sent global model to Client {client.id} with event.")
        return event_map

    def wait_for_clients(self, event_map):
        """
        Wait for all selected clients to finish local training.
        """
        print(f"[Server] Waiting for {len(event_map)} clients to complete local training...")
        for client_id, event in event_map.items():
            event.wait()
            print(f"[Server] Client {client_id} signaled ready.")

    def receive_ota_signal(self, selected_clients):
        """
        OTA aggregation from selected clients.
        """
        if not selected_clients:
            print("[Server] No clients selected. Returning zero gradient.")
            return torch.zeros_like(self.global_model_flat)

        print(f"[Server] Receiving OTA signal from clients: {[c.id for c in selected_clients]}")

        aggregated_signal = torch.zeros_like(self.global_model_flat, dtype=torch.cfloat)
        self.client_powers = {}  # Save (ptk, gamma_tk) per client

        for client in selected_clients:
            ptk = np.random.uniform(0.5, 2.0)
            gamma_tk = 1.0 / (1 + client.staleness)
            s_tk = client.get_ota_signal(ptk, gamma_tk)
            aggregated_signal += s_tk
            self.client_powers[client.id] = (ptk, gamma_tk)
            print(f"[Server] Client {client.id}: ptk={ptk:.2f}, gamma_tk={gamma_tk:.2f}, staleness={client.staleness}")

        noise = torch.randn_like(aggregated_signal) * self.noise_std
        aggregated_signal += noise

        normalization_factor = sum(ptk * gamma_tk for ptk, gamma_tk in self.client_powers.values())
        normalized_gradient = aggregated_signal / normalization_factor
        print(f"[Server] OTA aggregation complete. Normalization factor: {normalization_factor:.4f}")

        return normalized_gradient

    def global_update(self, normalized_gradient, eta_global):
        """
        Update the global model using the aggregated gradient.
        """
        self.global_model_flat -= eta_global * normalized_gradient.real
        self.set_model_from_flat(self.global_model_flat)
        print("[Server] Global model updated.")

    def update_virtual_queue(self, selected_clients):
        """
        Update the Lyapunov virtual queue.
        """
        round_energy = 0.0
        for client in selected_clients:
            ptk, gamma_tk = self.client_powers[client.id]
            energy = (ptk * gamma_tk)**2 * client.gtk.norm().item()**2
            round_energy += energy

        self.Q = max(self.Q + round_energy - (self.P_max / self.T), 0)
        print(f"[Server] Updated virtual queue Q(t): {self.Q:.2f}")
