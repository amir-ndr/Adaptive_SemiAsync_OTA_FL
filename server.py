import threading
import numpy as np
import torch
from queue import Queue

class Server:
    def __init__(self, model, num_clients, total_rounds, E_max, V, global_lr=0.01, communication_latency=1.0):
        self.model = model
        self.num_clients = num_clients
        self.total_rounds = total_rounds
        self.E_max = E_max
        self.V = V
        self.global_lr = global_lr
        self.communication_latency = communication_latency

        self.clients = []
        self.selected_clients = []
        self.ready_clients = set()
        self.client_gradients = {}
        self.client_energies = {}
        self.global_model = model.state_dict()
        self.lock = threading.Lock()
        self.round_complete = threading.Event()
        self.round = 0

        # Lyapunov virtual energy queue
        self.energy_queue = 0
        self.energy_per_round = E_max / total_rounds

        # Metrics
        self.training_times = []
        self.energy_consumption = []

    def register_clients(self, clients):
        self.clients = clients
        for client in clients:
            client.set_callbacks(self.receive_signal, self.receive_gradient)

    def receive_signal(self, client_id):
        with self.lock:
            self.ready_clients.add(client_id)
            if all(cid in self.ready_clients for cid in self.selected_clients):
                self.round_complete.set()

    def receive_gradient(self, client_id, gradient, power, gamma):
        with self.lock:
            self.client_gradients[client_id] = gradient
            energy = power * torch.norm(gradient).item() ** 2  # OTA transmission energy
            self.client_energies[client_id] = energy
            self.clients[client_id].power = power
            self.clients[client_id].gamma = gamma

    def select_clients_dpp(self):
        N_t = [k for k in range(self.num_clients)]
        candidate_set = []
        J_min = float("inf")
        selected = []

        # Sort by estimated local training time
        N_t_sorted = sorted(N_t, key=lambda k: self.clients[k].estimated_training_time())

        for k in N_t_sorted:
            candidate_set.append(k)
            D_t = max([self.clients[i].estimated_training_time() for i in candidate_set])
            E_t = sum([self.clients[i].estimated_energy() for i in candidate_set])
            J = self.V * D_t + self.energy_queue * E_t

            if J < J_min:
                J_min = J
                selected = list(candidate_set)
            else:
                break

        return selected

    def aggregate_gradients(self):
        total_weight = sum(
            self.clients[i].power * self.clients[i].gamma for i in self.selected_clients
        )
        aggregated_gradient = None

        for k in self.selected_clients:
            alpha_k = self.clients[k].power * self.clients[k].gamma / total_weight
            grad = self.client_gradients[k]
            if aggregated_gradient is None:
                aggregated_gradient = {key: alpha_k * val.clone() for key, val in grad.items()}
            else:
                for key in aggregated_gradient:
                    aggregated_gradient[key] += alpha_k * grad[key]

        return aggregated_gradient

    def update_global_model(self, aggregated_gradient):
        new_state_dict = self.global_model.copy()
        for key in new_state_dict:
            new_state_dict[key] -= self.global_lr * aggregated_gradient[key].real  # only use real part
        self.model.load_state_dict(new_state_dict)
        self.global_model = new_state_dict

    def update_virtual_queue(self):
        round_energy = sum(self.client_energies[k] for k in self.selected_clients)
        self.energy_queue = max(self.energy_queue + round_energy - self.energy_per_round, 0)
        self.energy_consumption.append(round_energy)

    def run_round(self, t):
        self.round = t
        self.client_gradients.clear()
        self.client_energies.clear()
        self.ready_clients.clear()
        self.round_complete.clear()

        self.selected_clients = self.select_clients_dpp()

        for k in self.selected_clients:
            self.clients[k].receive_global_model(self.model, self.clients[k].staleness)

        self.round_complete.wait()

        agg_grad = self.aggregate_gradients()
        self.update_global_model(agg_grad)
        self.update_virtual_queue()

        print(f"[Server] Round {t} complete. Selected clients: {self.selected_clients}. Energy queue: {self.energy_queue:.2f}")

    def evaluate_global_model(self):
        self.model.eval()
        return 0.0  # Placeholder for actual evaluation

    def report_metrics(self):
        return self.training_times, self.energy_consumption
