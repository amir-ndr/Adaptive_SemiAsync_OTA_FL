import torch
import numpy as np
import copy
import time
import logging
import random
from typing import Tuple, Optional, Dict, List
from torch.utils.data import Subset, DataLoader

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class SAFAClient:
    def __init__(self, client_id: int, data_indices: list, model: torch.nn.Module,
                 fk: float, mu_k: float, P_max: float, C: float, Ak: int,
                 train_dataset: torch.utils.data.Dataset, device: str = 'cpu',
                 local_epochs: int = 1, crash_prob: float = 0.1):
        self.client_id = client_id
        self.data_indices = list(data_indices)
        self.original_model = copy.deepcopy(model).to(device)
        self.local_model = copy.deepcopy(model).to(device)
        self.device = device
        self.local_epochs = local_epochs
        
        # Hardware parameters
        self.fk = float(fk)
        self.mu_k = float(mu_k)
        self.C = float(C)
        self.Ak = int(Ak)
        
        # Communication parameters - now OTA-style
        self.P_max = float(P_max)
        self.h_t_k = None  # Channel coefficient
        self.tx_power = 0.0  # Transmission power
        
        # Training data
        self.train_dataset = train_dataset
        
        # SAFA state tracking
        self.version = 0
        self.crash_prob = min(max(float(crash_prob), 0), 1)
        self.last_selected_round = -1
        self.rounds_missed = 0
        self.staleness = 0
        self.state = "active"
        self.ready = True
        self.cached_update = None
        
        # Training artifacts
        self.last_update = None
        self.update_norm = 0.0
        self.last_comp_time = 0.0
        
        # Energy tracking - now OTA-style
        self.total_energy = 0.0
        self.computation_energy = 0.0
        self.communication_energy = 0.0
        
        logger.info(f"SAFA Client {client_id} initialized | "
                   f"CPU: {fk/1e9:.2f}GHz | "
                   f"Crash: {crash_prob:.1%} | "
                   f"Data: {len(data_indices)} samples")

    def update_model(self, model_state_dict: dict, new_version: int) -> None:
        self.original_model.load_state_dict(model_state_dict)
        self.local_model.load_state_dict(model_state_dict)
        old_version = self.version
        self.version = new_version
        self.staleness = 0
        logger.debug(f"Client {self.client_id} updated v{old_version}→v{new_version}")

    def get_state(self, global_version: int, tau: int) -> str:
        lag = global_version - self.version
        if lag == 0:
            return "up-to-date"
        elif lag > tau:
            return "deprecated"
        else:
            return "tolerable"

    def compute_update(self, current_global_version: int) -> Tuple[Optional[torch.Tensor], float, bool]:
        self.staleness = current_global_version - self.version
        self.last_update = None
        self.update_norm = 0.0
        start_time = time.time()
        
        try:
            if random.random() < self.crash_prob:
                raise RuntimeError("Pre-computation crash")
            
            if not self.data_indices:
                self.last_update = torch.zeros(self._model_dimension(), device=self.device)
                return self.last_update, 0.0, True
            
            initial_state = copy.deepcopy(self.local_model.state_dict())
            
            subset = Subset(self.train_dataset, self.data_indices)
            loader = DataLoader(subset, batch_size=self.Ak, shuffle=True)
            
            optimizer = torch.optim.Adam(self.local_model.parameters(), lr=0.001)
            
            for epoch in range(self.local_epochs):
                for images, labels in loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    
                    if random.random() < self.crash_prob/10:
                        raise RuntimeError(f"Mid-batch crash @ epoch {epoch+1}")
                    
                    optimizer.zero_grad()
                    outputs = self.local_model(images)
                    loss = torch.nn.functional.cross_entropy(outputs, labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), 1.0)
                    optimizer.step()
            
            current_state = self.local_model.state_dict()
            model_delta = []
            for key in initial_state:
                model_delta.append((current_state[key] - initial_state[key]).view(-1))
            model_delta = torch.cat(model_delta)
            
            self.last_update = model_delta
            self.update_norm = torch.norm(model_delta).item()
            self.last_comp_time = time.time() - start_time
            
            # Computation energy - same as OTA system
            samples_processed = self.Ak * self.local_epochs
            total_flops = samples_processed * self.C
            self.computation_energy = self.mu_k * total_flops * (self.fk**2)
            
            self.version += 1
            
            logger.info(f"Client {self.client_id} update ready | "
                       f"Norm: {self.update_norm:.4f} | "
                       f"Time: {self.last_comp_time:.4f}s | "
                       f"Comp Energy: {self.computation_energy:.4f}J")
            
            return self.last_update, self.last_comp_time, True
            
        except RuntimeError as e:
            logger.error(f"Client {self.client_id} failed: {str(e)}")
            self.state = "crashed"
            return None, 0.0, False

    def transmit_update(self) -> Tuple[float, bool]:
        """Transmit update with OTA-style energy calculation"""
        if self.state == "crashed" or self.last_update is None:
            return 0.0, False
            
        try:
            if random.random() < self.crash_prob/2:
                raise RuntimeError("Transmission failure")
            
            # OTA-STYLE ENERGY CALCULATION - matches your semi-async OTA paper
            # E_com = ||s_k||^2 = (p_k)^2 * ||δ_k||^2 / |h_k|^2
            delta_norm_sq = torch.norm(self.last_update).item() ** 2
            self.communication_energy = (self.tx_power ** 2) * delta_norm_sq / abs(self.h_t_k) ** 2
            self.total_energy = self.computation_energy + self.communication_energy
            
            # Transmission time for latency (not used in energy calculation)
            model_size_bits = self.last_update.numel() * 32
            bandwidth = 20e6  # 20 Mbps
            tx_time = model_size_bits / bandwidth
            
            logger.info(f"Client {self.client_id} transmitted | "
                        f"Comm Energy: {self.communication_energy:.4e}J | "
                        f"Power: {self.tx_power:.4f}W | Gain: {abs(self.h_t_k):.4f}")
            
            return tx_time, True
            
        except RuntimeError as e:
            logger.error(f"Client {self.client_id} transmission failed: {str(e)}")
            self.state = "crashed"
            return 0.0, False

    def configure_transmission(self, h_t_k: complex, tx_power: float):
        """Set transmission parameters for current round"""
        self.h_t_k = h_t_k
        self.tx_power = min(float(tx_power), self.P_max)
        logger.debug(f"Client {self.client_id} TX config | "
                     f"Power: {self.tx_power:.4f}W | |h|: {abs(h_t_k):.4f}")

    def cache_update(self, update: torch.Tensor) -> None:
        self.cached_update = update.clone()
        logger.debug(f"Client {self.client_id} cached update")

    def get_cached_update(self) -> Optional[torch.Tensor]:
        return self.cached_update.clone() if self.cached_update is not None else None

    def _model_dimension(self) -> int:
        return sum(p.numel() for p in self.local_model.parameters())

    def reset_for_round(self) -> None:
        self.state = "active"
        self.computation_energy = 0.0
        self.communication_energy = 0.0
        self.total_energy = 0.0
        logger.debug(f"Client {self.client_id} reset for new round")

    def mark_round_missed(self) -> None:
        self.rounds_missed += 1
        logger.debug(f"Client {self.client_id} missed round (total {self.rounds_missed})")

    def get_priority_score(self) -> float:
        return 1.0 + self.rounds_missed * 0.5

    def get_energy_report(self) -> Dict[str, float]:
        return {
            "total": float(self.total_energy),
            "computation": float(self.computation_energy),
            "communication": float(self.communication_energy)
        }

    def get_client_status(self) -> Dict[str, float]:
        return {
            "version": int(self.version),
            "staleness": int(self.staleness),
            "update_norm": float(self.update_norm),
            "last_comp_time": float(self.last_comp_time),
            "state": str(self.state),
            "rounds_missed": int(self.rounds_missed)
        }

class SAFAServer:
    def __init__(self, global_model: torch.nn.Module, clients: List[SAFAClient],
                 V: float = 1.0, sigma_n: float = 0.1, tau: int = 5,
                 T_max: float = 100, E_max: float = 1.0, T_total_rounds: int = 50,
                 device: str = 'cpu'):
        self.global_model = global_model
        self.clients = clients
        self.device = device
        self.V = V
        self.sigma_n = sigma_n
        self.tau = tau
        self.T_max = T_max
        self.T_total_rounds = T_total_rounds
        self.d = self._get_model_dimension()
        
        # Energy and resource tracking - same as OTA system
        self.total_energy_per_round = []
        self.cumulative_energy_per_client = {c.client_id: 0.0 for c in clients}
        self.Q_e = {c.client_id: 0.0 for c in clients}
        self.Q_time = 0.0
        
        # History tracking
        self.selected_history = []
        self.accuracy_history = []
        self.round_durations = []
        self.global_version = 0
        
        logger.info(f"SAFA Server initialized | Clients: {len(clients)} | "
                    f"Model dim: {self.d} | Rounds: {T_total_rounds}")

    def _get_model_dimension(self) -> int:
        return sum(p.numel() for p in self.global_model.parameters())
    
    def select_clients(self) -> List[SAFAClient]:
        """Select clients with OTA-style channel and power setup"""
        # Classify clients
        up_to_date = []
        tolerable = []
        compensatory = []
        
        for client in self.clients:
            state = client.get_state(self.global_version, self.tau)
            if state == "up-to-date":
                up_to_date.append(client)
            elif state == "tolerable":
                tolerable.append(client)
            elif state == "deprecated":
                compensatory.append(client)
        
        # Selection strategy
        selected = []
        if up_to_date:
            selected = random.sample(up_to_date, min(3, len(up_to_date)))
        elif tolerable:
            selected = random.sample(tolerable, min(3, len(tolerable)))
        else:
            compensatory.sort(key=lambda c: c.get_priority_score(), reverse=True)
            selected = compensatory[:min(3, len(compensatory))]
        
        # Configure transmission with OTA-style parameters
        for client in selected:
            # Generate channel gain (same Rayleigh fading as OTA)
            magnitude = np.random.rayleigh(scale=1/np.sqrt(2))
            phase = np.random.uniform(0, 2*np.pi)
            h_t_k = magnitude * np.exp(1j * phase)
            
            # Set power using similar policy as OTA
            tx_power = self._compute_power_for_client(client, abs(h_t_k))
            client.configure_transmission(h_t_k, tx_power)
        
        logger.info(f"Selected {len(selected)} clients: "
                   f"{[c.client_id for c in selected]}")
        return selected

    def _compute_power_for_client(self, client: SAFAClient, channel_gain: float) -> float:
        """Compute power using same policy as OTA system"""
        Q_e = self.Q_e[client.client_id]
        # Use last known update norm (or default to 1.0)
        update_norm = client.update_norm if client.update_norm > 0 else 1.0
        
        # Same energy-cost coefficient as OTA system
        c_k = Q_e * (update_norm ** 2) / (channel_gain ** 2 + 1e-8)
        
        # Simple proportional allocation
        if c_k < 1e-8:
            return min(0.1, client.P_max)
        
        return min(client.P_max, 0.5 / np.sqrt(c_k))

    def broadcast_model(self, selected_clients: List[SAFAClient]) -> None:
        global_state = self.global_model.state_dict()
        for client in selected_clients:
            client.update_model(global_state, self.global_version)
        logger.info(f"Broadcast global model v{self.global_version}")

    def aggregate_updates(self, updates: List[Tuple[int, torch.Tensor]]) -> torch.Tensor:
        if not updates:
            return torch.zeros(self.d, device=self.device)
        
        total_weight = 0.0
        weighted_update = torch.zeros(self.d, device=self.device)
        
        for client_id, delta in updates:
            staleness = self.global_version - self.clients[client_id].version
            weight = 0.9 ** staleness  # Staleness discount
            weighted_update += weight * delta
            total_weight += weight
        
        if total_weight > 1e-8:
            return weighted_update / total_weight
        return weighted_update

    def update_global_model(self, aggregated_delta: torch.Tensor) -> None:
        with torch.no_grad():
            params = torch.nn.utils.parameters_to_vector(self.global_model.parameters())
            params += aggregated_delta
            torch.nn.utils.vector_to_parameters(params, self.global_model.parameters())
        self.global_version += 1
        logger.info(f"Global model updated to v{self.global_version}")

    def update_queues(self, selected: List[SAFAClient], D_t: float) -> None:
        """Update queues with same logic as OTA system"""
        round_energy = 0.0
        for client in selected:
            energy_report = client.get_energy_report()
            client_energy = energy_report["total"]
            round_energy += client_energy
            self.cumulative_energy_per_client[client.client_id] += client_energy
            
            # Per-round energy budget
            per_round_budget = self.E_max[client.client_id] / self.T_total_rounds
            energy_increment = client_energy - per_round_budget
            
            # Update energy queue
            self.Q_e[client.client_id] = max(0, self.Q_e[client.client_id] + energy_increment)
            
            logger.info(f"  Client {client.client_id} energy: "
                       f"{client_energy:.4e}J | ΔQ: {energy_increment:.4e} | "
                       f"New Q_e: {self.Q_e[client.client_id]:.2f}")

        self.total_energy_per_round.append(round_energy)
        
        # Update time queue
        per_round_time_budget = self.T_max / self.T_total_rounds
        time_increment = D_t - per_round_time_budget
        self.Q_time = max(0, self.Q_time + time_increment)
        
        logger.info(f"Time queue update: Δ={time_increment:.4f} | "
                   f"New Q_time: {self.Q_time:.2f}")

    def run_round(self, test_loader) -> float:
        round_start = time.time()
        
        # 1. Select clients and broadcast model
        selected = self.select_clients()
        self.broadcast_model(selected)
        
        # 2. Compute updates on clients
        updates = []
        comp_times = []
        for client in selected:
            delta, comp_time, success = client.compute_update(self.global_version)
            if success and delta is not None:
                updates.append((client.client_id, delta))
                comp_times.append(comp_time)
        
        # 3. Transmit updates to server
        tx_times = []
        for client in selected:
            if client.last_update is not None:
                tx_time, success = client.transmit_update()
                tx_times.append(tx_time)
        
        # 4. Aggregate updates and update global model
        aggregated = self.aggregate_updates(updates)
        self.update_global_model(aggregated)
        
        # 5. Calculate round duration (max computation + max transmission)
        max_comp = max(comp_times) if comp_times else 0
        max_tx = max(tx_times) if tx_times else 0
        D_t = max_comp + max_tx
        
        # 6. Update queues
        self.update_queues(selected, D_t)
        
        # 7. Evaluate model
        accuracy = self.evaluate_model(test_loader)
        self.accuracy_history.append(accuracy)
        
        # 8. Update client states
        for client in self.clients:
            if client in selected:
                client.reset_for_round()
            else:
                client.mark_round_missed()
        
        # Record metrics
        round_time = time.time() - round_start
        self.round_durations.append(D_t)
        self.selected_history.append([c.client_id for c in selected])
        
        logger.info(f"Round completed in {D_t:.2f}s | "
                   f"Accuracy: {accuracy:.2f}% | "
                   f"Global version: {self.global_version}")
        return accuracy

    def evaluate_model(self, test_loader) -> float:
        self.global_model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.global_model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return 100 * correct / total

    def get_metrics(self) -> Dict:
        return {
            "accuracy": self.accuracy_history,
            "round_durations": self.round_durations,
            "selected_clients": self.selected_history,
            "energy_per_round": self.total_energy_per_round,
            "cumulative_energy": self.cumulative_energy_per_client,
            "energy_queues": copy.deepcopy(self.Q_e),
            "time_queue": self.Q_time,
            "global_version": self.global_version
        }
    

import torch
import numpy as np
import time
from torch.utils.data import DataLoader
# from safa_client import SAFAClient
# from safa_server import SAFAServer
from model import CNNMnist
from dataloader import load_mnist, partition_mnist_noniid
import matplotlib.pyplot as plt
import logging
from collections import defaultdict

def main():
    # ===== Enhanced Logging Configuration =====
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("safa_fl_system.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting SAFA Federated Learning Simulation")

    # ========== Experiment Configuration ==========
    NUM_CLIENTS = 10
    NUM_ROUNDS = 300
    BATCH_SIZE = 32
    LOCAL_EPOCHS = 1
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    CRASH_PROB = 0.15
    LAG_TOLERANCE = 3
    SELECT_FRAC = 0.5
    EVAL_INTERVAL = 5
    
    logger.info(f"\n=== Configuration ==="
                f"\nClients: {NUM_CLIENTS}"
                f"\nRounds: {NUM_ROUNDS}"
                f"\nBatch Size: {BATCH_SIZE}"
                f"\nLocal Epochs: {LOCAL_EPOCHS}"
                f"\nDevice: {DEVICE}"
                f"\nCrash Probability: {CRASH_PROB:.0%}"
                f"\nStaleness Tolerance: {LAG_TOLERANCE}"
                f"\nSelection Fraction: {SELECT_FRAC:.0%}"
                f"\nEvaluation Interval: {EVAL_INTERVAL} rounds")

    # ========== Data Preparation ==========
    train_dataset, test_dataset = load_mnist()
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    client_data_map = partition_mnist_noniid(train_dataset, NUM_CLIENTS)

    # ========== Client Initialization ==========
    clients = []
    for cid in range(NUM_CLIENTS):
        indices = client_data_map[cid]
        if len(indices) == 0:
            indices = [0]  # Ensure at least one sample
            logger.warning(f"Client {cid} received dummy data")

        # Hardware diversity parameters
        cpu_freq = np.random.uniform(1e9, 2e9)  # 1-2 GHz
        crash_prob = CRASH_PROB * np.random.uniform(0.8, 1.2)
        
        clients.append(
            SAFAClient(
                client_id=cid,
                data_indices=indices,
                model=CNNMnist(),
                fk=cpu_freq,
                mu_k=1e-27,                      # Realistic energy coefficient
                P_max=0.2,                       
                C=1e6,                           # FLOPs per sample
                Ak=BATCH_SIZE,                   
                train_dataset=train_dataset,
                device=DEVICE,
                local_epochs=LOCAL_EPOCHS,
                crash_prob=crash_prob
            )
        )
        logger.info(f"Client {cid} initialized | "
                   f"Samples: {len(indices)} | "
                   f"CPU: {cpu_freq/1e9:.2f}GHz | "
                   f"Crash Prob: {crash_prob:.1%}")

    # ========== Server Initialization ==========
    global_model = CNNMnist().to(DEVICE)
    server = SAFAServer(
        global_model=global_model,
        clients=clients,
        lag_tolerance=LAG_TOLERANCE,
        select_frac=SELECT_FRAC,
        learning_rate=0.01,  # ADDED: Critical for model updates
        device=DEVICE
    )

    # ========== Enhanced Metrics Tracking ==========
    metrics = {
        'round_durations': [],
        'energy_consumption': [],
        'crashed_counts': [],
        'effective_updates': [],
        'selected_counts': [],
        'client_selections': defaultdict(int),
        'accuracies': [],
        'staleness': [],
        'comm_ratio': [],
        'rounds': list(range(NUM_ROUNDS))
    }

    # ========== Training Loop ==========
    logger.info("\n=== Starting Training ===")
    for round_idx in range(NUM_ROUNDS):
        round_start = time.time()
        logger.info(f"\n--- Round {round_idx+1}/{NUM_ROUNDS} ---")
        
        # Run federated round
        eval_this_round = (round_idx % EVAL_INTERVAL == 0) or (round_idx == NUM_ROUNDS - 1)
        round_duration = server.run_round(
            test_loader=test_loader if eval_this_round else None
        )

        # Record metrics
        metrics['round_durations'].append(round_duration)
        
        # Energy metrics - FIXED indexing
        if server.energy_history and round_idx < len(server.energy_history):
            round_energy = server.energy_history[round_idx]
            total_energy = sum(e['total'] for e in round_energy) if round_energy else 0
            comm_energy = sum(e['communication'] for e in round_energy) if round_energy else 0
            
            metrics['energy_consumption'].append(total_energy)
            metrics['comm_ratio'].append(comm_energy / total_energy if total_energy > 0 else 0)
        else:
            metrics['energy_consumption'].append(0)
            metrics['comm_ratio'].append(0)
            
        # Client status metrics - FIXED
        metrics['crashed_counts'].append(
            sum(1 for c in clients if c.state == "crashed")
        )
        
        # Selection metrics - FIXED indexing
        if server.selection_history and round_idx < len(server.selection_history):
            selected_ids = server.selection_history[round_idx]
            metrics['selected_counts'].append(len(selected_ids))
            
            # Update client selection counts
            for cid in selected_ids:
                metrics['client_selections'][cid] += 1
        else:
            metrics['selected_counts'].append(0)
        
        # Effective updates - FIXED
        metrics['effective_updates'].append(
            len(server.energy_history[round_idx]) if server.energy_history and round_idx < len(server.energy_history) else 0
        )
        
        # Staleness - FIXED indexing
        if server.staleness_history and round_idx < len(server.staleness_history):
            metrics['staleness'].append(server.staleness_history[round_idx])
        else:
            metrics['staleness'].append(0)

        # Accuracy tracking - FIXED
        if eval_this_round and server.accuracy_history:
            metrics['accuracies'].append(server.accuracy_history[-1])

        # Periodic logging
        if (round_idx + 1) % 10 == 0 or round_idx == 0:
            logger.info(
                f"Round {round_idx+1:03d}/{NUM_ROUNDS} | "
                f"Duration: {round_duration:.2f}s | "
                f"Selected: {metrics['selected_counts'][-1]} | "
                f"Crashed: {metrics['crashed_counts'][-1]} | "
                f"Effective: {metrics['effective_updates'][-1]} | "
                f"Energy: {metrics['energy_consumption'][-1]:.2f}J | "
                f"Staleness: {metrics['staleness'][-1]:.2f}"
            )

    # ========== Final Evaluation ==========
    final_acc = server.evaluate(test_loader)
    metrics['accuracies'].append(final_acc)
    
    logger.info(f"\n=== Training Complete ==="
                f"\nFinal Accuracy: {final_acc:.2f}%"
                f"\nTotal Energy: {sum(metrics['energy_consumption']):.2f}J"
                f"\nAvg Round Time: {np.mean(metrics['round_durations']):.2f}s"
                f"\nAvg Staleness: {np.mean(metrics['staleness']):.2f}")

    # ========== Enhanced Visualization ==========
    plt.figure(figsize=(18, 15))
    
    # 1. Accuracy Progress
    plt.subplot(321)
    eval_rounds = [EVAL_INTERVAL*i for i in range(len(metrics['accuracies'])-1)] + [NUM_ROUNDS]
    plt.plot(eval_rounds, metrics['accuracies'], 'o-')
    plt.title("Test Accuracy Progress")
    plt.xlabel("Communication Rounds")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)

    # 2. Client Participation
    plt.subplot(322)
    plt.plot(metrics['rounds'], metrics['selected_counts'], label='Selected')
    plt.plot(metrics['rounds'], metrics['effective_updates'], label='Effective')
    plt.title("Client Participation per Round")
    plt.xlabel("Rounds")
    plt.ylabel("Number of Clients")
    plt.legend()
    plt.grid(True)

    # 3. Energy Consumption
    plt.subplot(323)
    plt.plot(metrics['rounds'], metrics['energy_consumption'])
    plt.title("Energy Consumption per Round")
    plt.xlabel("Rounds")
    plt.ylabel("Energy (Joules)")
    plt.grid(True)

    # 4. Round Duration
    plt.subplot(324)
    plt.plot(metrics['rounds'], metrics['round_durations'])
    plt.title("Round Duration")
    plt.xlabel("Rounds")
    plt.ylabel("Time (seconds)")
    plt.grid(True)

    # 5. System Health
    plt.subplot(325)
    plt.plot(metrics['rounds'], metrics['staleness'], 'b-', label='Avg Staleness')
    plt.plot(metrics['rounds'], metrics['crashed_counts'], 'r-', label='Crashed Clients')
    plt.title("System Health Metrics")
    plt.xlabel("Rounds")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True)

    # 6. Client Selection Distribution
    plt.subplot(326)
    plt.bar(range(NUM_CLIENTS), [metrics['client_selections'][cid] for cid in range(NUM_CLIENTS)])
    plt.title("Client Selection Distribution")
    plt.xlabel("Client ID")
    plt.ylabel("Times Selected")
    plt.xticks(range(NUM_CLIENTS))
    plt.grid(True)

    plt.suptitle("SAFA Federated Learning Performance Metrics", fontsize=16, y=0.99)
    plt.tight_layout(pad=3.0)
    plt.savefig("safa_fl_results.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Energy Breakdown
    plt.figure(figsize=(12, 6))
    plt.plot(metrics['rounds'], metrics['comm_ratio'])
    plt.title("Communication Energy Ratio per Round")
    plt.xlabel("Rounds")
    plt.ylabel("Comm/Total Energy Ratio")
    plt.grid(True)
    plt.savefig("safa_energy_breakdown.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Client Statistics
    client_stats = server.get_client_stats()
    print("\n=== Client Statistics ===")
    print(f"{'ID':<5}{'Selected':<10}{'Missed':<10}{'Staleness':<12}{'Version':<10}{'State':<12}")
    for cid in range(NUM_CLIENTS):
        stats = client_stats.get(cid, {})
        print(f"{cid:<5}"
              f"{stats.get('selected_count',0):<10}"
              f"{stats.get('rounds_missed',0):<10}"
              f"{stats.get('staleness',0):<12}"
              f"{stats.get('version',0):<10}"
              f"{stats.get('state','unknown'):<12}")

if __name__ == "__main__":
    main()