import torch
import numpy as np
import copy
import time
import logging
import random
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class SAFAServer:
    def __init__(self, global_model: torch.nn.Module, 
                 clients: List,
                 lag_tolerance: int = 3,
                 select_frac: float = 0.4,
                 learning_rate: float = 0.1,  # Added learning rate
                 device: str = 'cpu'):
        """
        Fixed SAFA Server implementation with proper aggregation and model updating
        
        Args:
            global_model: Initial global model architecture
            clients: List of SAFAClient instances
            lag_tolerance: Maximum allowed staleness (τ)
            select_frac: Fraction of clients to select per round
            learning_rate: Global learning rate for model updates
            device: Computation device ('cpu' or 'cuda')
        """
        self.global_model = global_model.to(device)
        self.clients = {c.client_id: c for c in clients}
        self.device = device
        self.lag_tolerance = max(1, int(lag_tolerance))
        self.select_frac = min(max(0.1, float(select_frac)), 1.0)
        self.learning_rate = learning_rate
        
        # Version and state tracking
        self.global_version = 0
        self.round_idx = 0
        
        # Caching system
        self.cache = {}  # {client_id: model_parameters}
        self.bypass = {}  # {client_id: model_parameters} for undrafted clients
        
        # History tracking
        self.selection_history = []
        self.energy_history = []
        self.staleness_history = []
        self.accuracy_history = []
        self.round_times = []
        self.client_stats = defaultdict(dict)
        
        # logger.info(f"SAFA Server initialized | "
        #            f"Clients: {len(clients)} | "
        #            f"τ: {lag_tolerance} | "
        #            f"Select: {select_frac:.0%} | "
        #            f"LR: {learning_rate}")

    def classify_clients(self) -> Tuple[list, list, list]:
        """
        Classify clients into states based on SAFA definitions:
        - Up-to-date: Version matches global
        - Tolerable: Version lag <= τ
        - Deprecated: Version lag > τ
        """
        up_to_date = []
        tolerable = []
        deprecated = []
        
        for client in self.clients.values():
            lag = self.global_version - client.version
            if lag == 0:
                up_to_date.append(client)
            elif lag <= self.lag_tolerance:
                tolerable.append(client)
            else:
                deprecated.append(client)
                
        # logger.info(f"Client classification (v{self.global_version}) | "
        #            f"Up-to-date: {len(up_to_date)} | "
        #            f"Tolerable: {len(tolerable)} | "
        #            f"Deprecated: {len(deprecated)}")
        
        return up_to_date, tolerable, deprecated

    def distribute_models(self, clients: List) -> None:
        """Synchronize specified clients with current global model"""
        global_state = self.global_model.state_dict()
        for client in clients:
            client.update_model(global_state, self.global_version)
            logger.debug(f"Distributed v{self.global_version} to client {client.client_id}")

    def cfcfm_selection(self, updates: Dict[int, Tuple[float, torch.Tensor]]) -> Tuple[List[int], List[int]]:
        """
        Compensatory First-Come-First-Merge selection (SAFA Algorithm 1)
        """
        # Separate clients by selection history
        priority_updates = []
        secondary_updates = []
        
        for client_id, (arrival_time, update) in updates.items():
            client = self.clients[client_id]
            if client.last_selected_round < self.round_idx - 1:  # Missed last round
                priority_updates.append((arrival_time, client_id, update))
            else:
                secondary_updates.append((arrival_time, client_id, update))
        
        # Sort both groups by arrival time
        priority_updates.sort(key=lambda x: x[0])
        secondary_updates.sort(key=lambda x: x[0])
        
        # Determine selection quota
        quota = max(1, int(self.select_frac * len(self.clients)))
        selected = []
        undrafted = []
        
        # First select from priority group
        for arrival_time, client_id, update in priority_updates:
            if len(selected) < quota:
                selected.append(client_id)
            else:
                undrafted.append(client_id)
        
        # Then from secondary group if quota not met
        if len(selected) < quota:
            for arrival_time, client_id, update in secondary_updates:
                if len(selected) < quota:
                    selected.append(client_id)
                else:
                    undrafted.append(client_id)
        else:
            undrafted.extend([cid for _, cid, _ in secondary_updates])
        
        for client_id in selected:
            client = self.clients[client_id]
            # Basic water-filling power allocation
            client.tx_power = min(
                client.P_max, 
                0.1 / (abs(client.h_t_k) + 1e-8)  # Inverse to channel quality
            )
        
        # Update client selection status
        for client_id in selected:
            self.clients[client_id].last_selected_round = self.round_idx
            self.clients[client_id].rounds_missed = 0
        for client_id in undrafted:
            self.clients[client_id].mark_round_missed()
        
        # logger.info(f"CFCFM selection complete | "
        #            f"Selected: {len(selected)} | "
        #            f"Undrafted: {len(undrafted)}")
        
        return selected, undrafted

    def discriminative_aggregation(self, selected: List[int]) -> None:
        """
        Three-step discriminative aggregation with proper model updating
        """
        # Step 1: Pre-aggregation cache update
        current_global_flat = self._flatten_params(self.global_model.state_dict())
        
        for client_id, client in self.clients.items():
            if client_id in selected:
                # Picked clients: update cache with their new model parameters
                if client.last_update is not None:
                    # Client sends delta, so convert to full model: w_k = w_global + delta
                    client_model = current_global_flat + client.last_update
                    self.cache[client_id] = client_model
            elif self.global_version - client.version > self.lag_tolerance:
                # Deprecated clients: reset to global model
                self.cache[client_id] = current_global_flat
        
        # Step 2: SAFA aggregation (Eq 7) - weighted average of models
        total_samples = 0
        weighted_sum = None
        
        for client_id, model_params in self.cache.items():
            client = self.clients[client_id]
            nk = len(client.data_indices)
            total_samples += nk
            
            if weighted_sum is None:
                weighted_sum = nk * model_params
            else:
                weighted_sum += nk * model_params
        
        if total_samples == 0:
            logger.error("Aggregation failed: total_samples=0")
            return
            
        # Compute new global model
        new_global_flat = weighted_sum / total_samples
        
        # Unflatten and update global model
        new_state = self._unflatten_params(new_global_flat, self.global_model.state_dict())
        self.global_model.load_state_dict(new_state)
        self.global_version += 1
        
        # Step 3: Post-aggregation cache update
        for client_id in selected:
            if client_id in self.clients and self.clients[client_id].last_update is not None:
                # Keep selected client's model in cache
                self.cache[client_id] = current_global_flat + self.clients[client_id].last_update
        
        # logger.info(f"Aggregation complete | Global v{self.global_version}")

    def run_round(self, test_loader: Optional[torch.utils.data.DataLoader] = None) -> float:
        """Execute a complete SAFA round"""
        round_start = time.time()
        self.round_idx += 1
        # logger.info(f"\n=== SAFA Round {self.round_idx} (Global v{self.global_version}) ===")
        
        # 1. Classify clients and distribute models
        up_to_date, tolerable, deprecated = self.classify_clients()
        self.distribute_models(up_to_date + deprecated)
        
        # 2. Perform local training on all active clients
        updates = {}  # client_id: (arrival_time, update)
        comp_times = []
        
        for client in list(self.clients.values()):
            client.reset_for_round()
            client.set_channel_gain()
            
            # Perform local training
            update, comp_time, success = client.compute_update(self.global_version)
            if not success or update is None:
                continue  # Skip crashed clients
                
            # Simulate transmission and get arrival time
            tx_time, tx_success = client.transmit_update(h_k=client.h_t_k)
            if not tx_success:
                continue
                
            # Simulate arrival time (computation + transmission + random network delay)
            arrival_time = comp_time + tx_time + random.uniform(0, 0.5)
            updates[client.client_id] = (arrival_time, update)
            comp_times.append(comp_time + tx_time)
        
        # 3. Post-training client selection (CFCFM)
        selected_ids, undrafted_ids = self.cfcfm_selection(updates)
        self.selection_history.append(selected_ids)
        
        # 4. Update bypass cache with undrafted updates
        for client_id in undrafted_ids:
            if client_id in updates:
                self.clients[client_id].cache_update(updates[client_id][1])
        
        # 5. Perform discriminative aggregation
        self.discriminative_aggregation(selected_ids)
        
        # 6. Collect metrics and energy reports
        round_energy = []
        for client_id in selected_ids:
            if client_id in self.clients:
                client = self.clients[client_id]
                round_energy.append({
                    "client_id": client_id,
                    "total": client.total_energy,
                    "computation": client.computation_energy,
                    "communication": client.communication_energy
                })
        self.energy_history.append(round_energy)
        
        # Calculate average staleness
        staleness_sum = 0
        active_count = 0
        for client in self.clients.values():
            if client.version > 0:
                staleness_sum += self.global_version - client.version
                active_count += 1
        avg_staleness = staleness_sum / active_count if active_count else 0
        self.staleness_history.append(avg_staleness)
        
        # 7. Evaluation
        if test_loader is not None:
            acc = self.evaluate(test_loader)
            self.accuracy_history.append(acc)
            # logger.info(f"Test accuracy: {acc:.2f}%")
            return acc, time.time() - round_start  # Return actual duration
        
        # if test_loader is not None:
        #     acc = self.evaluate(test_loader)
        #     self.accuracy_history.append(acc)
        
        # 8. Final logging
        round_duration = time.time() - round_start
        self.round_times.append(round_duration)
        
        # logger.info(f"Round {self.round_idx} completed | "
        #            f"Duration: {round_duration:.2f}s | "
        #            f"Selected: {len(selected_ids)} | "
        #            f"Avg staleness: {avg_staleness:.2f}")
        
        return round_duration

    def evaluate(self, test_loader: torch.utils.data.DataLoader) -> float:
        """Evaluate global model on test set"""
        self.global_model.eval()
        correct, total = 0, 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.global_model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        return accuracy

    def _flatten_params(self, state_dict: dict) -> torch.Tensor:
        """Flatten model parameters into 1D tensor"""
        return torch.cat([p.view(-1) for p in state_dict.values()])

    def _unflatten_params(self, flat_params: torch.Tensor, 
                         ref_state_dict: dict) -> dict:
        """Unflatten parameters into model state_dict"""
        new_state = {}
        offset = 0
        for k, v in ref_state_dict.items():
            numel = v.numel()
            new_state[k] = flat_params[offset:offset+numel].view_as(v)
            offset += numel
        return new_state

    def get_energy_summary(self) -> Dict[str, List[float]]:
        """Get comprehensive energy consumption statistics"""
        per_round_total = []
        per_round_mean = []
        per_round_comm_ratio = []
        
        for round_energy in self.energy_history:
            if not round_energy:
                per_round_total.append(0.0)
                per_round_mean.append(0.0)
                per_round_comm_ratio.append(0.0)
                continue
                
            total_energy = sum(e["total"] for e in round_energy)
            comm_energy = sum(e["communication"] for e in round_energy)
            
            per_round_total.append(total_energy)
            per_round_mean.append(total_energy / len(round_energy))
            per_round_comm_ratio.append(comm_energy / total_energy if total_energy > 0 else 0)
        
        return {
            "per_round_total": per_round_total,
            "per_round_mean": per_round_mean,
            "cumulative": np.cumsum(per_round_total).tolist(),
            "comm_ratio": per_round_comm_ratio
        }

    def get_client_stats(self) -> Dict[int, Dict[str, float]]:
        """Get statistics for each client"""
        stats = {}
        for client_id, client in self.clients.items():
            stats[client_id] = {
                "version": client.version,
                "staleness": self.global_version - client.version,
                "selected_count": sum(1 for rnd in self.selection_history 
                                    if client_id in rnd),
                "rounds_missed": client.rounds_missed,
                "last_selected": client.last_selected_round,
                "state": client.state
            }
        return stats