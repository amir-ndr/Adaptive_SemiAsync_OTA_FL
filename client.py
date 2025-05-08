import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import threading
import time

class Client(threading.Thread):
    def __init__(self, client_id, model_fn, dataset, dataloader, device, local_lr=0.01):
        super(Client, self).__init__()
        self.client_id = client_id
        self.device = device
        self.model_fn = model_fn
        self.model = model_fn().to(device)
        self.dataset = dataset
        self.dataloader = dataloader
        self.lock = threading.Lock()

        # Client-specific characteristics
        self.local_lr = local_lr
        self.f_k = np.random.uniform(1e8, 3e8)  # Computational capability (FLOPS)
        self.p_k = np.random.uniform(0.1, 5.0)  # Transmit power (W)
        self.mu_k = 1e-28  # CPU efficiency coefficient
        self.C = np.random.uniform(5e5, 2e6)  # FLOPs per sample
        
        # Wireless channel characteristics (match model parameter dimensions)
        model_params_size = sum(p.numel() for p in self.model.parameters())
        self.h_k = torch.randn(model_params_size, dtype=torch.cfloat, device=device)
        
        # Training state management
        self.staleness = 0
        self.received_model = None
        self.selected = False
        self.grad_ready_event = threading.Event()
        self.local_gradient = None
        
        # Callbacks and lifecycle management
        self.notify_done_callback = None
        self.send_gradient_callback = None
        self.alive = True

    def receive_global_model(self, global_model, staleness):
        """Update client with fresh global model and reset training state"""
        with self.lock:
            self.staleness = staleness
            self.received_model = global_model.to(self.device)
            self.model.load_state_dict(global_model.state_dict())
            self.selected = True

    def estimated_training_time(self):
        """Calculate remaining local computation time"""
        return (self.C * len(self.dataloader.dataset)) / self.f_k

    def estimated_energy(self):
        """Calculate estimated total energy consumption"""
        with self.lock:
            comp_energy = self.mu_k * (self.f_k ** 2) * self.C * len(self.dataset)
            
            # Staleness-aware communication energy estimate
            tau = max(self.staleness, 1)  # Prevent division by zero
            gamma = 1.0 / (tau + 1e-6)
            comm_energy = (self.p_k ** 2) * (gamma ** 2) * torch.norm(self.h_k).item()
            
            return comp_energy + comm_energy

    def run(self):
        """Main client execution loop"""
        while self.alive:
            if self.selected and self.received_model is not None:
                self._perform_local_training()
                
                # Notify server training is complete
                if self.notify_done_callback:
                    self.notify_done_callback(self.client_id)
                
                # Wait for server's OTA transmission signal
                self.grad_ready_event.wait()
                self.grad_ready_event.clear()
                
                self._transmit_gradient()
                self.selected = False
                self.staleness = 0
            else:
                # Idle state - increment staleness counter
                time.sleep(0.1)
                self.staleness += 1

    def _perform_local_training(self, E=5):
        """Execute local SGD with staleness-aware initialization"""
        optimizer = optim.SGD(self.model.parameters(), lr=self.local_lr)
        self.model.train()

        # Store initial parameters for gradient calculation
        with self.lock:
            initial_params = torch.nn.utils.parameters_to_vector(
                self.received_model.parameters()
            ).detach().clone()

        # Local training loop
        for _ in range(E):
            for x, y in self.dataloader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                output = self.model(x)
                loss = nn.CrossEntropyLoss()(output, y)
                loss.backward()
                optimizer.step()

        # Calculate local gradient relative to received model
        with self.lock:
            trained_params = torch.nn.utils.parameters_to_vector(self.model.parameters())
            self.local_gradient = (trained_params - initial_params) / self.local_lr

        print(f"Client {self.client_id} completed training | Staleness: {self.staleness}")

    def _transmit_gradient(self):
        """Prepare and transmit gradient via OTA aggregation"""
        if self.local_gradient is None:
            return

        with self.lock:
            # Staleness-aware weighting
            tau = max(self.staleness, 1)  # Ensure τ ≥ 1
            gamma = 1.0 / (tau + 1e-6)
            
            # Construct transmission signal (Eq. 7)
            h = self.h_k
            p = self.p_k
            b_k = torch.conj(h) / (h.abs() ** 2 + 1e-6)
            s_k = (p * gamma) * b_k * self.local_gradient.to(torch.cfloat)

            # Calculate actual energy consumption
            E_comm = torch.norm(s_k).item() ** 2
            E_comp = self.mu_k * (self.f_k ** 2) * self.C * len(self.dataset)
            total_energy = E_comp + E_comm

            if self.send_gradient_callback:
                self.send_gradient_callback(
                    client_id=self.client_id,
                    signal=s_k.detach().cpu(),
                    energy=total_energy
                )

    def set_callbacks(self, done_cb, send_cb):
        """Register server notification callbacks"""
        self.notify_done_callback = done_cb
        self.send_gradient_callback = send_cb

    def trigger_transmission(self):
        """Signal client to initiate OTA transmission"""
        self.grad_ready_event.set()

    def stop(self):
        """Gracefully terminate client thread"""
        self.alive = False
        self.grad_ready_event.set()  # Unblock if waiting