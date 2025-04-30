import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import threading

class Client(threading.Thread):
    def __init__(self, client_id, data_indices, train_dataset, test_dataset, model, device, compute_power, channel, eta, E_local):
        super(Client, self).__init__()
        self.id = client_id
        self.device = device
        self.model = model.to(self.device)
        
        self.train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        self.eta = eta
        self.E_local = E_local    # Number of local epochs before sending
        self.C = 1e6
        self.D_k = len(data_indices)
        self.f_k = compute_power
        self.h_k = channel

        self.model_dim = sum(p.numel() for p in self.model.parameters())
        self.model_stale = torch.zeros(self.model_dim, device=self.device)
        self.model_updated = torch.zeros(self.model_dim, device=self.device)
        self.gtk = torch.zeros(self.model_dim, device=self.device)
        
        self.staleness = 0
        self.bk = 0  # Readiness indicator

        # Thread control
        self.lock = threading.Lock()
        self.new_model_event = threading.Event()
        self.stop_event = threading.Event()

        # New addition: signaling server when finished
        self.server_ready_event = None  # This will be set externally per round

    def flatten_model(self):
        return torch.cat([param.data.view(-1) for param in self.model.parameters()])

    def set_model_from_flat(self, flat_model):
        pointer = 0
        for param in self.model.parameters():
            num_param = param.numel()
            param.data = flat_model[pointer:pointer + num_param].view_as(param).clone()
            pointer += num_param

    def receive_global_model(self, w_global_flat, server_ready_event=None):
        """
        Server sends new global model to the client and the event for signaling.
        """
        with self.lock:
            self.model_stale = w_global_flat.clone()
            self.set_model_from_flat(self.model_stale)
            self.staleness = 0
            self.bk = 0  # Busy with new local computation
            self.server_ready_event = server_ready_event  # Event the server wants us to set
            self.new_model_event.set()  # Wake up to start local training

    def local_train(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.eta)
        criterion = torch.nn.CrossEntropyLoss()

        print(f"[Client {self.id}] Starting local_train() with {len(self.train_loader.dataset)} samples.")

        self.model.train()

        for epoch in range(self.E_local):
            print(f"[Client {self.id}] Starting epoch {epoch}...")

            batch_counter = 0
            for i, (x, y) in enumerate(self.train_loader):
                # print(f"[Client {self.id}] Training batch {i}...")
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                output = self.model(x)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
                batch_counter += 1

            print(f"[Client {self.id}] Finished epoch {epoch}.")

        self.model_updated = self.flatten_model()
        self.gtk = - (1.0 / self.eta) * (self.model_updated - self.model_stale)

        self.bk = 1  # Ready to transmit
        print(f"[Client {self.id}] Finished local_train(), bk=1.")
        
        # Immediately notify server
        if self.server_ready_event is not None:
            self.server_ready_event.set()
            print(f"[Client {self.id}] Event triggered for server.")


    def get_ota_signal(self, ptk, gamma_tk):
        """
        Build OTA transmitted signal.
        """
        h_conj = torch.conj(self.h_k)
        h_norm_sq = torch.abs(self.h_k) ** 2
        weight_vector = h_conj / h_norm_sq
        s_tk = ptk * gamma_tk * weight_vector * self.gtk
        return s_tk

    def increment_staleness(self):
        self.staleness += 1

    def get_computation_time(self):
        return (self.C * self.D_k) / self.f_k

    def run(self):
        try:
            while not self.stop_event.is_set():
                self.new_model_event.wait()
                if self.stop_event.is_set():
                    break
                print(f"[Client {self.id}] Received new model, starting local training.")
                self.local_train()
                self.new_model_event.clear()
        except Exception as e:
            print(f"[Client {self.id}] CRASHED with exception: {e}")

    def stop(self):
        """
        Stop client thread gracefully.
        """
        self.stop_event.set()
        self.new_model_event.set()  # Unblock if waiting
