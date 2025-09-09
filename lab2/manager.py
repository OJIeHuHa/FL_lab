import random
import threading
import time
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from typing import Dict, List

class CustomClientManager(ClientManager):
    def __init__(self):
        super().__init__()
        self._clients: Dict[str, ClientProxy] = {}
        self._lock = threading.Lock()

    def num_available(self) -> int:
        with self._lock:
            return len(self._clients)

    def num_total(self) -> int:
        with self._lock:
            return len(self._clients)

    def register(self, client: ClientProxy) -> bool:
        with self._lock:
            if client.cid not in self._clients:
                self._clients[client.cid] = client
                print(f"[CustomClientManager] Client {client.cid} registered.")
                return True
            else:
                print(f"[CustomClientManager] Client {client.cid} already registered.")
                return False

    def unregister(self, client: ClientProxy) -> bool:
        with self._lock:
            if client.cid in self._clients:
                del self._clients[client.cid]
                print(f"[CustomClientManager] Client {client.cid} unregistered.")
                return True
            else:
                print(f"[CustomClientManager] Client {client.cid} not found.")
                return False

    def all(self) -> List[ClientProxy]:
        with self._lock:
            return dict(self._clients)

    def sample(self, num_clients: int = None) -> List[ClientProxy]:
        with self._lock:
            available_clients = list(self._clients.values())
            num_to_sample = max(1, len(available_clients))
            return random.sample(available_clients, num_to_sample)

    def wait_for(self, num_clients: int, timeout: float) -> List[ClientProxy]:
        start = time.time()
        while self.num_available() < num_clients:
            if time.time() - start > timeout:
                print(f"Timeout reached while waiting for {num_clients} clients")
                break
            print(f"Waiting for clients... {self.num_available()}/{num_clients} connected")
            time.sleep(1)
        return self.all()

    @property
    def all_clients(self) -> Dict[str, ClientProxy]:
        with self._lock:
            return dict(self._clients)