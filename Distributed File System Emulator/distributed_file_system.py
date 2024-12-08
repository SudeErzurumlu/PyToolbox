Distributed File System Emulator: distributed_file_system.pyimport os
import threading
import socket
import json

class DistributedNode:
    def __init__(self, node_id, directory, port):
        """
        Initializes a distributed node.
        Args:
            node_id (str): Unique ID for the node.
            directory (str): Directory to store replicated files.
            port (int): Port for communication.
        """
        self.node_id = node_id
        self.directory = directory
        self.port = port
        self.peers = []
        os.makedirs(directory, exist_ok=True)

    def add_peer(self, peer_address):
        """
        Adds a peer to the node's network.
        """
        self.peers.append(peer_address)

    def replicate_file(self, filename, content):
        """
        Replicates a file to all peers.
        """
        for peer in self.peers:
            self.send_to_peer(peer, {"action": "replicate", "filename": filename, "content": content})

    def send_to_peer(self, peer_address, data):
        """
        Sends data to a peer.
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect(peer_address)
            s.sendall(json.dumps(data).encode())

    def handle_connection(self, conn, addr):
        """
        Handles incoming peer connections.
        """
        data = json.loads(conn.recv(1024).decode())
        if data["action"] == "replicate":
            filepath = os.path.join(self.directory, data["filename"])
            with open(filepath, "w") as f:
                f.write(data["content"])
            print(f"File {data['filename']} replicated on {self.node_id}")

    def start_server(self):
        """
        Starts the node's server to listen for connections.
        """
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind(("localhost", self.port))
        server.listen(5)
        print(f"Node {self.node_id} listening on port {self.port}...")
        while True:
            conn, addr = server.accept()
            threading.Thread(target=self.handle_connection, args=(conn, addr)).start()

# Example Usage:
# node1 = DistributedNode("Node1", "./node1_files", 5001)
# node2 = DistributedNode("Node2", "./node2_files", 5002)
# node1.add_peer(("localhost", 5002))
# threading.Thread(target=node1.start_server).start()
# threading.Thread(target=node2.start_server).start()
# node1.replicate_file("example.txt", "This is replicated content.")
