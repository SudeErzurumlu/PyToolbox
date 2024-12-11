import hashlib
import time
from threading import Thread, Lock

class Block:
    def __init__(self, index, previous_hash, transactions, timestamp=None):
        """
        Represents a single block in the blockchain.
        """
        self.index = index
        self.previous_hash = previous_hash
        self.transactions = transactions
        self.timestamp = timestamp or time.time()
        self.nonce = 0
        self.hash = self.compute_hash()

    def compute_hash(self):
        """
        Calculates the hash of the block.
        """
        block_string = f"{self.index}{self.previous_hash}{self.transactions}{self.timestamp}{self.nonce}"
        return hashlib.sha256(block_string.encode()).hexdigest()

    def mine(self, difficulty):
        """
        Mines the block by finding a hash with the required difficulty.
        """
        while not self.hash.startswith("0" * difficulty):
            self.nonce += 1
            self.hash = self.compute_hash()

class Blockchain:
    def __init__(self, difficulty=4):
        """
        Represents the blockchain network.
        """
        self.chain = []
        self.pending_transactions = []
        self.difficulty = difficulty
        self.lock = Lock()
        self.create_genesis_block()

    def create_genesis_block(self):
        """
        Creates the genesis block of the blockchain.
        """
        genesis_block = Block(0, "0", [])
        genesis_block.mine(self.difficulty)
        self.chain.append(genesis_block)

    def add_transaction(self, transaction):
        """
        Adds a new transaction to the list of pending transactions.
        """
        with self.lock:
            self.pending_transactions.append(transaction)

    def mine_block(self):
        """
        Mines a new block and adds it to the chain.
        """
        if not self.pending_transactions:
            return None
        new_block = Block(len(self.chain), self.chain[-1].hash, self.pending_transactions)
        new_block.mine(self.difficulty)
        with self.lock:
            self.chain.append(new_block)
            self.pending_transactions = []

# Example Usage:
# blockchain = Blockchain()
# blockchain.add_transaction("Alice -> Bob: 10 BTC")
# blockchain.mine_block()
