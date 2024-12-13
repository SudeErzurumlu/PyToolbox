import hashlib
import time
import random

class Block:
    def __init__(self, index, previous_hash, timestamp, data, nonce=0):
        self.index = index
        self.previous_hash = previous_hash
        self.timestamp = timestamp
        self.data = data
        self.nonce = nonce
        self.hash = self.compute_hash()

    def compute_hash(self):
        """
        Generates a SHA-256 hash for the block.
        """
        block_string = f"{self.index}{self.previous_hash}{self.timestamp}{self.data}{self.nonce}"
        return hashlib.sha256(block_string.encode()).hexdigest()

class Blockchain:
    def __init__(self, consensus_algorithm="Proof-of-Work"):
        self.chain = [self.create_genesis_block()]
        self.consensus_algorithm = consensus_algorithm

    def create_genesis_block(self):
        """
        Creates the first block in the blockchain.
        """
        return Block(0, "0", time.time(), "Genesis Block")

    def get_latest_block(self):
        """
        Returns the latest block in the chain.
        """
        return self.chain[-1]

    def add_block(self, new_block):
        """
        Adds a new block to the chain after validation.
        """
        new_block.previous_hash = self.get_latest_block().hash
        new_block.hash = new_block.compute_hash()
        if self.validate_block(new_block):
            self.chain.append(new_block)

    def validate_block(self, block):
        """
        Validates the block based on the consensus algorithm.
        """
        if self.consensus_algorithm == "Proof-of-Work":
            return block.hash.startswith("0000")  # Example difficulty
        elif self.consensus_algorithm == "Proof-of-Stake":
            return random.choice([True, False])  # Placeholder logic
        return False

    def mine_block(self, data, difficulty=4):
        """
        Mines a new block using Proof-of-Work.
        """
        new_block = Block(len(self.chain), self.get_latest_block().hash, time.time(), data)
        while not new_block.hash.startswith("0" * difficulty):
            new_block.nonce += 1
            new_block.hash = new_block.compute_hash()
        self.add_block(new_block)
        return new_block

# Example Usage
blockchain = Blockchain()
blockchain.mine_block("First Transaction")
blockchain.mine_block("Second Transaction")
for block in blockchain.chain:
    print(vars(block))
