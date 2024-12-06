import hashlib
import time

class Block:
    def __init__(self, index, previous_hash, data, timestamp=None):
        """
        Initializes a new block in the blockchain.
        Args:
            index (int): The block index.
            previous_hash (str): The hash of the previous block.
            data (str): The data stored in the block.
            timestamp (float): The time the block was created.
        """
        self.index = index
        self.previous_hash = previous_hash
        self.data = data
        self.timestamp = timestamp or time.time()
        self.hash = self.calculate_hash()

    def calculate_hash(self):
        """
        Calculates the hash of the block.
        Returns:
            str: The SHA-256 hash of the block.
        """
        block_data = f"{self.index}{self.previous_hash}{self.data}{self.timestamp}"
        return hashlib.sha256(block_data.encode()).hexdigest()

class Blockchain:
    def __init__(self):
        """
        Initializes a blockchain with a genesis block.
        """
        self.chain = [self.create_genesis_block()]

    def create_genesis_block(self):
        """
        Creates the first block in the blockchain.
        Returns:
            Block: The genesis block.
        """
        return Block(0, "0", "Genesis Block")

    def get_latest_block(self):
        """
        Gets the latest block in the blockchain.
        Returns:
            Block: The latest block.
        """
        return self.chain[-1]

    def add_block(self, data):
        """
        Adds a new block to the blockchain.
        Args:
            data (str): The data for the new block.
        """
        latest_block = self.get_latest_block()
        new_block = Block(len(self.chain), latest_block.hash, data)
        self.chain.append(new_block)

    def is_chain_valid(self):
        """
        Validates the integrity of the blockchain.
        Returns:
            bool: True if valid, False otherwise.
        """
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]

            if current_block.hash != current_block.calculate_hash():
                return False
            if current_block.previous_hash != previous_block.hash:
                return False
        return True

# Example usage:
# blockchain = Blockchain()
# blockchain.add_block("First block after genesis.")
# blockchain.add_block("Another block.")
# for block in blockchain.chain:
#     print(block.__dict__)
# print("Is blockchain valid?", blockchain.is_chain_valid())
