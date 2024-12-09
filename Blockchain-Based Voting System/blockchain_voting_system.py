import hashlib
import json
import time

class Blockchain:
    def __init__(self):
        """
        Initializes the blockchain with a genesis block.
        """
        self.chain = []
        self.create_block(previous_hash="0", proof=1)

    def create_block(self, proof, previous_hash):
        """
        Creates a new block in the blockchain.
        """
        block = {
            "index": len(self.chain) + 1,
            "timestamp": time.time(),
            "proof": proof,
            "previous_hash": previous_hash,
            "data": []
        }
        self.chain.append(block)
        return block

    def add_vote(self, voter_id, candidate):
        """
        Adds a vote transaction to the latest block.
        """
        self.chain[-1]["data"].append({"voter_id": voter_id, "candidate": candidate})

    def hash(self, block):
        """
        Generates a hash for a block.
        """
        return hashlib.sha256(json.dumps(block, sort_keys=True).encode()).hexdigest()

    def proof_of_work(self, previous_proof):
        """
        Proof-of-work algorithm to validate a new block.
        """
        new_proof = 1
        while True:
            hash_attempt = hashlib.sha256(str(new_proof**2 - previous_proof**2).encode()).hexdigest()
            if hash_attempt[:4] == "0000":
                return new_proof
            new_proof += 1

    def is_chain_valid(self):
        """
        Validates the blockchain.
        """
        for i in range(1, len(self.chain)):
            block = self.chain[i]
            previous_block = self.chain[i - 1]
            if block["previous_hash"] != self.hash(previous_block):
                return False
            if not self.hash_proof_of_work(block["proof"], previous_block["proof"]):
                return False
        return True

# Example Usage:
# blockchain = Blockchain()
# blockchain.add_vote("Voter1", "CandidateA")
# proof = blockchain.proof_of_work(blockchain.chain[-1]["proof"])
# blockchain.create_block(proof, blockchain.hash(blockchain.chain[-1]))
# print(blockchain.chain)
