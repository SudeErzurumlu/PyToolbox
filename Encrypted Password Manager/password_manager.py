from cryptography.fernet import Fernet
import os

class PasswordManager:
    def __init__(self, key_file="secret.key", data_file="passwords.enc"):
        """
        Initializes the password manager with encryption.
        Args:
            key_file (str): File to store the encryption key.
            data_file (str): File to store encrypted passwords.
        """
        self.key_file = key_file
        self.data_file = data_file
        self.key = self.load_or_generate_key()
        self.fernet = Fernet(self.key)

    def load_or_generate_key(self):
        """
        Loads an existing encryption key or generates a new one.
        Returns:
            bytes: Encryption key.
        """
        if os.path.exists(self.key_file):
            with open(self.key_file, 'rb') as f:
                return f.read()
        else:
            key = Fernet.generate_key()
            with open(self.key_file, 'wb') as f:
                f.write(key)
            return key

    def add_password(self, service, username, password):
        """
        Encrypts and saves a password for a given service.
        Args:
            service (str): The service name.
            username (str): The username.
            password (str): The password.
        """
        entry = f"{service},{username},{password}\n".encode('utf-8')
        encrypted_entry = self.fernet.encrypt(entry)
        with open(self.data_file, 'ab') as f:
            f.write(encrypted_entry + b"\n")
        print(f"Password for '{service}' saved securely.")

    def retrieve_passwords(self):
        """
        Decrypts and displays all saved passwords.
        """
        if not os.path.exists(self.data_file):
            print("No passwords stored yet.")
            return
        with open(self.data_file, 'rb') as f:
            for line in f:
                try:
                    decrypted_entry = self.fernet.decrypt(line.strip())
                    service, username, password = decrypted_entry.decode('utf-8').split(',')
                    print(f"Service: {service}, Username: {username}, Password: {password}")
                except Exception as e:
                    print(f"Error decrypting an entry: {e}")

# Example usage:
# manager = PasswordManager()
# manager.add_password("example.com", "user123", "securepassword123")
# manager.retrieve_passwords()
