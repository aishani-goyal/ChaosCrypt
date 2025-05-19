# --- Importing necessary libraries ---
import os
import cv2
import numpy as np
import base64
from docx import Document
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes
import json
import random
import string
from PyPDF2 import PdfReader, PdfWriter  # Import PDF handling libraries
import hashlib
from getpass import getpass

# --- Chaos Functions ---
def logistic_map(seed, r, size):
    """
    Generate a chaotic sequence based on the Logistic Map.
    :param seed: Initial value (float between 0 and 1)
    :param r: Growth rate (chaotic behavior when 3.57 < r <= 4)
    :param size: Length of sequence needed
    :return: List of integers in [0,255]
    """
    x = seed
    chaotic_seq = []
    for _ in range(size):
        x = r * x * (1 - x)
        chaotic_seq.append(int(x * 255) % 256)  # Normalize to 0-255
    return chaotic_seq

def xor_data(data_bytes, chaotic_seq):
    """
    XOR each byte of data with chaotic sequence.
    :param data_bytes: Original byte data
    :param chaotic_seq: Chaotic sequence
    :return: XORed bytes
    """
    return bytes([b ^ chaotic_seq[i % len(chaotic_seq)] for i, b in enumerate(data_bytes)])

# --- Arnold Cat Map for image scrambling ---
def arnold_cat_map(channel, iterations):
    """
    Apply Arnold Cat Map scrambling to a single image channel.
    :param channel: 2D numpy array (single channel)
    :param iterations: Number of scrambling iterations
    :return: Scrambled channel
    """
    N = channel.shape[0]
    for _ in range(iterations):
        new_channel = np.zeros_like(channel)
        for x in range(N):
            for y in range(N):
                new_x = (x + y) % N
                new_y = (x + 2 * y) % N
                new_channel[new_x, new_y] = channel[x, y]
        channel = new_channel
    return channel

def inverse_arnold_cat_map(channel, iterations):
    """
    Apply inverse Arnold Cat Map descrambling to a single image channel.
    :param channel: 2D numpy array (single channel)
    :param iterations: Number of descrambling iterations
    :return: Descrambled channel
    """
    N = channel.shape[0]
    for _ in range(iterations):
        new_channel = np.zeros_like(channel)
        for x in range(N):
            for y in range(N):
                new_x = (2 * x - y) % N
                new_y = (-x + y) % N
                new_channel[new_x, new_y] = channel[x, y]
        channel = new_channel
    return channel

# --- AES (CBC Mode) Encryption/Decryption Functions ---
def aes_encrypt(data, key):
    """
    AES encrypt data using CBC mode.
    :param data: Data to encrypt (bytes)
    :param key: 16-character AES key
    :return: IV + ciphertext (bytes)
    """
    iv = get_random_bytes(16)
    # Hash the key to get a valid 32-byte AES key
    hashed_key = hashlib.sha256(key.encode()).digest()
    cipher = AES.new(hashed_key, AES.MODE_CBC, iv)
    return iv + cipher.encrypt(pad(data, AES.block_size))

def aes_decrypt(data, key):
    """
    AES decrypt data using CBC mode.
    :param data: IV + ciphertext (bytes)
    :param key: 16-character AES key
    :return: Decrypted data (bytes)
    """
    iv = data[:16]
    ciphertext = data[16:]
    # Hash the key to get the same 32-byte AES key
    hashed_key = hashlib.sha256(key.encode()).digest()
    cipher = AES.new(hashed_key, AES.MODE_CBC, iv)
    return unpad(cipher.decrypt(ciphertext), AES.block_size)

# --- File Handling Functions --- 
def encrypt_file(file_path, r, seed, aes_key, arnold_iter=5):
    """
    Encrypt a file (text/docx, pdf, or image).
    :param file_path: Path to the file
    :param r: Logistic map parameter
    :param seed: Logistic map seed
    :param aes_key: AES key
    :param arnold_iter: Number of Arnold Cat Map iterations
    """
    ext = os.path.splitext(file_path)[1].lower()
    with open(file_path, 'rb') as f:
        content = f.read()

    if ext in ['.txt', '.docx', '.pdf']:
        # --- Text, Word Document, and PDF encryption ---
        chaotic_seq = logistic_map(seed, r, len(content))
        xored = xor_data(content, chaotic_seq)
        encrypted = aes_encrypt(xored, aes_key)

        out_path = file_path + ".enc"
        with open(out_path, 'wb') as f:
            f.write(encrypted)
        print(f"[+] Encrypted file saved to: {out_path}")

        os.remove(file_path)
        print(f"[+] Original file {file_path} deleted.")

    elif ext in ['.jpg', '.jpeg', '.png']:
        # --- Image encryption ---
        image = cv2.imread(file_path, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (256, 256))  # Resize image to 256x256

        scrambled = np.zeros_like(image)
        for i in range(3):  # Scramble each BGR channel separately
            scrambled[:, :, i] = arnold_cat_map(image[:, :, i], arnold_iter)

        chaotic_seq = logistic_map(seed, r, scrambled.size)
        xored = xor_data(scrambled.flatten(), chaotic_seq)
        encrypted = aes_encrypt(xored, aes_key)

        out_path = file_path + ".enc"
        with open(out_path, 'wb') as f:
            f.write(encrypted)
        print(f"[+] Encrypted image saved to: {out_path}")

        os.remove(file_path)
        print(f"[+] Original image {file_path} deleted.")

def decrypt_file(file_path, r, seed, aes_key, arnold_iter=5):
    """
    Decrypt a file (image or text/docx).
    :param file_path: Path to encrypted file
    :param r: Logistic map parameter
    :param seed: Logistic map seed
    :param aes_key: AES key
    :param arnold_iter: Number of Arnold Cat Map iterations
    :return: True if successful, False otherwise
    """
    if not os.path.exists(file_path):
        print("[-] File not found. Please check the path.")
        return False

    try:
        with open(file_path, 'rb') as f:
            data = f.read()

        if file_path.endswith('.enc'):
            try:
                decrypted = aes_decrypt(data, aes_key)
                print(f"[+] AES decryption successful.")
                
                chaotic_seq = logistic_map(seed, r, len(decrypted))
                xored = xor_data(decrypted, chaotic_seq)
                print(f"[+] XORing completed.")

                try:
                    # Try treating as image
                    unscrambled = np.frombuffer(xored, dtype=np.uint8).reshape((256, 256, 3))
                    final_img = np.zeros_like(unscrambled)
                    for i in range(3):
                        final_img[:, :, i] = inverse_arnold_cat_map(unscrambled[:, :, i], arnold_iter)

                    out_path = file_path[:-4]
                    cv2.imwrite(out_path, final_img)
                    print(f"[+] Decrypted image saved to: {out_path}")
                    os.remove(file_path)
                    print(f"[+] Encrypted image {file_path} deleted.")
                    return True

                except ValueError:
                    # If not image, treat as text/docx
                    out_path = file_path.replace('.enc', '')
                    with open(out_path, 'wb') as f:
                        f.write(xored)
                    print(f"[+] Decrypted text file saved to: {out_path}")
                    os.remove(file_path)
                    print(f"[+] Encrypted file {file_path} deleted.")
                    return True

            except Exception as e:
                print(f"[-] Failed to decrypt file: {e}")
                return False

    except Exception as e:
        print(f"[-] Error reading encrypted file: e{e}")
        return False
    
def get_key_from_password(password):
    return hashlib.sha256(password.encode()).digest()[:16]

def generate_key_file(key_path="encryption.key.enc", password=""):
    r = round(random.uniform(3.57, 4.0), 4)
    seed = round(random.uniform(0.01, 0.99), 6)
    aes_key = ''.join(random.choices(string.ascii_letters + string.digits, k=16))

    key_data = json.dumps({"r": r, "seed": seed, "aes_key": aes_key}).encode()
    encryption_key = get_key_from_password(password)
    encrypted_key = aes_encrypt(key_data, encryption_key.decode(errors='ignore'))

    with open(key_path, "wb") as f:
        f.write(encrypted_key)

    print(f"[+] Encrypted key file saved as: {key_path}")
    return r, seed, aes_key

def load_key_file(key_path, password):
    encryption_key = get_key_from_password(password)
    with open(key_path, "rb") as f:
        encrypted_data = f.read()
    decrypted = aes_decrypt(encrypted_data, encryption_key.decode(errors='ignore'))
    key_data = json.loads(decrypted.decode())
    print(f"[+] Loaded and decrypted key file from: {key_path}")
    return key_data["r"], key_data["seed"], key_data["aes_key"]

# --- Main Interface ---
if __name__ == "__main__":
    file_path = input("Enter file path to encrypt/decrypt: ").strip()
    mode = input("Mode (e = encrypt / d = decrypt): ").strip().lower()
    key_choice = input("Use existing key file? (y/n): ").strip().lower()

    if key_choice == 'y':
        key_path = input("Enter key file path (e.g., encryption.key.enc): ").strip()
        if not os.path.exists(key_path):
            print("[-] Key file path invalid. Exiting.")
            exit(1)
        password = getpass("Enter password to decrypt the key file: ").strip()
        r, seed, aes_key = load_key_file(key_path, password)
    else:
        password = getpass("Enter password to encrypt new key file: ").strip()
        r, seed, aes_key = generate_key_file(password=password)
        key_path = "encryption.key.enc"

    if mode == 'e':
        encrypt_file(file_path, r, seed, aes_key)
    elif mode == 'd':
        success = decrypt_file(file_path, r, seed, aes_key)
        if success and key_choice == 'y' and os.path.exists(key_path):
            os.remove(key_path)
            print(f"[+] Key file {key_path} deleted.")
    else:
        print("[-] Invalid mode. Use 'e' or 'd'.")