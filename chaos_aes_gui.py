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

# --- Chaos Functions ---
def logistic_map(seed, r, size):
    x = seed
    chaotic_seq = []
    for _ in range(size):
        x = r * x * (1 - x)
        chaotic_seq.append(int(x * 255) % 256)
    return chaotic_seq

def xor_data(data_bytes, chaotic_seq):
    return bytes([b ^ chaotic_seq[i % len(chaotic_seq)] for i, b in enumerate(data_bytes)])

# --- Arnold Cat Map for images ---
def arnold_cat_map(channel, iterations):
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

# --- AES-CBC Functions ---
def aes_encrypt(data, key):
    iv = get_random_bytes(16)
    cipher = AES.new(key.encode(), AES.MODE_CBC, iv)
    ciphertext = cipher.encrypt(pad(data, AES.block_size))
    return iv + ciphertext  # Prepend IV

def aes_decrypt(data, key):
    iv = data[:16]
    ciphertext = data[16:]
    cipher = AES.new(key.encode(), AES.MODE_CBC, iv)
    return unpad(cipher.decrypt(ciphertext), AES.block_size)

# --- File Handlers --- 
def encrypt_file(file_path, r, seed, aes_key, arnold_iter=5):
    ext = os.path.splitext(file_path)[1].lower()
    with open(file_path, 'rb') as f:
        content = f.read()

    if ext in ['.txt', '.docx', '.pdf']:   # <- both are treated similarly now
        chaotic_seq = logistic_map(seed, r, len(content))
        xored = xor_data(content, chaotic_seq)
        encrypted = aes_encrypt(xored, aes_key)

        out_path = file_path + '.enc'
        with open(out_path, 'wb') as f:
            f.write(encrypted)
        print(f"[+] Encrypted file saved to: {out_path}")

        os.remove(file_path)
        print(f"[+] Original file {file_path} deleted.")


    elif ext in ['.jpg', '.jpeg', '.png']:
        image = cv2.imread(file_path, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (256, 256))

        scrambled = np.zeros_like(image)
        for i in range(3):  # BGR channels
            scrambled[:, :, i] = arnold_cat_map(image[:, :, i], arnold_iter)

        chaotic_seq = logistic_map(seed, r, scrambled.size)
        xored = xor_data(scrambled.flatten(), chaotic_seq)
        encrypted = aes_encrypt(xored, aes_key)

        out_path = file_path + '.enc'
        with open(out_path, 'wb') as f:
            f.write(encrypted)
        print(f"[+] Encrypted image saved to: {out_path}")

        os.remove(file_path)
        print(f"[+] Original image {file_path} deleted.")

def decrypt_file(file_path, r, seed, aes_key, arnold_iter=5):
    if not os.path.exists(file_path):
        print("[-] File not found. Please check the path.")
        return False  # decryption unsuccessful

    with open(file_path, 'rb') as f:
        data = f.read()

    if file_path.endswith('.enc'):
        try:
            decrypted = aes_decrypt(data, aes_key)
            chaotic_seq = logistic_map(seed, r, len(decrypted))
            xored = xor_data(decrypted, chaotic_seq)

            try:
                unscrambled = np.frombuffer(xored, dtype=np.uint8).reshape((256, 256, 3))
                final_img = np.zeros_like(unscrambled)
                for i in range(3):
                    final_img[:, :, i] = inverse_arnold_cat_map(unscrambled[:, :, i], arnold_iter)

                out_path = file_path[:-4]
                cv2.imwrite(out_path, final_img)
                print(f"[+] Decrypted image saved to: {out_path}")
                os.remove(file_path)
                print(f"[+] Encrypted image {file_path} deleted.")
                return True  # decryption successful
            except ValueError:
                # It's likely a text file
                out_path = file_path.replace('.enc', '')
                with open(out_path, 'wb') as f:
                    f.write(xored)
                print(f"[+] Decrypted text file saved to: {out_path}")
                os.remove(file_path)
                print(f"[+] Encrypted file {file_path} deleted.")
                return True  # decryption successful

        except Exception as e:
            print(f"[-] Failed to decrypt file: {e}")
            return False  # decryption unsuccessful


def generate_key_file(key_path="encryption.key"):
    if os.path.exists(key_path):
        base, ext = os.path.splitext(key_path)
        counter = 1
        new_key_path = f"{base}_{counter}{ext}"
        while os.path.exists(new_key_path):
            counter += 1
            new_key_path = f"{base}_{counter}{ext}"
        key_path = new_key_path

    r = round(random.uniform(3.57, 4.0), 4)
    seed = round(random.uniform(0.01, 0.99), 6)
    aes_key = ''.join(random.choices(string.ascii_letters + string.digits, k=16))

    key_data = {
        "r": r,
        "seed": seed,
        "aes_key": aes_key
    }

    with open(key_path, "w") as f:
        json.dump(key_data, f)

    print(f"[+] Key file saved as: {key_path}")
    return r, seed, aes_key

def load_key_file(key_path):
    with open(key_path, "r") as f:
        key_data = json.load(f)
    print(f"[+] Loaded key from: {key_path}")
    return key_data["r"], key_data["seed"], key_data["aes_key"]

# --- Main Interface ---
if __name__ == "__main__":
    file_path = input("Enter file path to encrypt/decrypt: ").strip()
    mode = input("Mode (e = encrypt / d = decrypt): ").strip().lower()
    key_choice = input("Use existing key file? (y/n): ").strip().lower()

    if key_choice == 'y':
        key_path = input("Enter key file path (e.g., encryption.key): ").strip()
        if not key_path or not os.path.exists(key_path):
            print("[-] Key file path is invalid or does not exist. Exiting.")
            exit(1)
        r, seed, aes_key = load_key_file(key_path)
    else:
        key_path = None
        r, seed, aes_key = generate_key_file()


    if mode == 'e':
        encrypt_file(file_path, r, seed, aes_key)

    elif mode == 'd':
        success = decrypt_file(file_path, r, seed, aes_key)

        if success and key_choice == 'y' and os.path.exists(key_path):
            os.remove(key_path)
            print(f"[+] Key file {key_path} deleted.")
        elif not success:
            print("[-] Decryption failed.")

    else:
        print("Invalid mode. Use 'e' for encrypt or 'd' for decrypt.")
