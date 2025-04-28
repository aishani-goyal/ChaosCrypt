#LIBRARIES
import numpy as np
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
import os
import struct
from PIL import Image
import io
import docx
import gradio as gr

# --------------------------
# Encryption & Decryption Functions
# --------------------------

def logistic_map(x, r, n):
    sequence = []
    for _ in range(n):
        x = r * x * (1 - x)
        sequence.append(int(x * 256) % 256)
    return np.array(sequence, dtype=np.uint8)

def arnold_cat_map(img_array, iterations):
    N = img_array.shape[0]
    result = np.copy(img_array)
    for _ in range(iterations):
        temp = np.copy(result)
        for x in range(N):
            for y in range(N):
                result[x, y] = temp[(x + y) % N, (x + 2 * y) % N]
    return result

def inverse_arnold_cat_map(img_array, iterations):
    N = img_array.shape[0]
    result = np.copy(img_array)
    for _ in range(iterations):
        temp = np.copy(result)
        for x in range(N):
            for y in range(N):
                result[x, y] = temp[(2 * x - y) % N, (-x + y) % N]
    return result

def chaos_encrypt(data, r, seed):
    sequence = logistic_map(seed, r, len(data))
    encrypted_data = np.bitwise_xor(np.frombuffer(data, dtype=np.uint8), sequence)
    return encrypted_data.tobytes()

def chaos_decrypt(data, r, seed):
    sequence = logistic_map(seed, r, len(data))
    decrypted_data = np.bitwise_xor(np.frombuffer(data, dtype=np.uint8), sequence)
    return decrypted_data.tobytes()

def aes_encrypt(data, key):
    cipher = AES.new(key, AES.MODE_CBC)
    ct_bytes = cipher.encrypt(pad(data, AES.block_size))
    return cipher.iv + ct_bytes

def aes_decrypt(data, key):
    iv = data[:AES.block_size]
    cipher = AES.new(key, AES.MODE_CBC, iv)
    pt = unpad(cipher.decrypt(data[AES.block_size:]), AES.block_size)
    return pt

def generate_key_file():
    r = np.random.uniform(3.57, 4.0)
    seed = np.random.uniform(0, 1)
    aes_key = os.urandom(32)
    with open("encryption.key", "wb") as f:
        f.write(struct.pack('f', r))
        f.write(struct.pack('f', seed))
        f.write(aes_key)
    return r, seed, aes_key

def load_key_file(key_path):
    with open(key_path, "rb") as f:
        r = struct.unpack('f', f.read(4))[0]
        seed = struct.unpack('f', f.read(4))[0]
        aes_key = f.read(32)
    return r, seed, aes_key

def encrypt_file(file_path, r, seed, aes_key):
    ext = os.path.splitext(file_path)[1].lower()
    if ext in ['.txt', '.docx', '.pdf']:
        with open(file_path, "rb") as f:
            data = f.read()
        encrypted_data = chaos_encrypt(data, r, seed)
        final_data = aes_encrypt(encrypted_data, aes_key)
        with open(file_path + ".enc", "wb") as f:
            f.write(final_data)
    elif ext in ['.jpg', '.jpeg', '.png']:
        img = Image.open(file_path)
        img = img.convert('RGB')
        img_data = np.array(img)
        if img_data.shape[0] != img_data.shape[1]:
            min_side = min(img_data.shape[:2])
            img_data = img_data[:min_side, :min_side]
        img_data = arnold_cat_map(img_data, 5)
        img_bytes = img_data.tobytes()
        encrypted_bytes = chaos_encrypt(img_bytes, r, seed)
        with open(file_path + ".enc", "wb") as f:  
            f.write(encrypted_bytes)
    else:
        raise ValueError("Unsupported file format")
    os.remove(file_path)


def decrypt_file(file_path, r, seed, aes_key):
    ext = os.path.splitext(file_path)[1].lower()
    original_ext = os.path.splitext(file_path[:-4])[1].lower()

    if original_ext in ['.txt', '.docx', '.pdf']:
        with open(file_path, "rb") as f:
            data = f.read()
        decrypted_data = aes_decrypt(data, aes_key)
        final_data = chaos_decrypt(decrypted_data, r, seed)
        with open(file_path[:-4], "wb") as f:
            f.write(final_data)
    elif original_ext in ['.jpg', '.jpeg', '.png']:
        with open(file_path, "rb") as f:
            encrypted_bytes = f.read()
        decrypted_bytes = chaos_decrypt(encrypted_bytes, r, seed)
        encrypted_img = np.frombuffer(decrypted_bytes, dtype=np.uint8)
        size = int(np.sqrt(len(encrypted_img) / 3))
        img_data = encrypted_img.reshape((size, size, 3))
        img_data = inverse_arnold_cat_map(img_data, 5)
        decrypted_img = Image.fromarray(img_data.astype(np.uint8))
        decrypted_img.save(file_path[:-4])
    else:
        raise ValueError("Unsupported file format")
    os.remove(file_path)
    return True

# --------------------------
# Gradio Interface
# --------------------------

def handle_encryption(file, key_file):
    if file is None:
        return "", "Please upload a file to encrypt."
    
    if key_file is None:
        r, seed, aes_key = generate_key_file()
    else:
        key_path = key_file.name
        r, seed, aes_key = load_key_file(key_path)

    file_path = file.name
    encrypt_file(file_path, r, seed, aes_key)

    encrypted_file_path = file_path + ".enc"
    if os.path.exists(encrypted_file_path):
        return encrypted_file_path, "✅ Encryption Successful!"
    else:
        return "", "❌ Encryption Failed."

def handle_decryption(file, key_file):
    if file is None or key_file is None:
        return "", "Please upload both the encrypted file and key file."

    file_path = file.name
    key_path = key_file.name

    r, seed, aes_key = load_key_file(key_path)
    success = decrypt_file(file_path, r, seed, aes_key)

    if success:
        decrypted_file_path = file_path[:-4]
        return decrypted_file_path, "✅ Decryption Successful!"
    else:
        return "", "❌ Decryption Failed."

def generate_keyfile_button():
    r, seed, aes_key = generate_key_file()
    keyfile_path = "encryption.key"
    return keyfile_path

# Gradio App

with gr.Blocks() as demo:
    gr.Markdown("## 🔐 Chaos + AES Encryption Tool")

    with gr.Row():
        with gr.Column():
            file_input = gr.File(label="Upload File to Encrypt/Decrypt")
            key_input = gr.File(label="Upload Key File (.key)")
            operation = gr.Radio(["Encrypt", "Decrypt"], value="Encrypt", label="Choose Operation")
            run_btn = gr.Button("Run")
        
        with gr.Column():
            output_file = gr.File(label="Download Result")
            output_text = gr.Textbox(label="Status")

    with gr.Row():
        key_gen_btn = gr.Button("Generate New Key File")
        key_gen_output = gr.File(label="Download Generated Key File")

    run_btn.click(
        lambda file, key, op: handle_encryption(file, key) if op == "Encrypt" else handle_decryption(file, key),
        inputs=[file_input, key_input, operation],
        outputs=[output_file, output_text]
    )

    key_gen_btn.click(
        fn=generate_keyfile_button,
        outputs=key_gen_output
    )

    demo.launch()
