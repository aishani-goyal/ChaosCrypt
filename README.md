**Chaos-AES File Encryption Tool 🔒**
- A lightweight encryption tool combining Chaos Theory (Logistic Map, Arnold Cat Map) and AES (CBC mode) to securely encrypt text, document, and image files.
- Supports .txt, .docx, .jpg, .jpeg, .png formats.

**Features**
- Chaos-based encryption for extra randomness.
- AES-128 CBC encryption with random IV.
- Image scrambling with Arnold Cat Map.
- Key file generation and management.
- Auto-delete original files after encryption/decryption.

**Requirements**
- Install dependencies: pip install numpy opencv-python python-docx pycryptodome

**How to Run**
- Clone the repository.
- Run the script: python chaos_aes_tool.py
- Follow prompts:
  - Enter file path
  - Select mode: e (encrypt) or d (decrypt)
  - Choose to generate or load a key file.

**Usage Example**
Enter file path to encrypt/decrypt: sample.jpg
Mode (e = encrypt / d = decrypt): e
Use existing key file? (y/n): n
[+] Key file saved as: encryption.key
[+] Encrypted image saved to: sample.jpg.enc
[+] Original image sample.jpg deleted.
For decryption, use the same .key file!

**Security Highlights**
- Logistic Map chaotic sequence for XORing data.
- Arnold Cat Map scrambling (images only).
- AES-CBC encryption with random IV.
- Key file contains necessary r, seed, and AES key.

**License**
-This project is open source and free to use.

