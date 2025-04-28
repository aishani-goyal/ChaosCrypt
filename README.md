# 🔐 **ChaosCrypt**  
_An advanced encryption tool combining Chaos Theory and AES for secure file encryption._

---

### 🌟 **Features**  
- **Chaos-based encryption** for added randomness using Logistic Map and Arnold Cat Map.
- **AES-128 CBC encryption** with a random IV for robust data security.
- **Image scrambling** using the Arnold Cat Map (applies to images only).
- **Key file generation** and management for seamless encryption/decryption.
- **Auto-delete original files** after successful encryption/decryption to enhance security.

---

### 📂 **Supported File Formats**  
- `.txt`, `.docx`, `.pdf`, `.jpg`, `.jpeg`, `.png`

---

### ⚙️ **Requirements**  
- Install dependencies via pip:  
  ```bash
  pip install numpy opencv-python python-docx pycryptodome
  ```

---

### 🚀 **How to Run**  
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ChaosCrypt.git
   ```
2. Navigate to the directory and run the script:
   ```bash
   python chaos_aes_tool.py
   ```
3. Follow the on-screen prompts:
   - Enter file path 📂
   - Select mode: `e` (encrypt) or `d` (decrypt) 🔒🔓
   - Choose to generate or load a key file 🔑

---

### 📝 **Usage Example**  
1. **Encrypting a file**:
   - Enter file path: `sample.jpg`
   - Mode: `e`  
   - Use existing key file? (`y/n`): `n`  
   - **Key file saved** as: `encryption.key`  
   - **Encrypted image saved** as: `sample.jpg.enc`  
   - **Original image deleted** after encryption 🔥  

2. **Decrypting a file**:
   - Enter file path: `sample.jpg.enc`
   - Mode: `d`  
   - Use existing key file? (`y/n`): `y`  
   - **Decrypted image saved** as: `sample.jpg`  
   - **Original encrypted file deleted** 🔓  

---

### 🔐 **Security Highlights**  
- **Logistic Map** chaotic sequence used for XORing data, enhancing encryption randomness.
- **Arnold Cat Map** for scrambling image files only, adding another layer of complexity.
- **AES-CBC** encryption with a random Initialization Vector (IV) for robust protection.
- **Key file** stores essential parameters like `r`, `seed`, and the **AES key** for future use.

---

### 📝 **License**  
This project is **open source** and free to use.  
