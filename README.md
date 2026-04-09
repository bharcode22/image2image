# 🚀 AI Image Generator (FLUX.2-klein-4B)

Project ini adalah API untuk generate gambar menggunakan model **FLUX.2-klein-4B** dari Hugging Face dengan FastAPI.

---

## 📦 Requirements

* Python 3.11
* GPU (disarankan, minimal 8GB VRAM)
* pip

---

## ⚙️ Setup Environment

### 1. Clone Repository

```bash
git clone https://github.com/bharcode22/image2image.git
cd image2image
```
```bash
git clone git@github.com:bharcode22/image2image.git
cd image2image
```

---

### 2. Buat Virtual Environment

```bash
python3.11 -m venv ai-env
```

---

### 3. Aktifkan Virtual Environment

```bash
source ai-env/bin/activate
```

---

### 4. Install Dependencies

```bash
pip install --upgrade pip
pip install fastapi uvicorn torch diffusers huggingface_hub accelerate python-multipart pillow tqdm
```

---

## 🤖 Download Model

Download model dari Hugging Face:

```bash
hf download black-forest-labs/FLUX.2-klein-4B
```

Model akan tersimpan dengan nama:

```
black-forest-labs--FLUX.2-klein-4B
```

---

## ▶️ Menjalankan Server

```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

---

## ⚡ Menjalankan dengan PM2 (Production)

### Install PM2 (jika belum)

```bash
npm install -g pm2
```

---

### Jalankan dengan PM2

```bash
pm2 start "ai-env/bin/python -m uvicorn main:app --host 0.0.0.0 --port 8000" --name ai-server
```

---

### Cek Status

```bash
pm2 list
```

---

### Lihat Logs

```bash
pm2 logs ai-server
```

---

## 🌐 Endpoint API

### 🔹 Generate Image

**POST** `/generate`

```json
{
  "prompt": "A beautiful sunset",
  "height": 832,
  "width": 1216
}
```

---

### 🔹 List Images

**GET** `/generate/list`

---

### 🔹 Download Image

**GET** `/generate/download/{job_id}`

---

## ⚠️ Troubleshooting

### ❌ CUDA Out of Memory

* Kurangi ukuran gambar:

  * contoh: `512x768`
* Kurangi `num_inference_steps`
* Gunakan `enable_model_cpu_offload()`

---

### ❌ Module Not Found

Pastikan virtual environment aktif:

```bash
source ai-env/bin/activate
```

---

### ❌ PM2 Error Path

Gunakan absolute path:

```bash
/home/pod/folder/zaq/ai-env/bin/python
```

---

## 📌 Notes

* Model ini cukup berat, pastikan VRAM mencukupi
* Gunakan `nvitop` atau `nvtop` untuk monitoring GPU

---

## 👨‍💻 Author

bharcode 🚀