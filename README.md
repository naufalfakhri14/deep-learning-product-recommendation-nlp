# 💄 Skincare Product Recommender System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13.0-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**Sistem rekomendasi produk skincare berbasis Deep Learning menggunakan Neural Network Embeddings dan Cosine Similarity**

[Demo](#demo) • [Fitur](#-fitur-utama) • [Teknologi](#-teknologi-yang-digunakan) • [Instalasi](#-instalasi) • [Cara Penggunaan](#-cara-penggunaan) • [Arsitektur](#-arsitektur-sistem)

</div>

---

## 📋 Deskripsi Proyek

Sistem rekomendasi ini menggunakan **Deep Learning** dan **Natural Language Processing (NLP)** untuk membantu pengguna menemukan produk skincare yang sesuai berdasarkan:
- ✨ Kesamaan bahan aktif (active ingredients)
- 🎯 Manfaat produk (benefits)
- 🏷️ Brand preference
- 📦 Jenis produk (product type)

### 🎯 Problem Statement
Dengan banyaknya produk skincare di pasaran, konsumen sering kesulitan menemukan produk yang sesuai dengan kebutuhan kulitnya. Sistem ini membantu dengan memberikan rekomendasi produk berdasarkan kesamaan komposisi dan manfaat.

---

## ✨ Fitur Utama

### 1. 🔍 Pencarian Berdasarkan Bahan Aktif
- Cari produk dengan bahan aktif spesifik (niacinamide, vitamin C, retinol, dll)
- Filter berdasarkan brand:
  - Semua brand
  - Brand yang sama dengan hasil pertama
  - Brand spesifik
- Menampilkan daftar produk dengan gambar dan informasi lengkap

### 2. 📋 Rekomendasi Produk Serupa
- Pilih produk dari daftar
- Dapatkan rekomendasi produk serupa berdasarkan:
  - Kesamaan bahan aktif (similarity score)
  - Filter jenis produk (serum, moisturizer, toner, dll)
  - Jumlah rekomendasi (3-10 produk)
- Tampilan similarity score untuk setiap rekomendasi

### 3. 🖼️ Visualisasi Produk
- Gambar produk untuk setiap item
- Informasi lengkap: brand, jenis produk, jenis kulit, bahan aktif, manfaat
- Similarity badge dengan gradient color

### 4. 📊 Dashboard Interaktif
- UI/UX modern dengan desain beauty theme
- Statistik dataset (jumlah produk & brand)
- Tips penggunaan bahan aktif populer

---

## 🛠️ Teknologi yang Digunakan

### Machine Learning & Deep Learning
- **TensorFlow/Keras** - Neural Network untuk embedding
- **scikit-learn** - Cosine Similarity, metrics
- **Sastrawi** - Indonesian text stemming

### Data Processing
- **Pandas** - Data manipulation
- **NumPy** - Numerical operations
- **openpyxl** - Excel file processing

### Web Application
- **Streamlit** - Interactive web app
- **Pillow (PIL)** - Image processing

### Visualization
- **Matplotlib** - Data visualization
- **Seaborn** - Statistical plots

---

## 📊 Dataset

### Sumber Data
Dataset terdiri dari **215 produk skincare** dari **8 brand lokal Indonesia**:

| Brand | Jumlah Produk |
|-------|---------------|
| Avoskin | 30 |
| Emina | 30 |
| Glad2Glow | 25 |
| Originote | 25 |
| Scarlett | 25 |
| Somethinc | 25 |
| Wardah | 30 |
| Whitelab | 25 |

### Struktur Data
```
📁 Dataset
├── product_name          # Nama produk
├── product_type          # Jenis produk (Serum, Moisturizer, dll)
├── active_ingredients    # Bahan aktif produk
├── skin_type            # Jenis kulit yang cocok
├── benefits             # Manfaat produk
└── brand                # Brand produk
```

---

## 🏗️ Arsitektur Sistem

### 1. **Data Pipeline**

```
📥 Raw Data (Excel)
    ↓
🧹 Data Cleaning & Validation
    ↓
📝 Text Preprocessing (Lowercasing, Stemming, Cleaning)
    ↓
🔢 Tokenization & Padding
    ↓
🧠 Neural Network Embedding
    ↓
📊 Similarity Matrix Computation
    ↓
💾 Export untuk Deployment
```

### 2. **Model Architecture**

#### A. Product Embedding Model
```
Input (Sequence Length: 50)
    ↓
Embedding Layer (vocab_size: 5000, dim: 128)
    ↓
GlobalAveragePooling1D
    ↓
Dense (64 units, ReLU) + Dropout (0.3)
    ↓
Dense (32 units, ReLU) → Product Embedding
```

**Output:** Vektor 32-dimensi untuk setiap produk

#### B. Brand Classification Model
```
Input (Sequence Length: 50)
    ↓
Embedding Layer (128-dim)
    ↓
GlobalAveragePooling1D
    ↓
Dense (64 units, ReLU) + Dropout (0.3)
    ↓
Dense (32 units, ReLU, 'product_embedding')
    ↓
Dropout (0.3)
    ↓
Dense (8 units, Softmax) → Brand Prediction
```

**Metrics:**
- Training Accuracy: ~95%+
- Validation Accuracy: ~90%+

### 3. **Recommendation System**

#### Cosine Similarity
```python
similarity_score = cosine_similarity(embedding_A, embedding_B)
```

#### Filtering Process
1. **Compute similarity** untuk semua produk
2. **Sort by similarity score** (descending)
3. **Apply filters:**
   - Brand filter (same/different/specific)
   - Product type filter
   - Skin type filter
4. **Return top-N** recommendations

---

## 🚀 Instalasi

### Prerequisites
- Python 3.8 atau lebih tinggi
- pip (Python package manager)

### 1. Clone Repository
```bash
git clone https://github.com/username/skincare-recommender.git
cd skincare-recommender
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

**File `requirements.txt`:**
```txt
streamlit==1.28.0
tensorflow==2.13.0
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
Pillow==10.0.0
Sastrawi==1.0.1
openpyxl==3.1.2
matplotlib==3.7.2
seaborn==0.12.2
```

### 3. Persiapan Data

#### a. Struktur Folder
```
Tubes DL/
├── app.py                      # Streamlit application
├── main.ipynb                  # Jupyter notebook untuk training
├── requirements.txt
├── README.md
├── deployment_files/           # Model & data artifacts
│   ├── skincare_model.h5
│   ├── embedding_model.h5
│   ├── similarity_matrix.npy
│   ├── product_embeddings.npy
│   ├── skincare_products.csv
│   ├── tokenizer.pkl
│   └── label_encoder.pkl
├── avoskin/                    # Folder gambar brand
├── emina/
├── glad2glow/
├── originote/
├── scarlett/
├── somethinc/
├── wardah/
├── whitelab/
└── Excel Files (8 files)       # Data source
```

#### b. Training Model
Jalankan notebook `main.ipynb` untuk:
1. Load dan preprocess data
2. Train neural network model
3. Generate embeddings dan similarity matrix
4. Export artifacts ke `deployment_files/`

```python
# Di Jupyter Notebook, jalankan semua cell dari atas ke bawah
# Cell terakhir akan export semua file yang dibutuhkan
```

---

## 💻 Cara Penggunaan

### 1. Jalankan Aplikasi Streamlit

```bash
streamlit run app.py
```

Aplikasi akan buka di browser: `http://localhost:8501`

### 2. Menggunakan Fitur Pencarian

#### **Opsi 1: Cari Berdasarkan Bahan Aktif** 🔍
1. Pilih metode "🔍 Cari Berdasarkan Bahan Aktif"
2. Pilih filter brand (opsional)
3. Ketik bahan aktif (contoh: "niacinamide", "vitamin c")
4. Klik "🔍 Cari"
5. Lihat daftar produk yang mengandung bahan tersebut

**Contoh bahan aktif populer:**
- **Niacinamide** - Mencerahkan, mengecilkan pori
- **Vitamin C** - Antioksidan, brightening
- **Hyaluronic Acid** - Melembabkan intensif
- **Retinol** - Anti-aging
- **Salicylic Acid** - Mengatasi jerawat

#### **Opsi 2: Pilih dari Daftar Produk** 📋
1. Pilih metode "📋 Pilih dari Daftar Produk"
2. Atur jumlah rekomendasi (3-10)
3. Pilih filter jenis produk (opsional)
4. Filter tampilan berdasarkan brand (opsional)
5. Pilih produk dari dropdown
6. Klik "🎯 Dapatkan Rekomendasi"
7. Lihat produk pilihan dan rekomendasi serupa dengan similarity score

---

## 📈 Hasil & Evaluasi

### Model Performance

#### Product Embedding Model
- **Dimensi Embedding:** 32
- **Vocabulary Size:** 1,405 kata unik
- **Similarity Range:** 0.14 - 0.99
- **Mean Similarity:** 0.66 ± 0.14

#### Brand Classification Model
- **Training Accuracy:** ~95%+
- **Validation Accuracy:** ~90%+
- **Total Parameters:** 650,600
- **Best Epoch:** ~20-30 (dari 100 epochs)

### Similarity Distribution
```
Mean Similarity: 0.6569
Std Similarity:  0.1412
Min Similarity:  0.1427
Max Similarity:  0.9956
```

### Cross-Brand Analysis
- Model mampu menemukan produk serupa **across different brands**
- Similarity tertinggi terjadi pada produk dengan komposisi bahan aktif yang mirip
- Brand diversity dalam rekomendasi menunjukkan sistem tidak bias terhadap brand tertentu

---

## 🔬 Metodologi

### 1. Text Preprocessing
```python
# Cleaning Steps:
1. Lowercase semua text
2. Remove unit ukuran (ml, gr, mg, %)
3. Remove special characters (kecuali %)
4. Stemming menggunakan Sastrawi (Indonesian stemmer)
5. Normalisasi whitespace
```

### 2. Feature Engineering
- **Text Concatenation:** product_name + active_ingredients + benefits
- **Tokenization:** Max 5000 words vocabulary
- **Sequence Length:** Fixed at 50 tokens (padding/truncating)

### 3. Model Training Strategy
```python
# Hyperparameters:
- Optimizer: Adam (lr=0.001)
- Loss: MSE (embedding), Categorical Crossentropy (classification)
- Batch Size: 32
- Epochs: 100 (with early stopping consideration)
- Train/Val Split: 80/20
- Stratification: By brand labels
```

### 4. Similarity Computation
```python
# Cosine Similarity Formula:
similarity = (A · B) / (||A|| × ||B||)

# Where:
- A, B = Product embeddings (32-dim vectors)
- Range: [-1, 1] (in practice: [0, 1] for our data)
```

---

## 🎨 User Interface

### Design Principles
- **Beauty Theme:** Pink gradient, modern card design
- **Responsive Layout:** Wide layout dengan sidebar
- **Interactive Elements:** 
  - Expander untuk tips bahan aktif
  - Selectbox untuk filtering
  - Slider untuk jumlah rekomendasi
  - Button dengan hover effects
- **Visual Feedback:**
  - Success/warning messages
  - Similarity badges dengan color coding
  - Product images dengan fallback

### Color Scheme
- **Primary:** Pink gradient (#ff9a9e → #fecfef)
- **Secondary:** Purple tones (#d63384, #8b4789)
- **Background:** Soft pink gradient (#ffeef8 → #fff5f7)
- **Cards:** White with shadow & pink borders

---

## 📁 File Structure Details

### Input Files (Excel)
```
avoskin.xlsx       # 30 produk Avoskin
emina.xlsx         # 30 produk Emina
glad2glow.xlsx     # 25 produk Glad2Glow
originote.xlsx     # 25 produk Originote
scarlett.xlsx      # 25 produk Scarlett
somethinc.xlsx     # 25 produk Somethinc
wardah.xlsx        # 30 produk Wardah
whitelab.xlsx      # 25 produk Whitelab
```

### Output Files (deployment_files/)
```
skincare_model.h5           # Main neural network model (650KB)
embedding_model.h5          # Embedding extractor (650KB)
product_embeddings.npy      # Product vectors (215x32)
similarity_matrix.npy       # Similarity scores (215x215)
skincare_products.csv       # Processed dataset
tokenizer.pkl               # Keras Tokenizer
label_encoder.pkl           # Brand label encoder
```

### Image Files
```
avoskin/product_name.jpg    # Gambar produk per brand
emina/product_name.png
...
```

---

## 🐛 Troubleshooting

### Error: "File tidak ditemukan"
**Solusi:** Pastikan sudah menjalankan `main.ipynb` dan export artifacts ke `deployment_files/`

### Error: "Model loading failed"
**Solusi:** 
```bash
pip install tensorflow==2.13.0 --upgrade
```

### Warning: "use_container_width deprecated"
**Solusi:** Update Streamlit:
```bash
pip install streamlit --upgrade
```

### Gambar produk tidak muncul
**Solusi:**
1. Pastikan folder brand ada (avoskin/, emina/, dll)
2. Pastikan nama file gambar sesuai dengan product_name
3. Format gambar: .jpg atau .png

---

## 🚀 Future Improvements

### Planned Features
- [ ] **User Rating System** - Feedback untuk improve recommendations
- [ ] **Skin Concern Analysis** - Filter berdasarkan masalah kulit
- [ ] **Price Range Filter** - Budget-friendly recommendations
- [ ] **Ingredient Conflict Detection** - Warning untuk kombinasi berbahaya
- [ ] **Personalized Profile** - Save user preferences
- [ ] **Mobile App Version** - React Native / Flutter
- [ ] **Multi-language Support** - English, Indonesian

### Model Improvements
- [ ] **Transformer-based Embeddings** - BERT/DistilBERT untuk Indonesian
- [ ] **Multi-modal Learning** - Include image features
- [ ] **User Behavior Tracking** - Collaborative filtering
- [ ] **A/B Testing Framework** - Test different recommendation strategies

---

## 👥 Team & Contributions

### Developer
- **Your Name** - Lead Developer, ML Engineer, Data Scientist

### Acknowledgments
- Dataset collected from official brand websites
- Indonesian text processing using **Sastrawi** library
- UI inspiration from modern beauty e-commerce sites

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📞 Contact & Support

- **Email:** naufaalfakhri@gmail.com
- **GitHub:** [@naufalfakhri14](https://github.com/naufalfakhri14)
- **LinkedIn:** [Naufal Fakhri](https://linkedin.com/in/naufaal-fakhri)
  

### Report Issues
Jika menemukan bug atau memiliki saran, silakan buat [Issue di GitHub](https://github.com/naufalfakhri14/deep-learning-product-recommendation-nlp)

---

## 🌟 Star This Repository!

Jika project ini membantu Anda, jangan lupa untuk memberikan ⭐ di GitHub!

---

<div align="center">

**Made with 💖 for Skincare Enthusiasts**

💄 Powered by Deep Learning & AI 💄

</div>
