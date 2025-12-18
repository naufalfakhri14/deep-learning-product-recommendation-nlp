# 📊 Flowchart Sistem Rekomendasi Skincare

## 1. Overall System Architecture

```mermaid
graph TB
    Start([Start]) --> Input[📥 Load Excel Files<br/>8 Brand Files]
    Input --> Merge[🔗 Merge DataFrames<br/>215 Products Total]
    Merge --> Clean[🧹 Data Cleaning<br/>Remove Missing Values]
    Clean --> Preprocess[📝 Text Preprocessing<br/>Clean + Stem Text]
    Preprocess --> Tokenize[🔢 Tokenization<br/>Vocab: 5000 words]
    Tokenize --> Pad[📏 Padding<br/>Max Length: 50]
    Pad --> Model1[🧠 Train Embedding Model<br/>Output: 32-dim vectors]
    Pad --> Model2[🏷️ Train Classification Model<br/>8 Brand Classes]
    Model1 --> Embed[📊 Extract Embeddings<br/>215 x 32 matrix]
    Model2 --> Extract[🎯 Extract Embedding Layer<br/>from Trained Model]
    Extract --> Similarity[🔍 Compute Similarity<br/>Cosine Similarity Matrix]
    Embed --> Similarity
    Similarity --> Export[💾 Export Artifacts<br/>Models + Data + Matrix]
    Export --> Deploy[🚀 Streamlit App<br/>Web Interface]
    Deploy --> End([End])
    
    style Start fill:#ff9a9e
    style End fill:#fecfef
    style Deploy fill:#d63384,color:#fff
    style Model1 fill:#8b4789,color:#fff
    style Model2 fill:#8b4789,color:#fff
```

## 2. Data Processing Pipeline

```mermaid
flowchart LR
    A[📂 Raw Excel Data] --> B{Validate<br/>Columns}
    B -->|Valid| C[✅ Load Data]
    B -->|Invalid| D[❌ Error]
    C --> E[🔗 Concat All Brands]
    E --> F[🧹 Drop Missing Values]
    F --> G[📝 Text Cleaning]
    G --> H[🔤 Lowercase]
    H --> I[🗑️ Remove Units]
    I --> J[✂️ Remove Special Chars]
    J --> K[🌿 Stemming Sastrawi]
    K --> L[✨ Clean Dataset]
    
    style A fill:#ffeef8
    style L fill:#d63384,color:#fff
```

## 3. Text Preprocessing Detail

```mermaid
graph TD
    A[Original Text] --> B[Lowercase Conversion]
    B --> C[Remove Units<br/>ml, gr, mg, %]
    C --> D[Remove Special<br/>Characters]
    D --> E[Normalize<br/>Whitespace]
    E --> F[Indonesian<br/>Stemming]
    F --> G[Clean Text]
    
    H[product_name] --> A
    I[active_ingredients] --> A
    J[benefits] --> A
    
    G --> K[Concatenated<br/>Text]
    K --> L[Ready for<br/>Tokenization]
    
    style A fill:#ffeef8
    style G fill:#d63384,color:#fff
    style L fill:#ff9a9e
```

## 4. Neural Network Architecture

```mermaid
graph TB
    Input[Input Sequence<br/>Shape: 50] --> Embed[Embedding Layer<br/>Vocab: 5000, Dim: 128]
    Embed --> Pool[Global Average Pooling<br/>Output: 128]
    Pool --> Dense1[Dense Layer 1<br/>64 units, ReLU]
    Dense1 --> Drop1[Dropout 0.3]
    Drop1 --> Dense2[Dense Layer 2<br/>32 units, ReLU<br/>🎯 Product Embedding]
    
    Dense2 --> Branch{Model Type}
    
    Branch -->|Embedding| Output1[Output<br/>32-dim Vector]
    Branch -->|Classification| Drop2[Dropout 0.3]
    Drop2 --> Dense3[Dense Layer 3<br/>8 units, Softmax]
    Dense3 --> Output2[Brand Prediction<br/>8 Classes]
    
    style Input fill:#ffeef8
    style Dense2 fill:#d63384,color:#fff
    style Output1 fill:#8b4789,color:#fff
    style Output2 fill:#8b4789,color:#fff
```

## 5. Training Process

```mermaid
flowchart TD
    A[Start Training] --> B[Split Data<br/>Train: 80% | Val: 20%]
    B --> C[Initialize Model<br/>Adam Optimizer]
    C --> D{Training Loop<br/>100 Epochs}
    
    D --> E[Forward Pass]
    E --> F[Calculate Loss]
    F --> G[Backpropagation]
    G --> H[Update Weights]
    H --> I{Validate}
    
    I -->|Each Epoch| J[Calculate<br/>Validation Metrics]
    J --> K{Epoch < 100?}
    K -->|Yes| D
    K -->|No| L[Select Best Model<br/>Based on Val Loss]
    
    L --> M[Training Complete]
    M --> N[Evaluate on<br/>Validation Set]
    N --> O[Generate<br/>Confusion Matrix]
    O --> P[Classification<br/>Report]
    P --> Q[End Training]
    
    style A fill:#ff9a9e
    style M fill:#d63384,color:#fff
    style Q fill:#8b4789,color:#fff
```

## 6. Similarity Computation

```mermaid
graph LR
    A[Product A<br/>32-dim Vector] --> C[Cosine<br/>Similarity]
    B[Product B<br/>32-dim Vector] --> C
    C --> D{Similarity<br/>Score}
    D --> E[0.0 - 1.0<br/>Range]
    E --> F[Similarity<br/>Matrix]
    F --> G[215 x 215<br/>Matrix]
    
    style C fill:#d63384,color:#fff
    style G fill:#8b4789,color:#fff
```

## 7. Recommendation System Flow

```mermaid
flowchart TD
    Start([User Input]) --> Method{Search<br/>Method?}
    
    Method -->|Ingredient Search| A1[🔍 Search by<br/>Active Ingredient]
    Method -->|Product List| A2[📋 Select Product<br/>from List]
    
    A1 --> B1[Filter by<br/>Brand Option]
    A2 --> B2[Configure<br/>Filters]
    
    B1 --> C1[Find Matching<br/>Products]
    B2 --> C2[Get Product<br/>Index]
    
    C1 --> D1[Display<br/>Search Results]
    C2 --> D2[Retrieve Similarity<br/>Scores]
    
    D2 --> E[Sort by<br/>Similarity]
    E --> F{Apply<br/>Filters}
    
    F -->|Brand Filter| G[Filter by<br/>Brand]
    F -->|Type Filter| H[Filter by<br/>Product Type]
    F -->|No Filter| I[All Products]
    
    G --> J[Top-N<br/>Selection]
    H --> J
    I --> J
    
    J --> K[Display<br/>Recommendations]
    K --> L{User<br/>Action}
    
    L -->|New Search| Start
    L -->|View Details| M[Show Product<br/>Details + Image]
    L -->|Exit| End([End])
    
    style Start fill:#ff9a9e
    style K fill:#d63384,color:#fff
    style End fill:#8b4789,color:#fff
```

## 8. Streamlit App Flow

```mermaid
stateDiagram-v2
    [*] --> LoadData: App Start
    LoadData --> CheckData: Load Models & Data
    
    CheckData --> Error: Failed
    CheckData --> Dashboard: Success
    Error --> [*]
    
    Dashboard --> MethodSelection: Display UI
    
    MethodSelection --> IngredientSearch: Choose Ingredient
    MethodSelection --> ProductList: Choose Product List
    
    IngredientSearch --> SetBrandFilter: Configure
    ProductList --> SetFilters: Configure
    
    SetBrandFilter --> SearchIngredient: Enter Keyword
    SetFilters --> SelectProduct: Choose from List
    
    SearchIngredient --> DisplayMatches: Show Results
    SelectProduct --> GetRecommendations: Click Button
    
    GetRecommendations --> DisplayResults: Show Recommendations
    DisplayMatches --> DisplayResults: View Products
    
    DisplayResults --> MethodSelection: New Search
    DisplayResults --> [*]: Exit
```

## 9. Data Export Process

```mermaid
flowchart LR
    A[Trained Models] --> E[Export<br/>Process]
    B[Embeddings] --> E
    C[Similarity Matrix] --> E
    D[Processed Data] --> E
    
    E --> F1[📦 skincare_model.h5]
    E --> F2[📦 embedding_model.h5]
    E --> F3[📊 product_embeddings.npy]
    E --> F4[📊 similarity_matrix.npy]
    E --> F5[📄 skincare_products.csv]
    E --> F6[🔧 tokenizer.pkl]
    E --> F7[🔧 label_encoder.pkl]
    
    F1 --> G[deployment_files/]
    F2 --> G
    F3 --> G
    F4 --> G
    F5 --> G
    F6 --> G
    F7 --> G
    
    G --> H[🚀 Ready for<br/>Deployment]
    
    style E fill:#d63384,color:#fff
    style H fill:#8b4789,color:#fff
```

## 10. Complete End-to-End Workflow

```mermaid
graph TB
    subgraph "Phase 1: Data Collection"
    A1[Excel Files<br/>8 Brands] --> A2[Load Data]
    A2 --> A3[Merge & Clean]
    end
    
    subgraph "Phase 2: Preprocessing"
    A3 --> B1[Text Cleaning]
    B1 --> B2[Tokenization]
    B2 --> B3[Padding]
    end
    
    subgraph "Phase 3: Model Training"
    B3 --> C1[Train Embedding<br/>Model]
    B3 --> C2[Train Classification<br/>Model]
    C1 --> C3[Extract Embeddings]
    C2 --> C3
    end
    
    subgraph "Phase 4: Similarity"
    C3 --> D1[Compute Cosine<br/>Similarity]
    D1 --> D2[Generate<br/>Similarity Matrix]
    end
    
    subgraph "Phase 5: Deployment"
    D2 --> E1[Export All<br/>Artifacts]
    E1 --> E2[Build Streamlit<br/>App]
    E2 --> E3[User Interface]
    end
    
    subgraph "Phase 6: Usage"
    E3 --> F1[Search Products]
    E3 --> F2[Get Recommendations]
    F1 --> F3[View Results]
    F2 --> F3
    end
    
    style C1 fill:#d63384,color:#fff
    style C2 fill:#d63384,color:#fff
    style E2 fill:#8b4789,color:#fff
    style F3 fill:#ff9a9e
```

---

## 📝 Keterangan Simbol

| Simbol | Keterangan |
|--------|------------|
| 📥 | Input / Load Data |
| 🔗 | Merge / Combine |
| 🧹 | Cleaning / Preprocessing |
| 📝 | Text Processing |
| 🔢 | Tokenization / Encoding |
| 🧠 | Neural Network / Model |
| 📊 | Data Visualization / Matrix |
| 🔍 | Search / Similarity |
| 💾 | Save / Export |
| 🚀 | Deployment / Launch |
| ✅ | Success / Valid |
| ❌ | Error / Invalid |

---

## 🎯 Cara Menggunakan Flowchart

### Di README.md GitHub
Flowchart dengan format Mermaid akan otomatis ter-render di GitHub. Cukup copy-paste code mermaid ke dalam README.md Anda.

### Di Jupyter Notebook
Anda bisa menggunakan extension seperti `jupyterlab-mermaid` untuk menampilkan flowchart.

### Di Dokumentasi Online
Platform seperti:
- GitHub
- GitLab
- Notion
- Obsidian
- VS Code (dengan extension Markdown Preview Mermaid Support)

Semua mendukung rendering Mermaid diagram secara native.

### Export sebagai Gambar
Gunakan tools online seperti:
- https://mermaid.live
- https://mermaid.ink

Untuk mengkonversi flowchart menjadi PNG/SVG.

---

## 💡 Tips Presentasi

1. **Untuk Presentasi Teknis**: Gunakan flowchart 1, 4, 5, dan 10 (Overview + Detail)
2. **Untuk User Demo**: Gunakan flowchart 7 dan 8 (Recommendation Flow + App Flow)
3. **Untuk Dokumentasi**: Gunakan semua flowchart untuk pemahaman lengkap

---

<div align="center">

**Flowchart dibuat dengan 💖 menggunakan Mermaid.js**

</div>
