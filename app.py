import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow import keras
from PIL import Image

# Page config
st.set_page_config(
    page_title="✨ Skincare Recommender",
    page_icon="💄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk desain kecantikan
st.markdown("""
<style>
    /* Background gradient */
    .stApp {
        background: linear-gradient(135deg, #ffeef8 0%, #fff5f7 50%, #ffe8f0 100%);
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 50%, #ffdde1 100%);
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(255, 182, 193, 0.3);
    }
    
    .main-header h1 {
        color: #d63384;
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(214, 51, 132, 0.1);
    }
    
    .main-header p {
        color: #8b4789;
        font-size: 1.2rem;
        font-weight: 400;
    }
    
    /* Card styling */
    .product-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 10px rgba(255, 182, 193, 0.2);
        margin-bottom: 1rem;
        border-left: 4px solid #ff9a9e;
        transition: transform 0.3s ease;
    }
    
    .product-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 20px rgba(255, 182, 193, 0.3);
    }
    
    .product-name {
        color: #d63384;
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .product-brand {
        color: #8b4789;
        font-size: 1rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .similarity-badge {
        background: linear-gradient(135deg, #ff9a9e, #fecfef);
        color: white;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        margin-top: 0.5rem;
    }
    
    .info-label {
        color: #8b4789;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    .info-value {
        color: #5a5a5a;
        font-size: 0.95rem;
        margin-bottom: 0.8rem;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #fff5f7 0%, #ffeef8 100%);
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #ff9a9e, #fecfef);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 25px;
        padding: 0.7rem 2rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 10px rgba(255, 182, 193, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(255, 182, 193, 0.4);
    }
    
    /* Input styling */
    .stTextInput>div>div>input {
        border-radius: 15px;
        border: 2px solid #fecfef;
    }
    
    .stSelectbox>div>div>select {
        border-radius: 15px;
        border: 2px solid #fecfef;
    }
    
    /* Info box */
    .info-box {
        background: linear-gradient(135deg, #fff9fc 0%, #fff5f7 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 4px solid #ff9a9e;
        margin: 1rem 0;
    }
    
    /* Stats container */
    .stats-container {
        background: white;
        padding: 1rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(255, 182, 193, 0.2);
    }
    
    .stat-number {
        font-size: 2rem;
        font-weight: 700;
        color: #d63384;
    }
    
    .stat-label {
        font-size: 0.9rem;
        color: #8b4789;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>✨ Skincare Product Recommender ✨</h1>
    <p>Temukan produk skincare yang cocok berdasarkan bahan aktif favorit Anda</p>
</div>
""", unsafe_allow_html=True)

# Load data
@st.cache_resource
def load_models_and_data():
    try:
        # Load DataFrame
        df = pd.read_csv('deployment_files/skincare_products.csv')
        
        # Load similarity matrix
        similarity_matrix = np.load('deployment_files/similarity_matrix.npy')
        
        # Load model
        model = keras.models.load_model('deployment_files/skincare_model.h5')
        
        return df, similarity_matrix, model
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info("💡 Tip: Jalankan cell terakhir di notebook untuk export data ke folder deployment_files")
        return None, None, None

# Fungsi untuk mencari gambar produk
def get_product_image(product_name, brand):
    """Mencari gambar produk di folder brand"""
    try:
        brand_folder = brand.lower()
        image_path = os.path.join(brand_folder, f"{product_name}.jpg")
        
        # Cek jika file ada
        if os.path.exists(image_path):
            return image_path
        
        # Coba format png
        image_path_png = os.path.join(brand_folder, f"{product_name}.png")
        if os.path.exists(image_path_png):
            return image_path_png
            
        return None
    except Exception as e:
        return None

# Fungsi rekomendasi
def recommend_products(df, similarity_matrix, product_idx, top_n=5, 
                      filter_brand=None, filter_product_type=None, 
                      same_brand_only=False, different_brand_only=False):
    
    if product_idx >= len(df):
        return None
    
    target_product = df.iloc[product_idx]
    target_brand = target_product['brand']
    
    sim_scores = list(enumerate(similarity_matrix[product_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    recommendations = []
    for idx, score in sim_scores[1:]:
        product = df.iloc[idx]
        
        # Apply filters
        if same_brand_only and product['brand'] != target_brand:
            continue
        if different_brand_only and product['brand'] == target_brand:
            continue
        if filter_brand and product['brand'] != filter_brand:
            continue
        if filter_product_type and 'product_type' in df.columns:
            if product.get('product_type', '') != filter_product_type:
                continue
        
        recommendations.append({
            'index': idx,
            'similarity': score,
            'product_name': product['product_name'],
            'brand': product['brand'],
            'product_type': product.get('product_type', 'N/A'),
            'skin_type': product.get('skin_type', 'N/A'),
            'active_ingredients': product.get('active_ingredients', 'N/A'),
            'benefits': product.get('benefits', 'N/A')
        })
        
        if len(recommendations) >= top_n:
            break
    
    return pd.DataFrame(recommendations)

# Load models
df, similarity_matrix, model = load_models_and_data()

if df is not None:
    # Sidebar
    with st.sidebar:
        st.markdown("### 🔍 Metode Pencarian")
        
        # Search method
        search_method = st.radio(
            "Pilih Metode:",
            ["🔍 Cari Berdasarkan Bahan Aktif", "📋 Pilih dari Daftar Produk"],
            label_visibility="visible"
        )
        
        st.markdown("---")
        
        # Conditional filters based on search method
        if search_method == "🔍 Cari Berdasarkan Bahan Aktif":
            # Brand filter only
            st.markdown("#### Filter Brand")
            brand_filter_option = st.selectbox(
                "Pilihan Brand:",
                ["Semua Brand", "Brand yang Sama", "Pilih Brand Spesifik"]
            )
            
            specific_brand = None
            if brand_filter_option == "Pilih Brand Spesifik":
                brands = sorted(df['brand'].unique().tolist())
                specific_brand = st.selectbox("Nama Brand:", brands)
            
            top_n = None  # No limit for ingredient search
            product_type_filter = None
            
        else:  # Pilih dari Daftar Produk
            # Number of recommendations and product type filter
            st.markdown("#### Jumlah Rekomendasi")
            top_n = st.slider("Tampilkan:", 3, 10, 5)
            
            if 'product_type' in df.columns:
                st.markdown("#### Filter Jenis Produk")
                product_types = ['Semua'] + sorted([pt for pt in df['product_type'].unique() if pd.notna(pt)])
                product_type_filter = st.selectbox("Jenis Produk:", product_types)
                if product_type_filter == 'Semua':
                    product_type_filter = None
            else:
                product_type_filter = None
            
            brand_filter_option = "Semua Brand"
            specific_brand = None
        
        # Statistics
        st.markdown("---")
        st.markdown("### 📊 Dataset Info")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="stats-container">
                <div class="stat-number">{len(df)}</div>
                <div class="stat-label">Produk</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="stats-container">
                <div class="stat-number">{df['brand'].nunique()}</div>
                <div class="stat-label">Brand</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("""
        <div class="info-box">
            <p style="margin:0; color:#8b4789; font-size:0.85rem;">
            💡 <strong>Tips:</strong> Sistem ini menggunakan AI Deep Learning 
            untuk menemukan produk skincare dengan bahan aktif yang mirip!
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content
    if search_method == "🔍 Cari Berdasarkan Bahan Aktif":
        st.markdown("### 🔍 Cari Produk Berdasarkan Bahan Aktif")
        
        # Info bahan aktif populer
        with st.expander("💡 Bahan Aktif Populer & Manfaatnya", expanded=False):
            st.markdown("""
            **Brightening & Anti-Aging:**
            - **Niacinamide** - Mencerahkan, mengecilkan pori, mengontrol minyak
            - **Vitamin C** - Antioksidan, brightening, anti-aging
            - **Tranexamic Acid** - Mengurangi dark spot & hiperpigmentasi
            - **Alpha Arbutin** - Mencerahkan kulit secara merata
            - **Retinol** - Anti-aging, regenerasi kulit, mengurangi kerutan
            
            **Hydration & Moisturizing:**
            - **Hyaluronic Acid** - Melembabkan intensif, menjaga kelembaban
            - **Ceramide** - Memperkuat skin barrier
            - **Panthenol (Vitamin B5)** - Menenangkan & melembabkan
            - **Glycerin** - Humektan untuk melembabkan
            
            **Acne & Oil Control:**
            - **Salicylic Acid** - Mengatasi jerawat, eksfoliasi pori
            - **Tea Tree** - Antibakteri, mengatasi jerawat
            - **Centella Asiatica** - Menenangkan kulit berjerawat
            - **Zinc** - Mengontrol minyak berlebih
            
            **Exfoliation:**
            - **AHA (Glycolic/Lactic Acid)** - Eksfoliasi permukaan kulit
            - **BHA (Salicylic Acid)** - Eksfoliasi dalam pori
            - **PHA** - Eksfoliasi lembut untuk kulit sensitif
            """)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            search_ingredient = st.text_input(
                "Masukkan bahan aktif:",
                placeholder="Contoh: niacinamide, vitamin c, hyaluronic acid, retinol...",
                label_visibility="collapsed"
            )
        with col2:
            search_button = st.button("🔍 Cari", use_container_width=True)
        
        if search_button and search_ingredient:
            # Search in active ingredients or product name
            mask = (df['active_ingredients'].str.contains(search_ingredient, case=False, na=False) | 
                   df['product_name'].str.contains(search_ingredient, case=False, na=False))
            matches = df[mask]
            
            # Apply brand filter
            if brand_filter_option == "Brand yang Sama" and len(matches) > 0:
                first_brand = matches.iloc[0]['brand']
                matches = matches[matches['brand'] == first_brand]
            elif brand_filter_option == "Pilih Brand Spesifik" and specific_brand:
                matches = matches[matches['brand'] == specific_brand]
            
            if len(matches) > 0:
                st.success(f"✨ Ditemukan {len(matches)} produk dengan '{search_ingredient}'")
                
                # Show matches
                st.markdown("#### Produk yang Ditemukan:")
                for idx, row in matches.iterrows():
                    with st.expander(f"💄 {row['product_name']} - {row['brand'].upper()}"):
                        # Cari gambar produk
                        search_img = get_product_image(row['product_name'], row['brand'])
                        
                        if search_img:
                            col_img, col_info = st.columns([1, 2])
                            with col_img:
                                try:
                                    img = Image.open(search_img)
                                    st.image(img, use_container_width=True)
                                except:
                                    pass
                            with col_info:
                                st.markdown(f"**Brand:** {row['brand'].upper()}")
                                st.markdown(f"**Jenis Produk:** {row.get('product_type', 'N/A')}")
                                st.markdown(f"**Jenis Kulit:** {row.get('skin_type', 'N/A')}")
                                st.markdown(f"**Bahan Aktif:** {row['active_ingredients']}")
                                st.markdown(f"**Manfaat:** {row['benefits']}")
                        else:
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown(f"**Brand:** {row['brand'].upper()}")
                                st.markdown(f"**Jenis Produk:** {row.get('product_type', 'N/A')}")
                                st.markdown(f"**Jenis Kulit:** {row.get('skin_type', 'N/A')}")
                            with col2:
                                st.markdown(f"**Bahan Aktif:** {row['active_ingredients']}")
                                st.markdown(f"**Manfaat:** {row['benefits']}")
            else:
                st.warning(f"❌ Tidak ditemukan produk dengan bahan aktif '{search_ingredient}'")
                st.info("💡 Coba kata kunci lain atau pilih dari daftar produk")
    
    else:  # Pilih dari daftar
        st.markdown("### 📋 Pilih Produk dari Daftar")
        
        # Tampilkan info filter
        st.info("💡 Rekomendasi akan difilter berdasarkan **jenis produk** dan **kesamaan bahan aktif**")
        
        # Group by brand for display only
        selected_brand_for_list = st.selectbox(
            "Filter tampilan berdasarkan brand:",
            ['Semua Brand'] + sorted(df['brand'].unique().tolist())
        )
        
        if selected_brand_for_list == 'Semua Brand':
            filtered_df = df
        else:
            filtered_df = df[df['brand'] == selected_brand_for_list]
        
        # Product selection
        product_options = [f"{row['product_name']} - {row['brand'].upper()}" 
                          for idx, row in filtered_df.iterrows()]
        selected_product = st.selectbox(
            "Pilih produk:",
            product_options,
            label_visibility="collapsed"
        )
        
        if st.button("🎯 Dapatkan Rekomendasi", use_container_width=True):
            selected_idx = filtered_df.iloc[product_options.index(selected_product)].name
            st.session_state['selected_product_idx'] = selected_idx
            st.session_state['show_recommendations'] = True
            st.session_state['search_method_used'] = 'product_list'  # Mark as from product list
            st.session_state['product_type_filter'] = product_type_filter
            st.session_state['top_n'] = top_n
            st.session_state['brand_filter_option'] = 'Semua Brand'  # No brand filter for product list
            st.session_state['specific_brand'] = None
            st.rerun()
    
    # Show recommendations only if from product list method
    if 'show_recommendations' in st.session_state and st.session_state['show_recommendations']:
        # Only show if search method was 'product_list'
        if st.session_state.get('search_method_used') == 'product_list':
            product_idx = st.session_state['selected_product_idx']
            target_product = df.iloc[product_idx]
            
            st.markdown("---")
            st.markdown("## 💖 Produk yang Anda Pilih")
        
        # Cari gambar produk target
        target_img = get_product_image(target_product['product_name'], target_product['brand'])
        
        if target_img:
            col_img, col_info = st.columns([1, 2])
            with col_img:
                try:
                    img = Image.open(target_img)
                    st.image(img, use_container_width=True)
                except:
                    pass
            with col_info:
                st.markdown(f"""
                <div class="product-card" style="border-left: 4px solid #d63384;">
                    <div class="product-name">✨ {target_product['product_name']}</div>
                    <div class="product-brand">🏷️ {target_product['brand'].upper()}</div>
                    <hr style="border: 1px solid #fecfef; margin: 1rem 0;">
                    <div class="info-label">📦 Jenis Produk:</div>
                    <div class="info-value">{target_product.get('product_type', 'N/A')}</div>
                    <div class="info-label">🧴 Jenis Kulit:</div>
                    <div class="info-value">{target_product.get('skin_type', 'N/A')}</div>
                    <div class="info-label">🧪 Bahan Aktif:</div>
                    <div class="info-value">{target_product['active_ingredients']}</div>
                    <div class="info-label">✨ Manfaat:</div>
                    <div class="info-value">{target_product['benefits']}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="product-card" style="border-left: 4px solid #d63384;">
                <div class="product-name">✨ {target_product['product_name']}</div>
                <div class="product-brand">🏷️ {target_product['brand'].upper()}</div>
                <hr style="border: 1px solid #fecfef; margin: 1rem 0;">
                <div class="info-label">📦 Jenis Produk:</div>
                <div class="info-value">{target_product.get('product_type', 'N/A')}</div>
                <div class="info-label">🧴 Jenis Kulit:</div>
                <div class="info-value">{target_product.get('skin_type', 'N/A')}</div>
                <div class="info-label">🧪 Bahan Aktif:</div>
                <div class="info-value">{target_product['active_ingredients']}</div>
                <div class="info-label">✨ Manfaat:</div>
                <div class="info-value">{target_product['benefits']}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Get filter parameters from session_state or use defaults
        saved_product_type_filter = st.session_state.get('product_type_filter', None)
        saved_top_n = st.session_state.get('top_n', 5)
        saved_brand_filter = st.session_state.get('brand_filter_option', 'Semua Brand')
        saved_specific_brand = st.session_state.get('specific_brand', None)
        
        # Determine filter logic based on saved brand filter option
        same_brand = (saved_brand_filter == "Brand yang Sama")
        use_specific_brand = saved_specific_brand if saved_brand_filter == "Pilih Brand Spesifik" else None
        
        recommendations = recommend_products(
            df, similarity_matrix, product_idx, saved_top_n,
            filter_brand=use_specific_brand,
            filter_product_type=saved_product_type_filter,
            same_brand_only=same_brand,
            different_brand_only=False  # Removed "Brand Berbeda" option
        )
        
        if recommendations is not None and len(recommendations) > 0:
            st.markdown(f"## 🌟 Top {len(recommendations)} Rekomendasi Produk Serupa")
            st.markdown("*Berdasarkan kesamaan bahan aktif dan manfaat*")
            
            for i, row in recommendations.iterrows():
                similarity_percent = row['similarity'] * 100
                
                # Color based on similarity
                if similarity_percent >= 80:
                    color = "#d63384"
                elif similarity_percent >= 60:
                    color = "#ff9a9e"
                else:
                    color = "#fecfef"
                
                # Cari gambar produk
                rec_img = get_product_image(row['product_name'], row['brand'])
                
                if rec_img:
                    col_img, col_info = st.columns([1, 2])
                    with col_img:
                        try:
                            img = Image.open(rec_img)
                            st.image(img, use_container_width=True)
                        except:
                            pass
                    with col_info:
                        st.markdown(f"""
                        <div class="product-card">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div>
                                    <div class="product-name">#{i+1} {row['product_name']}</div>
                                    <div class="product-brand">🏷️ {row['brand'].upper()}</div>
                                </div>
                                <div class="similarity-badge" style="background: linear-gradient(135deg, {color}, #fecfef);">
                                    {similarity_percent:.1f}% Match
                                </div>
                            </div>
                            <hr style="border: 1px solid #fecfef; margin: 1rem 0;">
                            <div class="info-label">📦 Jenis Produk:</div>
                            <div class="info-value">{row['product_type']}</div>
                            <div class="info-label">🧴 Jenis Kulit:</div>
                            <div class="info-value">{row['skin_type']}</div>
                            <div class="info-label">🧪 Bahan Aktif:</div>
                            <div class="info-value">{row['active_ingredients']}</div>
                            <div class="info-label">✨ Manfaat:</div>
                            <div class="info-value">{row['benefits']}</div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="product-card">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <div class="product-name">#{i+1} {row['product_name']}</div>
                                <div class="product-brand">🏷️ {row['brand'].upper()}</div>
                            </div>
                            <div class="similarity-badge" style="background: linear-gradient(135deg, {color}, #fecfef);">
                                {similarity_percent:.1f}% Match
                            </div>
                        </div>
                        <hr style="border: 1px solid #fecfef; margin: 1rem 0;">
                        <div class="info-label">📦 Jenis Produk:</div>
                        <div class="info-value">{row['product_type']}</div>
                        <div class="info-label">🧴 Jenis Kulit:</div>
                        <div class="info-value">{row['skin_type']}</div>
                        <div class="info-label">🧪 Bahan Aktif:</div>
                        <div class="info-value">{row['active_ingredients']}</div>
                        <div class="info-label">✨ Manfaat:</div>
                        <div class="info-value">{row['benefits']}</div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning("❌ Tidak ada rekomendasi yang sesuai dengan filter yang dipilih")
            st.info("💡 Coba ubah filter atau pilih produk lain")
        
            # Reset button
            if st.button("🔄 Cari Produk Lain", use_container_width=True):
                st.session_state['show_recommendations'] = False
                st.rerun()

else:
    st.error("❌ Gagal memuat data. Pastikan semua file yang dibutuhkan tersedia.")
    st.markdown("""
    ### 📝 File yang Dibutuhkan:
    1. `skincare_data.pkl` - Data produk skincare
    2. `similarity_matrix.npy` - Matriks similarity
    3. `skincare_model.h5` - Model TensorFlow
    
    Jalankan notebook terlebih dahulu untuk menghasilkan file-file ini.
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #8b4789; padding: 2rem;">
    <p style="font-size: 0.9rem;">
        💄 Powered by Deep Learning & AI 💄<br>
        Made with 💖 for Skincare Enthusiasts
    </p>
</div>
""", unsafe_allow_html=True)
