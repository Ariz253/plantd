# app.py
import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import random
import os

# ============= PAGE CONFIG =============
st.set_page_config(page_title="🌱 Agro Health", layout="wide",initial_sidebar_state="collapsed")

# ============= CONFIG ==================
FRAMEWORK = "tensorflow"   # change later to "torch" / "tensorflow" / "sklearn"
IMAGE_SIZE = (128,128)
CLASS_NAMES = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 
    'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
    ]

# ============= LOAD MODEL ==============
@st.cache_resource
def load_model():
    if FRAMEWORK == "torch":
        import torch
        path = "model/model.pth"
        if not os.path.exists(path):
            st.error("⚠️ Model file not found at model/model.pth")
            st.stop()
        model = torch.load(path, map_location="cpu")
        model.eval()
        return model

    elif FRAMEWORK == "tensorflow":
        import tensorflow as tf
        path = "model/scratchcnn.keras"
        if not os.path.exists(path):
            st.error("⚠️ Model file not found at model/model.h5")
            st.stop()
        return tf.keras.models.load_model(path)

    elif FRAMEWORK == "sklearn":
        import joblib
        path = "model/model.pkl"
        if not os.path.exists(path):
            st.error("⚠️ Model file not found at model/model.pkl")
            st.stop()
        return joblib.load(path)

    return None  # dummy mode

model = None if FRAMEWORK == "dummy" else load_model()

# ============= PREDICT FUNCTION =========
def predict_disease(image: Image.Image):
    img = image.resize(IMAGE_SIZE).convert("RGB")
    arr = np.array(img)

    if FRAMEWORK == "torch":
        import torch
        x = torch.tensor(arr, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            logits = model(x)
            probs = torch.nn.functional.softmax(logits, dim=1)
            conf, idx = torch.max(probs, 1)
        return CLASS_NAMES[idx.item()], float(conf.item())

    elif FRAMEWORK == "tensorflow":
        pred = model.predict(np.expand_dims(arr, 0), verbose=0)[0]
        idx = int(np.argmax(pred))
        label = CLASS_NAMES[idx]
        return label, None

    elif FRAMEWORK == "sklearn":
        flat = arr.reshape(1, -1)
        pred = model.predict(flat)[0]
        conf = float(np.max(model.predict_proba(flat)[0]))
        return str(pred), conf

    return random.choice(CLASS_NAMES), random.uniform(0.75, 0.98)  # dummy

# ============= DISEASE INFO =============
@st.cache_data
def load_disease_info():
    if not os.path.exists("disease.csv"):
        st.error("⚠️ disease.csv file not found. Please add it to the project root.")
        st.stop()
    return pd.read_csv("disease.csv")

disease_info = load_disease_info()

# ============= DARK PROFESSIONAL STYLES ===================
st.markdown(
    """
    <style>
    /* ------------------ MAIN BACKGROUND ------------------ */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
        color: #e2e8f0 !important;
    }
    
    /* ------------------ TEXT COLORS ------------------ */
    [data-testid="stAppViewContainer"] h1,
    [data-testid="stAppViewContainer"] h2,
    [data-testid="stAppViewContainer"] h3,
    [data-testid="stAppViewContainer"] h4,
    [data-testid="stAppViewContainer"] h5,
    [data-testid="stAppViewContainer"] p,
    [data-testid="stAppViewContainer"] div,
    [data-testid="stAppViewContainer"] span,
    [data-testid="stAppViewContainer"] label {
        color: #e2e8f0 !important;
    }

    /* ------------------ SIDEBAR ------------------ */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #334155 100%);
        border-right: 1px solid #475569;
    }

    
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] span {
        color: #10b981 !important;
        font-weight: 600;
    }

    /* ------------------ TABS ------------------ */
    .stTabs [data-baseweb="tab-list"] {
        gap: .8rem;
        background: rgba(30, 41, 59, 0.6);
        padding: 8px;
        border-radius: 12px;
        backdrop-filter: blur(10px);
    }
    .stTabs [data-baseweb="tab"] {
        background: rgba(71, 85, 105, 0.4);
        border-radius: 8px;
        padding: 12px 20px;
        font-weight: 600;
        color: #cbd5e1 !important;
        border: 1px solid rgba(71, 85, 105, 0.3);
        transition: all 0.3s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(71, 85, 105, 0.6);
        border: 1px solid rgba(16, 185, 129, 0.4);
        color: #10b981 !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
        border: 1px solid #10b981 !important;
        color: #ffffff !important;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);
    }

    /* ------------------ MODERN CARDS ------------------ */
.card {
    min-height: 160px;           /* <- ensures cards in a row are same minimum height */
    height: 100%;
    display: flex;               /* flexbox layout */
    flex-direction: column;      
    justify-content: flex-start; /* top-align content for consistent visual baseline */
    align-items: flex-start;
    padding: 24px;
    border-radius: 16px;
    background: rgba(30, 41, 59, 0.8);
    border: 1px solid rgba(71, 85, 105, 0.4);
    margin-bottom: 20px;
    color: #e2e8f0 !important;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    text-align: left;            /* left text aligns better with top alignment */
    backdrop-filter: blur(10px);
    transition: all 0.3s ease;
    position: relative;
}
.card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 40px rgba(16, 185, 129, 0.2);
    border: 1px solid rgba(16, 185, 129, 0.3);
    cursor: pointer;
}
.card h3 {
    width: 100%;
    margin: 0 0 12px 0;
    background: linear-gradient(135deg, #10b981, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 1.4rem;
}
.card p {
    margin: 0;
    margin-top: 6px;
    color: #cbd5e1 !important;
    line-height: 1.6;
}



    /* ------------------ HERO SECTION ------------------ */
    .hero { 
        text-align: center; 
        margin: 40px auto; 
        max-width: 1000px;
        background: rgba(30, 41, 59, 0.6);
        padding: 40px;
        border-radius: 20px;
        border: 1px solid rgba(71, 85, 105, 0.3);
        backdrop-filter: blur(15px);
    }
    .hero-img {
        width: 100%;
        max-height: 300px;
        object-fit: cover;
        border-radius: 16px;
        margin-bottom: 24px;
        border: 2px solid rgba(16, 185, 129, 0.3);
    }
    .hero-text h1 {
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 16px;
        background: linear-gradient(135deg, #10b981, #34d399, #6ee7b7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .hero-text p {
        font-size: 1.25rem;
        line-height: 1.7;
        margin: 0;
        color: #cbd5e1 !important;
    }

    /* ------------------ FILE UPLOADER ------------------ */
    [data-testid="stFileUploader"] {
        background: rgba(30, 41, 59, 0.6);
        border-radius: 12px;
        padding: 20px;
        border: 2px dashed rgba(16, 185, 129, 0.4);
    }
    [data-testid="stFileUploader"] section div {
        color: #e2e8f0 !important;
        font-weight: 600;
    }
    [data-testid="stFileUploader"] section button {
        background: linear-gradient(135deg, #10b981, #059669) !important;
        color: #ffffff !important;
        font-weight: 700;
        border-radius: 10px;
        border: none;
        padding: 10px 20px;
        transition: all 0.3s ease;
    }
    [data-testid="stFileUploader"] section button:hover {
        background: linear-gradient(135deg, #059669, #047857) !important;
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.3);
    }

    /* ------------------ PREDICTION CARD ------------------ */
    .prediction-card {
        padding: 30px;
        border-radius: 20px;
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(30, 41, 59, 0.9) 100%);
        border: 2px solid rgba(16, 185, 129, 0.4);
        margin: 25px 0;
        color: #e2e8f0 !important;
        box-shadow: 0 12px 40px rgba(16, 185, 129, 0.2);
        text-align: center;
        backdrop-filter: blur(15px);
    }

    /* ------------------ DATAFRAME STYLING ------------------ */
    [data-testid="stDataFrame"] {
        background: rgba(30, 41, 59, 0.8);
        border-radius: 12px;
        border: 1px solid rgba(71, 85, 105, 0.4);
    }

    /* ------------------ SEARCH INPUT ------------------ */
    [data-testid="stTextInput"] input {
        background: rgba(30, 41, 59, 0.8) !important;
        border: 1px solid rgba(71, 85, 105, 0.4) !important;
        border-radius: 8px !important;
        color: #e2e8f0 !important;
    }
    [data-testid="stTextInput"] input:focus {
        border: 2px solid #10b981 !important;
        box-shadow: 0 0 0 2px rgba(16, 185, 129, 0.2) !important;
    }

    /* ------------------ SPINNER ------------------ */
    .stSpinner > div {
        border-top-color: #10b981 !important;
    }

    /* ------------------ HEADER STYLING ------------------ */
    .main-header {
        text-align: center;
        margin: 30px 0;
        padding: 20px;
        background: rgba(30, 41, 59, 0.6);
        border-radius: 12px;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
   

    </style>
    """,
    unsafe_allow_html=True,
)

# # ============= SIDEBAR ==================
# st.sidebar.title("🌿 Agro Health")
# st.sidebar.markdown("Detect plant leaf diseases easily 🌱")

# ============= TABS =====================
home, predict, library, help_tab = st.tabs(["🏠 Home", "🔍 Detect", "📚 Disease Library", "❓ Help"])

with home:
    st.markdown(
        """
        <div class="hero">
            <div class="hero-text">
                <h1>🌱 Agro Health</h1>
                <p>Upload a leaf photo → See if it's healthy or diseased → Get cure & prevention tips.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Feature cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            <div class="card">
                <h3>📸 Upload</h3>
                <p>Upload a clear and sharp photo of the plant leaf to allow for accurate and instant analysis.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="card">
                <h3>🔍 Detect</h3>
                <p>Our advanced AI model detects diseases with great precision and high speed.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
            <div class="card">
                <h3>💡 Solutions</h3>
                <p>Get actionable prevention tips and effective treatment guidance for the diseases.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Call-to-action
    st.markdown(
        """
        <div style="text-align: center; margin-top: 40px; padding: 30px; background: rgba(30, 41, 59, 0.6); border-radius: 16px; border: 1px solid rgba(16, 185, 129, 0.3);">
            <h3 style="background: linear-gradient(135deg, #10b981, #34d399); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">🚀 Ready to try?</h3>
            <p style="color: #cbd5e1;">Go to the <b style="color: #10b981;">Predict</b> tab and upload a leaf image to get started.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

#helper function to clean the disease name
def clean_label(label: str):
    """
    Splits raw class label into plant name and disease name.
    Removes underscores, extra symbols, and formatting noise.
    """
    parts = label.split("___")
    
    plant = parts[0].replace("_", " ").replace("(including sour)", "Cherry").strip()
    disease = parts[1] if len(parts) > 1 else "Healthy"
    
    # fix formatting
    disease = disease.replace("_", " ").replace("-", " ").strip()
    disease = disease.replace("  ", " ")
    
    # special edge cases
    disease = disease.replace("healthy", "Healthy").strip()
    plant = plant.replace("  ", " ")

    return plant, disease

with predict:
    st.markdown('<div class="main-header"><h2>🔍 Upload a Leaf & Detect Disease</h2></div>', unsafe_allow_html=True)

    file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

    if file:
        image = Image.open(file).convert("RGB")

        # Center the uploaded image
        # Center the uploaded image using columns
        col1, col2, col3 = st.columns([1, 1, 1])

        with col2:
            st.markdown("<h5 style='text-align: center; color: #10b981;'>📸 Uploaded Leaf</h5>", unsafe_allow_html=True)
            st.image(image, caption="Uploaded Leaf", width=400)

        with st.spinner("Analyzing leaf..."):
            label, conf = predict_disease(image)
            plant, disease = clean_label(label)

        # Build optional confidence HTML
        if conf is not None:
            conf_html = f"""
                <p style="font-size: 1.2rem;">
                    <b>Confidence Level:</b> 
                    <span style="color: #34d399;">{conf*100:.2f}%</span>
                </p>
            """
        else:
            conf_html = ""
        
        # Centered Prediction Card
        st.markdown(
            f"""
            <div class="prediction-card">
                <h3 style="color: #10b981; margin-bottom: 20px;">🌿 AI Detection Results</h3>
                <p style="font-size: 1.2rem;">
                    <b>Plant:</b> 
                    <span style="color: #34d399;">{plant}</span>
                </p>
                <p style="font-size: 1.2rem;">
                    <b>Disease:</b> 
                    <span style="color: #34d399;">{disease}</span>
                </p>
                {conf_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Cure & Prevention under Prediction
        match = disease_info[disease_info["disease"].astype(str) == str(label)]
        if not match.empty:
            cure = match["cure"].values[0]
            prev = match["prevention"].values[0]
            st.markdown(
                f"""
                <div class="card" style="text-align: left; margin-top: 25px;">
                    <h3 style="text-align: center; margin-bottom: 20px;">💊 Treatment & Prevention Guide</h3>
                    <div style="margin-bottom: 15px;">
                        <h4 style="color: #34d399; margin-bottom: 8px;">🩹 Treatment:</h4>
                        <p style="line-height: 1.6;">{cure}</p>
                    </div>
                    <div>
                        <h4 style="color: #34d399; margin-bottom: 8px;">🛡️ Prevention:</h4>
                        <p style="line-height: 1.6;">{prev}</p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.warning("⚠️ No treatment information available for this disease in our database.")

#custom disease library dataframe styler
def style_dark(df: pd.DataFrame):
    return (
        df.style
        .set_properties(**{
            "background-color": "rgba(30, 41, 59, 0.8)",
            "color": "#e2e8f0",
            "border-color": "#475569",
            "border-width": "1px",
            "border-style": "solid",
        })
        .set_table_styles([
            {"selector": "thead th", "props": [
                ("background-color", "rgba(51, 65, 85, 0.9)"),
                ("color", "#10b981"),
                ("font-weight", "bold"),
                ("border-color", "#475569"),
                ("padding", "8px"),
            ]},
            {"selector": "tbody tr:hover", "props": [
                ("background-color", "rgba(16, 185, 129, 0.15)"),
            ]},
            {"selector": "tbody td", "props": [
                ("padding", "10px"),
            ]}
        ])
        .hide(axis="index")
    )

with library:
    st.markdown('<div class="main-header"><h2>📚 Disease Knowledge Library</h2></div>', unsafe_allow_html=True)
    
    # Center the search input too
    col1, col2, col3 = st.columns([0.05, 0.9, 0.05])
    
    with col2:
        search = st.text_input("🔎 Search for a disease", placeholder="Type disease name...")
    
    df = disease_info.copy()
    if search:
        df = df[df["disease"].astype(str).str.contains(search, case=False, na=False)]
    
    styled_df = style_dark(df)

    # Use columns to center the table
    col1, col2, col3 = st.columns([0.05, 0.9, 0.05])
    
    with col2:
        st.markdown("""
        <style>
        .styled-table-wrapper {
            height: 500px;
            overflow-y: auto;
            overflow-x: auto;
            border: 1px solid rgba(71, 85, 105, 0.4);
            border-radius: 12px;
            padding: 10px;
            background: rgba(30, 41, 59, 0.8);
            margin-top: 15px;
            display: flex;
            justify-content: center;
            align-items: flex-start;
        }
        .styled-table-wrapper table {
            width: 100% !important;
            margin: 0 auto;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Display styled dataframe HTML
        st.markdown(f'<div class="styled-table-wrapper">{styled_df.to_html()}</div>', unsafe_allow_html=True)

with help_tab:
    st.header("❓ Help & User Guide")

    # Leaf photo tips
    st.markdown(
        """
        <div class="hero">
            <h1>🌱 How to Take Good Leaf Photos</h1>
            <p>Follow these simple steps to get the best results when detecting plant diseases.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div class="card">
                <h3>📸 Upload a Clear Leaf Image</h3>
                <p>Photograph a single leaf against a <b>plain, well-lit background</b>, avoiding distortions and clutter.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown(
            """
            <div class="card">
                <h3>💡 Good Lighting</h3>
                <p>Ensure that the leaf is well litin the photo. Avoid harsh shadows, glare, or overly dark areas, which can hide important features.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div class="card">
                <h3>🌿 Focus on the Leaf</h3>
                <p>Make the leaf the main subject. Avoid including multiple leaves or busy backgrounds.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown(
            """
            <div class="card">
                <h3>🚀 Get Your Results</h3>
                <p>Upload the leaf image in the Detect tab. The AI will detect whether it’s healthy or diseased and show treatment & prevention tips.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        """
        <div class="card" style="text-align: center;">
            <h3>✅ Tip for Best Accuracy</h3>
            <p>Clear, centered, and well-lit leaf photos on a plain background give the most accurate results.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # App usage instructions
    st.markdown(
        """
        <div class="hero">
            <h1>📱 How to Use the App</h1>
            <p>Quick guide to analyzing leaves and exploring plant disease information.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div class="card">
                <h3>🔍 Use the Detect Tab</h3>
                <p>Go to the <b>Detect</b> tab, upload a clear photo of your plant leaf as instructed, and let our AI analyze it.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown(
            """
            <div class="card">
                <h3>🌿 Get Results & Remedies</h3>
                <p>If a disease is detected, the app shows the disease name, confidence level, and detailed treatment & prevention tips.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div class="card">
                <h3>📚 Explore Disease Library</h3>
                <p>Go to the <b>Disease Library</b> tab to search for other diseases, and get tips for their prevention and cure.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown(
            """
            <div class="card">
                <h3>💡 Take Action</h3>
                <p>Use the information from both tabs to protect your plants and prevent the disease from spreading further.</p>
            </div>
            """,
            unsafe_allow_html=True
        )


