# app.py
import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import random
import os

# ============= PAGE CONFIG =============
st.set_page_config(page_title="🌱 Plant Disease Detector", layout="wide")

# ============= CONFIG ==================
FRAMEWORK = "dummy"   # change later to "torch" / "tensorflow" / "sklearn"
IMAGE_SIZE = (224, 224)
CLASS_NAMES = ["Healthy", "Powdery Mildew", "Leaf Spot", "Rust"]

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
        path = "model/model.h5"
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
    arr = np.array(img) / 255.0

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
        conf = float(np.max(pred))
        return CLASS_NAMES[idx], conf

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

# ============= STYLES ===================
st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(to right, #d1fae5, #f9fafb);
        color: #064e3b !important;
    }
    [data-testid="stAppViewContainer"] h1,
    [data-testid="stAppViewContainer"] h2,
    [data-testid="stAppViewContainer"] h3,
    [data-testid="stAppViewContainer"] p,
    [data-testid="stAppViewContainer"] div {
        color: #064e3b !important;
    }

    /* ------------------ SIDEBAR ------------------ */
    [data-testid="stSidebar"] {
        background-color: #1e1e26;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] span {
        color: #00ff99 !important;
        font-weight: 600;
    }

    /* ------------------ TABS ------------------ */
    .stTabs [data-baseweb="tab-list"] {
        gap: .5rem;
    }
    .stTabs [data-baseweb="tab"] {
        background: #e7f8ee;
        border-radius: 999px;
        padding: .5rem 1rem;
        font-weight: 600;
        color: #065f46 !important;
    }
    .stTabs [aria-selected="true"] {
        background: #bbf7d0 !important;
        border-bottom: 2px solid #10b981 !important;
    }

    /* ------------------ CARDS ------------------ */
    .card {
        padding: 20px;
        border-radius: 15px;
        background-color: #ecfdf5;
        border: 1px solid #a7f3d0;
        margin-bottom: 20px;
        color: #064e3b !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
        text-align: center;
    }

    /* ------------------ HERO ------------------ */
    .hero { text-align: center; margin: 30px auto; max-width: 900px; }
    .hero-img {
        width: 100%;
        max-height: 250px;
        object-fit: cover;
        border-radius: 12px;
        margin-bottom: 18px;
    }
    .hero-text h1 {
        font-size: 2.3rem;
        font-weight: 800;
        margin-bottom: 12px;
    }
    .hero-text p {
        font-size: 1.15rem;
        line-height: 1.6;
        margin: 0;
    }

    /* ------------------ FILE UPLOADER ------------------ */
    [data-testid="stFileUploader"] section div {
        color: #ffffff !important;
        font-weight: 600;
    }
    [data-testid="stFileUploader"] section button {
        background-color: #10b981 !important;
        color: #ffffff !important;
        font-weight: 700;
        border-radius: 8px;
        border: none;
        padding: 6px 12px;
    }
    [data-testid="stFileUploader"] section button:hover {
        background-color: #059669 !important;
        color: #ffffff !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============= SIDEBAR ==================
st.sidebar.title("🌿 Plant Disease App")
st.sidebar.markdown("Detect plant leaf diseases easily 🌱")

# ============= TABS =====================
home, predict, library, help_tab = st.tabs(["🏠 Home", "🔍 Predict", "📚 Disease Library", "❓ Help"])
with home:
    banner_src = "images/banner.jpg"
    st.markdown(
        f"""
        <div class="hero">
            <img src="{banner_src}" class="hero-img">
            <div class="hero-text">
                <h1>🌱 Plant Disease Detector App</h1>
                <p>Upload a leaf photo → See if it’s healthy or diseased → Get cure & prevention tips.</p>
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
                <p>Our advanced AI model detects diseases with precision and speed.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
            <div class="card">
                <h3>💡 Solutions</h3>
                <p>Get actionable prevention tips and effective treatment guidance.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Call-to-action
    st.markdown(
        """
        <div style="text-align: center; margin-top: 30px;">
            <h3>🚀 Ready to try?</h3>
            <p>Go to the <b>Predict</b> tab and upload a leaf image.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with predict:
    st.header("🔍 Upload a Leaf & Predict Disease")

    file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

    if file:
        image = Image.open(file).convert("RGB")

        # Center the uploaded image
        st.markdown("<h5 style='text-align: center;'>📸 Uploaded Leaf</h5>", unsafe_allow_html=True)
        st.image(image, caption="Uploaded Leaf", width="stretch")

        with st.spinner("Analyzing leaf..."):
            label, conf = predict_disease(image)

        # Centered Prediction Card
        st.markdown(
            f"""
            <div class="card" style="text-align: center;">
                <h3>🌿 Prediction</h3>
                <p><b>Disease:</b> {label}</p>
                <p><b>Confidence:</b> {conf*100:.2f}%</p>
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
                <div class="card" style="text-align: center;">
                    <h3>💊 Cure & Prevention</h3>
                    <p><b>Cure:</b> {cure}</p>
                    <p><b>Prevention:</b> {prev}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.warning("⚠️ No info available for this disease.")

with library:
    st.header("📚 Disease Library")
    search = st.text_input("🔎 Search for a disease")
    df = disease_info.copy()
    if search:
        df = df[df["disease"].astype(str).str.contains(search, case=False, na=False)]
    st.dataframe(df, width="stretch")


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
                <p>Ensure the leaf is well lit. Avoid harsh shadows, glare, or overly dark areas, which can hide important features.</p>
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
                <p>Upload the leaf image in the Predict tab. The AI will predict whether it’s healthy or diseased and show treatment & prevention tips.</p>
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
                <h3>🔍 Use the Predict Tab</h3>
                <p>Go to the <b>Predict</b> tab, upload a clear photo of your plant leaf as instructed, and let our AI analyze it.</p>
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


