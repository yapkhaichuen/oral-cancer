import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageDraw
from fpdf import FPDF
import tempfile

# --- Set Page Config ---
st.set_page_config(page_title="Oral Cancer Detector", layout="wide")

# --- Load Model ---
@st.cache_resource()
def load_model():
    model = tf.keras.models.load_model("model.keras")
    return model

model = load_model()

# --- Load Training History ---
@st.cache_data()
def load_training_history():
    try:
        history = np.load("training_history.npy", allow_pickle=True).item()
        return history
    except FileNotFoundError:
        return None

history = load_training_history()

# --- Function to Predict Cancer ---
def cancerPrediction(img):
    img = img.resize((256, 256))
    img_array = image.img_to_array(img) / 255.0
    input_arr_img = np.expand_dims(img_array, axis=0)
    probability = model.predict(input_arr_img)[0][0]
    prediction = "Cancer" if probability < 0.5 else "Non-Cancer"
    return prediction, probability

# --- Apple-Style Rounded Corners for Images ---
def round_corners(im, radius=40):
    mask = Image.new("L", im.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle((0, 0) + im.size, radius=radius, fill=255)
    im = im.convert("RGBA")
    im.putalpha(mask)
    return im

# --- Generate PDF Report ---
def generate_pdf(patient_name, patient_age, patient_gender, results, history):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # --- Patient Details ---
    pdf.set_font("Arial", style='B', size=16)
    pdf.cell(0, 10, "Oral Cancer Detection Report", ln=True, align='L')
    pdf.ln(5)
    
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"Patient Name: {patient_name}", ln=True)
    pdf.cell(0, 10, f"Age: {patient_age}", ln=True)
    pdf.cell(0, 10, f"Gender: {patient_gender}", ln=True)
    pdf.ln(5)

    # --- Predictions Table ---
    pdf.cell(60, 10, "Image", border=1)
    pdf.cell(60, 10, "Prediction", border=1)
    pdf.cell(60, 10, "Confidence", border=1, ln=True)

    for img_data in results:
        img_name, pred, conf, img = img_data
        img = img.convert("RGB")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_img_file:
            img.save(temp_img_file, format="JPEG")
            temp_img_path = temp_img_file.name

        pdf.cell(60, 10, "", border=1)
        pdf.image(temp_img_path, x=pdf.get_x()-60, y=pdf.get_y(), w=10, h=10)
        pdf.cell(60, 10, pred, border=1)
        pdf.cell(60, 10, f"{conf:.2%}", border=1, ln=True)
        pdf.ln(10)

    # --- Training Performance Charts ---
    if history:
        df = pd.DataFrame({
            "Epoch": range(1, len(history["accuracy"]) + 1),
            "Training Accuracy": history["accuracy"],
            "Validation Accuracy": history["val_accuracy"],
            "Training Loss": history["loss"],
            "Validation Loss": history["val_loss"]
        })

        fig, axes = plt.subplots(1, 2, figsize=(10, 3))

        # Accuracy Graph
        axes[0].plot(df["Epoch"], df["Training Accuracy"], label="Training Accuracy", marker="o")
        axes[0].plot(df["Epoch"], df["Validation Accuracy"], label="Validation Accuracy", marker="o", linestyle="--")
        axes[0].set_title("Accuracy Over Epochs")
        axes[0].set_xlabel("Epochs")
        axes[0].set_ylabel("Accuracy")
        axes[0].legend()

        # Loss Graph
        axes[1].plot(df["Epoch"], df["Training Loss"], label="Training Loss", marker="o")
        axes[1].plot(df["Epoch"], df["Validation Loss"], label="Validation Loss", marker="o", linestyle="--")
        axes[1].set_title("Loss Over Epochs")
        axes[1].set_xlabel("Epochs")
        axes[1].set_ylabel("Loss")
        axes[1].legend()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_chart:
            fig.savefig(temp_chart, bbox_inches="tight", format="png")
            temp_chart_path = temp_chart.name
        plt.close(fig)

        pdf.image(temp_chart_path, x=10, y=None, w=180)

    return pdf.output(dest='S').encode('latin1')

# --- UI Layout ---
st.title("🩺 Oral Cancer Detection App")
st.write("Upload images to analyze oral lesions and predict their likelihood of being cancerous.")

# --- Patient Form ---
with st.form("patient_form"):
    st.subheader("📋 Patient Information")
    patient_name = st.text_input("Patient Name", "")
    patient_age = st.number_input("Age", min_value=1, max_value=120, step=1)
    patient_gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    submit_form = st.form_submit_button("Save Details")

# --- Sidebar: Image Upload ---
st.sidebar.header("Upload Image(s)")
uploaded_files = st.sidebar.file_uploader("Choose image(s)", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

results = []

if uploaded_files:
    for uploaded_file in uploaded_files:
        img = Image.open(uploaded_file).convert("RGB")
        rounded_img = round_corners(img, radius=40)

        col1, col2 = st.columns([1, 2])

        with col1:
            st.image(rounded_img, caption=f"Uploaded: {uploaded_file.name}", use_container_width=True)

        with col2:
            with st.spinner("Analyzing..."):
                result, confidence = cancerPrediction(img)
            results.append((uploaded_file.name, result, confidence, img))
            st.success(f"Prediction: {result}")
            st.progress(float(confidence))
            st.write(f"**Confidence Score:** {confidence:.2%}")

# --- Generate PDF Report Button ---
if results and patient_name:
    pdf_data = generate_pdf(patient_name, patient_age, patient_gender, results, history)
    st.download_button("📄 Download Report", data=pdf_data, file_name="Cancer_Report.pdf", mime="application/pdf")

# --- Interactive Training Performance Charts (Side by Side) ---
if history:
    st.subheader("📊 Model Training Performance")

    df = pd.DataFrame({
        "Epoch": range(1, len(history["accuracy"]) + 1),
        "Training Accuracy": history["accuracy"],
        "Validation Accuracy": history["val_accuracy"],
        "Training Loss": history["loss"],
        "Validation Loss": history["val_loss"]
    })

    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(go.Figure(data=[go.Scatter(x=df["Epoch"], y=df["Training Accuracy"], mode='lines+markers', name="Training Accuracy"),
                                        go.Scatter(x=df["Epoch"], y=df["Validation Accuracy"], mode='lines+markers', name="Validation Accuracy", line=dict(dash='dash'))]), use_container_width=True)

    with col2:
        st.plotly_chart(go.Figure(data=[go.Scatter(x=df["Epoch"], y=df["Training Loss"], mode='lines+markers', name="Training Loss"),
                                        go.Scatter(x=df["Epoch"], y=df["Validation Loss"], mode='lines+markers', name="Validation Loss", line=dict(dash='dash'))]), use_container_width=True)

st.markdown("---")
st.write("Developed with ❤️ using Streamlit & TensorFlow.")
