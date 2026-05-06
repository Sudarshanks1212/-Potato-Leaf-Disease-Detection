import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import os
import sqlite3
import datetime
from groq import Groq
from tensorflow.keras.models import load_model
from dotenv import load_dotenv
from fpdf import FPDF

# ==========================================
# 0. Configuration & Database Setup
# ==========================================
load_dotenv("key.env")
GROQ_KEY = os.getenv("GROQ_API_KEY")

if GROQ_KEY:
    client = Groq(api_key=GROQ_KEY)

# Initialize SQLite Database
conn = sqlite3.connect('agri_history.db', check_same_thread=False)
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS scan_history 
             (date TEXT, disease TEXT, confidence REAL)''')
conn.commit()

st.set_page_config(page_title="AgriGuard Pro v4.6", page_icon="🌿", layout="wide")

# FIX: Removed hardcoded #ffffff to allow Theme Switching (Dark/Light)
st.markdown("""
    <style>
    /* Metric Card Styling */
    [data-testid="stMetric"] {
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid rgba(151, 166, 195, 0.2);
    }
    
    /* Sidebar Styling - Set to transparent/inherit to fix Dark Mode issue */
    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(151, 166, 195, 0.2);
    }

    /* About Box Styling */
    .about-box {
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #2e7d32;
        background-color: rgba(46, 125, 50, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 1. Helper Functions
# ==========================================
def save_to_history(disease, confidence):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO scan_history VALUES (?, ?, ?)", (now, disease, confidence))
    conn.commit()

def get_groq_response(prompt_text):
    try:
        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt_text}],
            model="llama-3.3-70b-versatile"
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"AI Advisor offline. Error: {str(e)}"

def create_pdf_report(diagnosis, confidence, roadmap):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 20)
    pdf.cell(200, 10, txt="AgriGuard Diagnostic Report", ln=1, align='C')
    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Date: {datetime.date.today()}", ln=1)
    pdf.cell(200, 10, txt=f"Diagnosis: {diagnosis}", ln=1)
    pdf.cell(200, 10, txt=f"Confidence: {confidence:.2f}%", ln=1)
    pdf.ln(10)
    pdf.multi_cell(0, 10, txt=f"Expert Roadmap:\n{roadmap}")
    return pdf.output(dest='S').encode('latin-1')

# ==========================================
# 2. SIDEBAR - PERMANENT AI CHATBOT (Theme Adaptive)
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/628/628283.png", width=70)
    st.title("🤖 Plant Doctor AI")
    st.caption("Powered by Groq LPU™")
    st.divider()

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I am your AI Agronomist. Ask me anything about potato diseases or farming."}
        ]

    chat_container = st.container(height=450)
    for msg in st.session_state.messages:
        chat_container.chat_message(msg["role"]).write(msg["content"])

    if user_input := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        chat_container.chat_message("user").write(user_input)
        
        with chat_container.chat_message("assistant"):
            response = get_groq_response(user_input)
            st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

# ==========================================
# 3. MAIN DASHBOARD TABS
# ==========================================
st.title("🌿 Potato Leaf Disease Detection and AI Prevention")

tab_diag, tab_hist, tab_stats, tab_about = st.tabs(["🔍 Diagnosis", "📜 History", "📊 Analytics", "ℹ️ System Info"])

# --- TAB 1: DIAGNOSIS ---
with tab_diag:
    col_in, col_out = st.columns([1, 1.2], gap="large")

    with col_in:
        st.subheader("📸 Specimen Scan")
        input_mode = st.radio("Source:", ["Upload Image", "Capture via Camera"], horizontal=True)
        image = None
        if input_mode == "Upload Image":
            file = st.file_uploader("Select Leaf", type=["jpg", "png", "jpeg"])
            if file: image = Image.open(file)
        else:
            cam = st.camera_input("Scanner Active")
            if cam: image = Image.open(cam)
        if image:
            st.image(image, use_container_width=True)

    with col_out:
        st.subheader("📊 Results")
        if image:
            model = load_model("cnn_model.h5", compile=False)
            class_names = ["Early Blight", "Late Blight", "Healthy"]
            
            img_arr = np.array(image.resize((224, 224))) / 255.0
            img_arr = np.expand_dims(img_arr, axis=0)
            
            with st.spinner("Analyzing..."):
                preds = model.predict(img_arr)
                label = class_names[np.argmax(preds[0])]
                conf = np.max(preds[0]) * 100

            st.metric("Diagnosis", label, f"{conf:.1f}%")

            if "last_processed" not in st.session_state or st.session_state.last_processed != label:
                with st.spinner("🤖 Drafting AI Roadmap..."):
                    advice_prompt = f"The potato leaf has {label} ({conf:.1f}% confidence). Give a short 3-step organic and chemical treatment plan."
                    ai_advice = get_groq_response(advice_prompt)
                    st.session_state.current_advice = ai_advice
                    st.session_state.last_processed = label
                    save_to_history(label, conf)
                    st.session_state.messages.append({"role": "assistant", "content": f"**Auto-Diagnosis:** Found **{label}**. Treatment plan generated in the dashboard."})
                st.rerun()

            st.info(st.session_state.get('current_advice', 'Generating...'))
            
            pdf = create_pdf_report(label, conf, st.session_state.get('current_advice', ''))
            st.download_button("📥 Download PDF Report", data=pdf, file_name=f"AgriGuard_{label}.pdf", mime="application/pdf")
        else:
            st.info("Upload a leaf image to start.")

# --- TAB 2: HISTORY ---
with tab_hist:
    st.subheader("📅 Diagnostic Records")
    history_df = pd.read_sql_query("SELECT * FROM scan_history ORDER BY date DESC", conn)
    if not history_df.empty:
        st.dataframe(history_df, use_container_width=True, hide_index=True)
        if st.button("🗑️ Clear All Records"):
            c.execute("DELETE FROM scan_history")
            conn.commit()
            st.rerun()
    else:
        st.write("No records yet.")

# --- TAB 3: ANALYTICS ---
with tab_stats:
    st.subheader("📊 Live Insights")
    if not history_df.empty:
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Disease Distribution**")
            st.bar_chart(history_df['disease'].value_counts())
        with c2:
            st.write("**Recent Confidence Levels**")
            st.line_chart(history_df.set_index('date')['confidence'])
    else:
        st.info("Perform a scan to see analytics.")

# --- TAB 4: ABOUT SYSTEM ---
with tab_about:
    st.markdown("<h2 style='color: #1a73e8;'>🤖 About this AI Agent</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    Welcome to the **AI Plant Pathologist**, a powerful tool designed to democratize agricultural analysis. 
    This application allows you to analyze crop health and ask questions about your data using 
    natural language. **No specialized coding experience is required!**
    """)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### ⚙️ How it Works")
        st.info("""
        1. **Vision Ingestion:** Your leaf specimen image is converted into pixel-data for our neural network.
        2. **AI Processing:** Using the **Groq API (Llama 3.3)**, visual results are translated into actionable treatment roadmaps.
        3. **Execution & Visualization:** The app logs inference results into a local SQLite database and provides tools to track trends.
        """)

    with col_b:
        st.markdown("### ✨ Key Features")
        st.success("""
        - **Natural Language Chat:** Ask the chatbot specific questions about your farming results.
        - **Smart Interactive Visualizations:** Auto-detects diagnosis patterns over time.
        - **Automated Stats:** Instantly view scanning frequency and model confidence summaries.
        - **Exportable Insights:** Download your custom AI-generated reports as professional PDF files.
        """)

    st.divider()
    
    with st.expander("🛠️ View Technical Stack"):
        st.write("""
        - **Vision Model:** TensorFlow (CNN Model)
        - **LLM Engine:** Groq (Llama 3.3 70B model)
        - **Database Engine:** SQLite (Scan Persistence)
        - **Frontend Framework:** Streamlit
        """)

    st.markdown("<p style='text-align: center; color: grey;'>Built with ❤️ for Precision Agriculture.</p>", unsafe_allow_html=True)

st.divider()
st.caption(f"System Version 4.6 | {datetime.datetime.now().strftime('%Y')} Precision Agri Lab")
