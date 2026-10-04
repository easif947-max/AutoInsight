import streamlit as st
import pandas as pd
import os

from database import init_db, register_user, authenticate_user, save_analysis, get_user_history
from main_crew import run_autoinsight_pipeline

st.set_page_config(
    page_title="AutoInsight 2.0 - AI Data Analysis Platform",
    page_icon="⚡",
    layout="wide"
)

# Initialize Database
init_db()

# Custom CSS: Black, Gold, and Purple Theme
st.markdown("""
<style>
    /* Dark Backgrounds */
    .stApp {
        background-color: #0b0712;
        color: #e2d9f3;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #120c1f !important;
        border-right: 1px solid #3b1d60;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #7b2cbf 0%, #ffb703 100%);
        color: #ffffff;
        font-weight: bold;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(255, 183, 3, 0.4);
    }
    
    /* Input Fields */
    .stTextInput input {
        background-color: #1c132b;
        color: #ffd166;
        border: 1px solid #5a189a;
        border-radius: 6px;
    }
    
    /* Cards and Containers */
    .css-card {
        background: #180e29;
        border: 1px solid #ffb703;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #ffb703 !important;
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_email' not in st.session_state:
    st.session_state.user_email = ""

# --- AUTHENTICATION PORTAL ---
if not st.session_state.logged_in:
    st.markdown("<h1 style='text-align: center; color: #ffb703;'>⚡ AutoInsight 2.0</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Multi-Agent Autonomous Data Analysis Engine</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        auth_tab1, auth_tab2 = st.tabs(["🔐 Login", "📝 Sign Up"])
        
        with auth_tab1:
            login_email = st.text_input("Email Address", key="log_email")
            login_pass = st.text_input("Password", type="password", key="log_pass")
            if st.button("Access Platform", use_container_width=True):
                if authenticate_user(login_email, login_pass):
                    st.session_state.logged_in = True
                    st.session_state.user_email = login_email.lower()
                    st.rerun()
                else:
                    st.error("Invalid credentials. Please check your details.")

        with auth_tab2:
            reg_email = st.text_input("Email Address", key="reg_email")
            reg_pass = st.text_input("Password", type="password", key="reg_pass")
            if st.button("Create Account", use_container_width=True):
                if register_user(reg_email, reg_pass):
                    st.success("Account created successfully! Please log in.")
                else:
                    st.error("Email already registered.")
    st.stop()

# --- MAIN LOGGED-IN PLATFORM ---

# Sidebar Profile & Navigation
st.sidebar.markdown(f"### 👤 Active User\n`{st.session_state.user_email}`")
if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.session_state.user_email = ""
    st.rerun()

st.sidebar.divider()

navigation = st.sidebar.radio(
    "Navigation Workspace:",
    [
        "📊 Dashboard & Upload",
        "🤖 ML Insights & Clustering",
        "📈 Predictive Forecasting",
        "📝 Business Reports",
        "🔍 Keyword & Synonym Explorer",
        "📜 My Saved History"
    ]
)

st.title("AutoInsight 2.0")

# --- TAB 1: DASHBOARD & UPLOAD ---
if navigation == "📊 Dashboard & Upload":
    st.subheader("Dataset Ingestion Zone")
    uploaded_file = st.file_uploader("Drop your dataset (CSV)", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("### Raw Dataset Preview")
        st.dataframe(df.head(), use_container_width=True)
        
        temp_csv_path = f"temp_{st.session_state.user_email.replace('@','_')}.csv"
        df.to_csv(temp_csv_path, index=False)
        
        if st.button("⚡ Execute Multi-Agent Analysis Pipeline", use_container_width=True):
            status_box = st.status("🤖 Multi-Agent Workflow Initiated...", expanded=True)
            
            def update_status(msg):
                status_box.write(msg)
                
            try:
                results = run_autoinsight_pipeline(temp_csv_path, status_callback=update_status)
                status_box.update(label="✅ Analysis Completed Successfully!", state="complete")
                
                # Extract results
                tasks_outputs = results.tasks_output
                prof_out = str(tasks_outputs[0]) if len(tasks_outputs) > 0 else "N/A"
                fore_out = str(tasks_outputs[1]) if len(tasks_outputs) > 1 else "N/A"
                rep_out = str(tasks_outputs[2]) if len(tasks_outputs) > 2 else "N/A"
                kw_out = str(tasks_outputs[3]) if len(tasks_outputs) > 3 else "N/A"
                
                # Persist strictly for logged-in user
                save_analysis(
                    st.session_state.user_email,
                    uploaded_file.name,
                    prof_out,
                    fore_out,
                    rep_out,
                    kw_out
                )
                
                st.session_state['latest_results'] = {
                    'profiling': prof_out,
                    'forecast': fore_out,
                    'report': rep_out,
                    'keyword': kw_out
                }
                
                st.success("Results saved to your personal database history!")
            except Exception as e:
                st.error(f"Error during agent pipeline run: {str(e)}")

# --- TAB 2: ML INSIGHTS ---
elif navigation == "🤖 ML Insights & Clustering":
    st.subheader("Data Profiling & ML Insights Agent Output")
    if 'latest_results' in st.session_state:
        st.markdown(st.session_state['latest_results']['profiling'])
    else:
        st.info("No active pipeline execution found. Run the analysis from the 'Dashboard & Upload' workspace.")

# --- TAB 3: PREDICTIVE FORECASTING ---
elif navigation == "📈 Predictive Forecasting":
    st.subheader("Forecasting & Trend Agent Output")
    if 'latest_results' in st.session_state:
        st.markdown(st.session_state['latest_results']['forecast'])
    else:
        st.info("No active pipeline execution found. Upload dataset and run pipeline first.")

# --- TAB 4: BUSINESS REPORTS ---
elif navigation == "📝 Business Reports":
    st.subheader("Executive Business Report")
    if 'latest_results' in st.session_state:
        st.markdown(st.session_state['latest_results']['report'])
    else:
        st.info("No active pipeline execution found.")

# --- TAB 5: KEYWORD EXPLORER ---
elif navigation == "🔍 Keyword & Synonym Explorer":
    st.subheader("Semantic Synonym & Keyword Filter Engine")
    if 'latest_results' in st.session_state:
        st.markdown(st.session_state['latest_results']['keyword'])
    else:
        st.info("No active pipeline execution found.")

# --- TAB 6: MY SAVED HISTORY ---
elif navigation == "📜 My Saved History":
    st.subheader("Isolated Historical Execution Runs")
    history = get_user_history(st.session_state.user_email)
    
    if history:
        for idx, record in enumerate(history):
            with st.expander(f"📁 Dataset: {record[0]} | Executed At: {record[1]}"):
                st.markdown("#### 1. Data Profiling & ML Insights")
                st.text(record[2])
                st.markdown("#### 2. Forecasting & Trend Projections")
                st.text(record[3])
                st.markdown("#### 3. Executive Report")
                st.markdown(record[4])
                st.markdown("#### 4. Keyword & Synonym Expansion")
                st.text(record[5])
    else:
        st.warning("No saved analysis runs found for your account.")
