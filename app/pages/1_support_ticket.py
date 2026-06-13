import streamlit as st
import os
import requests
import dagshub
import mlflow
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient
import re
import joblib


# ─────────────────────────────────────────────
# Page config — must be first Streamlit call
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Support Ticket Assistant",
    page_icon="🎧",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────

st.markdown("""
<style>
/* ── Imports ───────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Reset & base ──────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Background */
.stApp {
    background: #0d1117;
    color: #e6edf3;
}

/* ── Header ────────────────────────────────── */
.app-header {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 2rem 0 1.5rem;
    border-bottom: 1px solid #21262d;
    margin-bottom: 2rem;
}
.app-header .icon {
    font-size: 2.4rem;
    line-height: 1;
}
.app-header h1 {
    font-size: 1.6rem;
    font-weight: 700;
    color: #f0f6fc;
    margin: 0;
    letter-spacing: -0.02em;
}
.app-header .subtitle {
    font-size: 0.82rem;
    color: #7d8590;
    margin: 2px 0 0;
    font-weight: 400;
}

/* ── Model badge ───────────────────────────── */
.model-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.75rem;
    color: #8b949e;
    margin-bottom: 1.8rem;
    font-family: 'JetBrains Mono', monospace;
}
.model-badge .dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #3fb950;
    box-shadow: 0 0 6px #3fb95099;
}

/* ── Card / Panel ──────────────────────────── */
.card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 1.5rem 1.75rem;
    margin-bottom: 1.25rem;
}
.card-label {
    font-size: 0.72rem;
    font-weight: 600;
    color: #7d8590;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.75rem;
}

/* ── Textarea override ─────────────────────── */
textarea {
    background: #0d1117 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    color: #e6edf3 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9rem !important;
    resize: vertical !important;
    transition: border-color 0.15s !important;
}
textarea:focus {
    border-color: #388bfd !important;
    box-shadow: 0 0 0 3px #388bfd22 !important;
    outline: none !important;
}

/* ── Button ────────────────────────────────── */
.stButton > button {
    background: #238636 !important;
    color: #ffffff !important;
    border: 1px solid #2ea043 !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    padding: 0.55rem 1.4rem !important;
    letter-spacing: 0.01em !important;
    transition: background 0.15s, transform 0.1s !important;
    width: 100% !important;
}
.stButton > button:hover {
    background: #2ea043 !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}

/* ── Classification tag ────────────────────── */
.classification-tag {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: #0d1117;
    border: 1px solid #388bfd44;
    border-radius: 8px;
    padding: 8px 16px;
    font-size: 0.88rem;
    font-weight: 600;
    color: #79c0ff;
    margin-top: 0.5rem;
}
.classification-tag .tag-icon { font-size: 1rem; }

/* ── Response box ──────────────────────────── */
.response-box {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 1.25rem 1.5rem;
    font-size: 0.88rem;
    line-height: 1.75;
    color: #c9d1d9;
    white-space: pre-wrap;
    font-family: 'Inter', sans-serif;
    margin-top: 0.5rem;
}

/* ── Error / warning box ───────────────────── */
.warn-box {
    background: #1c1a00;
    border: 1px solid #9e6a03;
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 0.84rem;
    color: #d29922;
}

/* ── Divider ───────────────────────────────── */
.divider {
    border: none;
    border-top: 1px solid #21262d;
    margin: 1.5rem 0;
}

/* ── Spinner color ─────────────────────────── */
.stSpinner > div > div {
    border-top-color: #388bfd !important;
}

/* ── Hide streamlit chrome ─────────────────── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1rem !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Initialize DagsHub + MLflow
# ─────────────────────────────────────────────

dagshub.init(repo_owner='JS-Tharun', repo_name='Customer-Support-Automation', mlflow=True)

load_dotenv()

os.environ['MLFLOW_TRACKING_USERNAME'] = f"{os.getenv('MLFLOW_USERNAME')}"
os.environ['MLFLOW_TRACKING_PASSWORD'] = f"{os.getenv('MLFLOW_PASSWORD')}"

username = os.getenv("MLFLOW_USERNAME")
password = os.getenv("MLFLOW_PASSWORD")

mlflow.set_tracking_uri(
    f"https://{username}:{password}@dagshub.com/JS-Tharun/Customer-Support-Automation.mlflow"
)
mlflow.set_experiment(os.environ["MLFLOW_EXPERIMENT_NAME"])

client = MlflowClient()


# ─────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────

@st.cache_resource()
def load_classification_model(model_name: str):
    model_uri = f"models:/{model_name}@challenger"
    svm_model = mlflow.pyfunc.load_model(model_uri)
    model_version = client.get_model_version_by_alias(name=model_name, alias='challenger').version
    model_run_id  = client.get_model_version_by_alias(name=model_name, alias='challenger').run_id
    label_encoder_path = mlflow.artifacts.download_artifacts(
        artifact_uri=f"runs:/{model_run_id}/label_encoder.pkl"
    )
    le = joblib.load(label_encoder_path)
    return svm_model, le, model_version


model, le, version = load_classification_model('Baseline - SVM')


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = re.sub(r'[\r\n\t]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()


MODEL_URL = "http://localhost:11434/api/chat"

SYSTEM_PROMPT = """
Role: You are a customer support executive
Audience: Writing for customers who are escalating/raising tickets regarding an issue/incident.
Tone: Polite, empathetic and professional. Avoid explaining more than once.
Language: <Detected Language>
Format: Identify the department to which it should be escalated and follow the below instructions:

You MUST generate the response in the EXACT format below.
Do NOT add or remove sections.
Do NOT use Markdown.
Do NOT include explanations outside this format.
Do NOT repeat what the customer is experiencing/facing

Subject: <Short issue summary> | Ticket ID: <Generated Ticket ID>

Dear Customer,

<Paragraph 1: Empathetic response acknowledging the issue.>

<Paragraph 2: Mention that the ticket has been escalated to the specified department(if not "Other". Else dont mention the department.) and the necessary actions taken.>

<Paragraph 3: Inform the customer that resolution may take some time and offer further assistance in the meantime.>

Thank you for your patience and understanding.

Sincerely,

Tharun J S
Customer Support Executive
XYZ Private Limited
"""


def call_llm(text: str, label) -> str:
    payload = {
        "model": "qwen2.5",
        "stream": False,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Ticket body : {text}\n\n"
                    f"Raise the ticket to : {label} if not \"Other\". Else dont mention the department."
                ),
            },
        ],
    }
    resp = requests.post(MODEL_URL, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()["message"]["content"]


# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────

st.markdown("""
<div class="app-header">
    <div class="icon">🎧</div>
    <div>
        <h1>Support Ticket Assistant</h1>
        <div class="subtitle">Classify and draft a professional reply — instantly.</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="model-badge">
    <span class="dot"></span>
    Baseline SVM &nbsp;·&nbsp; v{version} &nbsp;·&nbsp; challenger
</div>
""", unsafe_allow_html=True)

# ── Input card ──────────────────────────────
st.markdown('<div class="card"><div class="card-label">Customer Message</div>', unsafe_allow_html=True)

support_request = st.text_area(
    label="",
    placeholder="Paste or type the customer's support request here…",
    height=180,
    label_visibility="collapsed",
)

col_btn, col_hint = st.columns([2, 5])
with col_btn:
    submit = st.button("Generate Response →")
with col_hint:
    st.markdown(
        "<div style='padding-top:10px;font-size:0.78rem;color:#7d8590;'>Classifies the issue, then drafts a reply via Qwen 2.5.</div>",
        unsafe_allow_html=True,
    )

st.markdown('</div>', unsafe_allow_html=True)


# ── Processing ──────────────────────────────
if submit:
    if not support_request.strip():
        st.markdown('<div class="warn-box">⚠️  Please enter a support message before submitting.</div>', unsafe_allow_html=True)
    else:
        text  = clean_text(support_request)
        pred  = model.predict([text])
        label = le.inverse_transform(pred)
        label_str = label[0] if hasattr(label, '__iter__') else str(label)

        # Classification result
        st.markdown(f"""
<div class="card">
    <div class="card-label">Classification</div>
    <div class="classification-tag">
        <span class="tag-icon">🏷</span>
        {label_str}
    </div>
</div>
""", unsafe_allow_html=True)

        # LLM response
        st.markdown('<div class="card"><div class="card-label">Draft Reply</div>', unsafe_allow_html=True)

        with st.spinner("Drafting reply…"):
            try:
                output = call_llm(text, label_str)
                st.markdown(f'<div class="response-box">{output}</div>', unsafe_allow_html=True)
            except requests.exceptions.ConnectionError:
                st.markdown(
                    '<div class="warn-box">⚠️  Could not reach the local LLM (Ollama). Make sure it is running on port 11434.</div>',
                    unsafe_allow_html=True,
                )
            except Exception as exc:
                st.markdown(
                    f'<div class="warn-box">⚠️  Unexpected error: {exc}</div>',
                    unsafe_allow_html=True,
                )

        st.markdown('</div>', unsafe_allow_html=True)