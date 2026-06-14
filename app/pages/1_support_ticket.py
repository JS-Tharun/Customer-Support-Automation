import streamlit as st
import os
import requests
import dagshub
import mlflow
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient
import re
import joblib


# ----------------------------------------------------------------------
# Page Configuration
# ----------------------------------------------------------------------

st.set_page_config(
    page_title="Customer Support Automation",
    page_icon="🎧",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# ----------------------------------------------------------------------
# Initialize DagsHub connection and MLflow tracking
# ----------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# Load the champion model from MLflow and cache it for streamlit
# -----------------------------------------------------------------------------

@st.cache_resource()
def load_classification_model(model_name: str) -> object:
    model = model_name
    model_uri = f"models:/{model}@challenger"
    svm_model = mlflow.pyfunc.load_model(model_uri)
    model_version = client.get_model_version_by_alias(name=model, alias='challenger').version
    model_run_id = client.get_model_version_by_alias(name=model, alias='challenger').run_id
    label_encoder_path = mlflow.artifacts.download_artifacts(
        artifact_uri=f"runs:/{model_run_id}/label_encoder.pkl"
    )
    le = joblib.load(label_encoder_path)

    return svm_model, le, model_version


# --------------------------------------------------------------------------------------------
# Input Preprocessing
# --------------------------------------------------------------------------------------------

def clean_text(text):
    text = re.sub(r'[\r\n\t]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip().lower()
    return text


# ------------------------------------------------------------------------------------------------
# Sidebar — Model Info
# ------------------------------------------------------------------------------------------------

with st.sidebar:
    st.title("ℹ️ About")
    st.markdown(
        "This tool classifies incoming customer support requests and "
        "generates a professional response draft using an LLM."
    )
    st.divider()

    with st.spinner("Loading model…"):
        model, le, version = load_classification_model('Baseline - SVM')

    st.success("Model ready")
    st.caption(f"**Model:** Baseline – SVM")
    st.caption(f"**Alias:** Challenger")
    st.caption(f"**Version:** {version}")
    st.divider()
    st.caption("XYZ Private Limited · Customer Support Automation")


# ------------------------------------------------------------------------------------------------
# Main — Header
# ------------------------------------------------------------------------------------------------

st.title("🎧 Customer Support Automation")
st.caption(
    "Paste a customer support ticket below. "
    "The system will classify the issue and draft a polished reply."
)
st.divider()


# ------------------------------------------------------------------------------------------------
# Main — Input
# ------------------------------------------------------------------------------------------------

st.subheader("1 · Describe the Issue")

support_request = st.text_area(
    label="Support request",
    placeholder="e.g. My order #12345 hasn't arrived after 10 days and tracking shows no updates…",
    height=160,
    label_visibility="collapsed",
)

char_count = len(support_request.strip())
if char_count > 0:
    st.caption(f"{char_count} characters")

submit_button = st.button(
    "Classify & Generate Reply",
    type="primary",
    use_container_width=True,
    disabled=(char_count == 0),
)


# ------------------------------------------------------------------------------------------------
# Main — Processing & Output
# ------------------------------------------------------------------------------------------------

model_url = "http://localhost:11434/api/chat"

if submit_button and support_request.strip():

    text = clean_text(support_request)

    # --- Classification ---
    st.divider()
    st.subheader("2 · Classification Result")

    with st.spinner("Classifying ticket…"):
        pred = model.predict([text])
        label = le.inverse_transform(pred)

    col_label, col_dept = st.columns([1, 2])
    with col_label:
        st.metric(label="Department", value=str(label[0]))
    with col_dept:
        if str(label[0]).lower() != "other":
            st.info(f"🔀 This ticket will be escalated to **{label[0]}**.")
        else:
            st.info("📋 This ticket will be handled by the general support queue.")

    # --- LLM Response ---
    st.divider()
    st.subheader("3 · Draft Response")

    system_prompt = """
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

    data = {
        "model": "qwen2.5",
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Ticket body : {text}\n\n"
                    f"Raise the ticket to : {label} if not \"Other\". Else dont mention the department."
                ),
            },
        ],
    }

    with st.spinner("Generating reply…"):
        response = requests.post(model_url, json=data)
        response.raise_for_status()
        output = response.json()["message"]["content"]

    st.text_area(
        label="Generated reply (ready to copy)",
        value=output,
        height=320,
        label_visibility="visible",
    )

    st.download_button(
        label="⬇️ Download Reply as .txt",
        data=output,
        file_name="support_reply.txt",
        mime="text/plain",
        use_container_width=True,
    )

    # --- Log summary ---
    st.divider()
    with st.expander("📋 Submission Summary", expanded=False):
        st.write("**Original request:**")
        st.write(support_request)
        st.write("**Cleaned input sent to model:**")
        st.code(text, language=None)
        st.write("**Predicted department:**", str(label[0]))

elif submit_button and not support_request.strip():
    st.warning("Please enter a support request before submitting.")