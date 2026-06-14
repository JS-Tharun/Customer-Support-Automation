import streamlit as st

# ----------------------------------------------------------------------
# Page Configuration
# ----------------------------------------------------------------------

st.set_page_config(
    page_title="Introduction · Customer Support Automation",
    page_icon="🎧",
    layout="centered",
    initial_sidebar_state="expanded",
)


# ------------------------------------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------------------------------------

with st.sidebar:
    st.title("🎧 CSA")
    st.caption("Customer Support Automation")
    st.divider()
    st.caption("XYZ Private Limited · 2025")


# ------------------------------------------------------------------------------------------------
# Hero
# ------------------------------------------------------------------------------------------------

st.title("🎧 Customer Support Automation")
st.caption("XYZ Private Limited · Internal Tool · v1.0")

st.info(
    "This system combines a machine learning classifier with a local LLM to "
    "automatically triage incoming support tickets and generate polished, "
    "empathetic reply drafts — reducing agent workload and response time."
)

st.divider()


# ------------------------------------------------------------------------------------------------
# Problem & Solution
# ------------------------------------------------------------------------------------------------

st.subheader("🚩 The Problem")
st.write(
    "Support teams at XYZ handle hundreds of tickets daily across multiple departments — "
    "billing, technical issues, shipping, account management, and more. "
    "Manually reading, categorising, and drafting replies is time-consuming, inconsistent, "
    "and leaves agents with less time for complex escalations that genuinely need human judgement."
)

st.subheader("✅ The Solution")
st.write(
    "This tool automates the first two steps of the support workflow: "
    "**classification** (which department should own this?) and **reply drafting** "
    "(what should we say back?). Agents review and send — they no longer start from a blank page."
)

st.divider()


# ------------------------------------------------------------------------------------------------
# How It Works — step-by-step
# ------------------------------------------------------------------------------------------------

st.subheader("⚙️ How It Works")

step1, step2, step3 = st.columns(3)

with step1:
    st.metric(label="Step 1", value="Receive Ticket")
    st.caption(
        "An agent pastes the raw customer message into the tool. "
        "The text is cleaned and normalised before processing."
    )

with step2:
    st.metric(label="Step 2", value="Classify")
    st.caption(
        "A fine-tuned SVM model (tracked via MLflow on DagsHub) predicts "
        "the department the ticket belongs to."
    )

with step3:
    st.metric(label="Step 3", value="Draft Reply")
    st.caption(
        "The classified ticket is sent to a local Qwen 2.5 LLM (via Ollama) "
        "which generates a professional, on-brand response."
    )

st.divider()


# ------------------------------------------------------------------------------------------------
# Tech Stack
# ------------------------------------------------------------------------------------------------

st.subheader("🛠️ Tech Stack")

col_ml, col_llm, col_ops = st.columns(3)

with col_ml:
    st.write("**ML & Classification**")
    st.write("- Scikit-learn SVM")
    st.write("- Label Encoder (joblib)")
    st.write("- TF-IDF text features")

with col_llm:
    st.write("**LLM & Generation**")
    st.write("- Qwen 2.5 (local)")
    st.write("- Ollama runtime")
    st.write("- Structured system prompt")

with col_ops:
    st.write("**Tracking & Ops**")
    st.write("- MLflow experiments")
    st.write("- DagsHub model registry")
    st.write("- Streamlit frontend")

st.divider()


# ------------------------------------------------------------------------------------------------
# Model Details
# ------------------------------------------------------------------------------------------------

st.subheader("🤖 Model Details")

col_a, col_b, col_c = st.columns(3)
with col_a:
    st.metric(label="Model", value="Baseline SVM")
with col_b:
    st.metric(label="Registry Alias", value="Challenger")
with col_c:
    st.metric(label="Tracking", value="DagsHub / MLflow")

with st.expander("Why SVM as the baseline?", expanded=False):
    st.write(
        "Support Vector Machines are well-suited to text classification tasks with "
        "moderate dataset sizes. They perform reliably on TF-IDF sparse vectors, "
        "train quickly, and serve as a strong baseline against which neural or "
        "transformer-based classifiers can be benchmarked. "
        "The 'Challenger' alias means this version is under evaluation before "
        "being promoted to 'Champion' in production."
    )

st.divider()


# ------------------------------------------------------------------------------------------------
# Workflow Diagram (ASCII — Streamlit code block)
# ------------------------------------------------------------------------------------------------

st.subheader("📐 Workflow Overview")

st.code(
    """
  Customer Message
        │
        ▼
  ┌─────────────┐
  │ Text Cleaning│   (lowercase, strip whitespace, remove special chars)
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │  SVM Model  │   (MLflow · DagsHub · Challenger alias)
  └──────┬──────┘
         │
         ▼
  ┌──────────────────┐
  │ Department Label │   e.g. Billing / Technical / Shipping / Other
  └──────┬───────────┘
         │
         ▼
  ┌──────────────┐
  │  Qwen 2.5   │   (Ollama · local · structured system prompt)
  └──────┬───────┘
         │
         ▼
  Draft Reply  →  Agent reviews  →  Send to Customer
""",
    language=None,
)

st.divider()


# ------------------------------------------------------------------------------------------------
# Departments Handled
# ------------------------------------------------------------------------------------------------

st.subheader("🗂️ Departments Handled")

departments = {
    "🔧 Technical": "Product bugs, connectivity issues, software errors, and feature malfunctions.",
    "🧾 Billing": "Payment failures, invoice disputes, refund requests, and subscription charges.",
    "📈 Sales": "Pre-purchase enquiries, pricing questions, product demos, and upsell requests.",
    "🤝 Customer Success": "Onboarding support, product adoption guidance, and renewal assistance.",
    "👥 HR": "Employee-related queries, payroll concerns, policy clarifications, and leave requests.",
    "⚖️ Legal": "Contract reviews, compliance concerns, data privacy requests, and dispute notices.",
}

for dept, desc in departments.items():
    with st.expander(dept):
        st.write(desc)

st.divider()


# ------------------------------------------------------------------------------------------------
# CTA
# ------------------------------------------------------------------------------------------------

st.subheader("▶️ Ready to Try It?")
st.write(
    "Head to the **Customer Support Automation** page using the sidebar to classify "
    "a ticket and generate a reply."
)

st.success("Use the sidebar to navigate to the main tool →")