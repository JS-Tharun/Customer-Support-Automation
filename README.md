# 🎧 Customer Support Automation

This system combines a machine learning classifier with a local LLM to automatically triage incoming support tickets and generate polished, empathetic reply drafts — reducing agent workload and response time.

---

## 🚩 The Problem

Support teams handle hundreds of tickets daily across multiple departments — billing, technical issues, shipping, account management, and more. Manually reading, categorising, and drafting replies is time-consuming, inconsistent, and leaves agents with less time for complex escalations that genuinely need human judgement.

## ✅ The Solution

This tool automates the first two steps of the support workflow: **classification** (which department should own this?) and **reply drafting** (what should we say back?). Agents review and send — they no longer start from a blank page.

---

## ⚙️ How It Works

| Step | Description |
|------|-------------|
| **1. Receive Ticket** | An agent pastes the raw customer message into the tool. The text is cleaned and normalised before processing. |
| **2. Classify** | A fine-tuned SVM model (tracked via MLflow on DagsHub) predicts the department the ticket belongs to. |
| **3. Draft Reply** | The classified ticket is sent to a local Qwen 2.5 LLM (via Ollama) which generates a professional, on-brand response. |

---

## 🛠️ Tech Stack

**ML & Classification**
- Scikit-learn SVM
- Label Encoder (joblib)
- TF-IDF text features

**LLM & Generation**
- Qwen 2.5 (local)
- Ollama runtime
- Structured system prompt

**Tracking & Ops**
- MLflow experiments
- DagsHub model registry
- Streamlit frontend

---

## 🤖 Model Details

| Model | Registry Alias | Tracking |
|-------|-----------------|----------|
| Baseline SVM | Challenger | DagsHub / MLflow |

<details>
<summary><strong>Why SVM as the baseline?</strong></summary>

Support Vector Machines are well-suited to text classification tasks with moderate dataset sizes. They perform reliably on TF-IDF sparse vectors, train quickly, and serve as a strong baseline against which neural or transformer-based classifiers can be benchmarked. The "Challenger" alias means this version is under evaluation before being promoted to "Champion" in production.

</details>

---

## 📐 Workflow Overview

```
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
```

---

## 🗂️ Departments Handled

| Department | Description |
|------------|--------------|
| 🔧 Technical | Product bugs, connectivity issues, software errors, and feature malfunctions. |
| 🧾 Billing | Payment failures, invoice disputes, refund requests, and subscription charges. |
| 📈 Sales | Pre-purchase enquiries, pricing questions, product demos, and upsell requests. |
| 🤝 Customer Success | Onboarding support, product adoption guidance, and renewal assistance. |
| 👥 HR | Employee-related queries, payroll concerns, policy clarifications, and leave requests. |
| ⚖️ Legal | Contract reviews, compliance concerns, data privacy requests, and dispute notices. |

