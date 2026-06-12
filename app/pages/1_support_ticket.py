import streamlit as st
import os
import requests
import dagshub
import mlflow
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient
import re


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
def load_champion_models():
    prod_models = ['Baseline - SVM']
    models = []
    model_versions = []

    for model in prod_models:
        model_uri = f"models:/{model}@champion"
        loaded_model = mlflow.pyfunc.load_model(model_uri)
        model_version = client.get_model_version_by_alias(name=model, alias="champion").version
        models.append(loaded_model)
        model_versions.append(model_version)
        

    return prod_models, models, model_version

model_names, loaded_models, model_versions = load_champion_models()
st.write(model_names)
st.write(loaded_models)



# --------------------------------------------------------------------------------------------
# Input Preprocessing
# --------------------------------------------------------------------------------------------

# Cleaning the text data
def clean_text(text):

    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.lower()  # Convert all text to lowercase

    return text



st.title("LLM Integration")


url = "http://localhost:11434/api/chat"

support_request = st.text_area(
    "Support Request"
)




submit_button = st.button("Submit Issue")

if submit_button and support_request:
    data = {
        "model": "qwen2.5",
        "stream": False,
        "messages": [
            {
                "role":"system", 
                "content":
                """
                Role: You are a customer support executive
                Audience: Writing for customers who are escalating/raising tickets regarding an issue/incident.
                Tone: Polite, empathetic and professional. Avoid explaining more than once.
                Language: <Detected Language>
                Format: Identify the department to which it should be escalated and follow the below instructions:
                
                You MUST generate the response in the EXACT format below.
                Do NOT add or remove sections.
                Do NOT use Markdown.
                Do NOT include explanations outside this format.

                Subject: <Short issue summary> | Ticket ID: <Generated Ticket ID>

                Dear Customer,

                <Paragraph 1: Empathetic response acknowledging the issue.>

                <Paragraph 2: Mention that the ticket has been escalated to the specified department and the necessary actions taken.>

                <Paragraph 3: Inform the customer that resolution may take some time and offer further assistance in the meantime.>

                Thank you for your patience and understanding.

                Sincerely,

                Tharun J S
                Customer Support Executive
                XYZ Private Limited
                """
            },
            {
                "role":"user", 
                "content":
                f"""
                    Ticket body : {support_request}

                    Raise the ticket to : Identify based on the ticket body

                """
            }
        ]
    }

    response = requests.post(url, json=data)
    response.raise_for_status()

    output = response.json()["message"]["content"]

    st.text(output)