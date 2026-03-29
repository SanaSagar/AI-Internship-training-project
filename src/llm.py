import requests
import json
import streamlit as st
from src.utils import get_logger

logger = get_logger(__name__)

OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"

@st.cache_data(ttl=60)
def get_available_models():
    """Fetch available models from local Ollama instance."""
    try:
        response = requests.get(OLLAMA_TAGS_URL, timeout=2)
        if response.status_code == 200:
            models = response.json().get('models', [])
            return [model['name'] for model in models]
        return []
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to connect to Ollama: {e}")
        return []

def generate_response(prompt, model="llama3", temperature=0.7, max_tokens=None):
    """Generate a response using the local Ollama API."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature
        }
    }
    if max_tokens:
        payload["options"]["num_predict"] = max_tokens

    try:
        response = requests.post(OLLAMA_API_URL, json=payload)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error generating response from model {model}: {e}")
        return f"Error: Unable to generate response. Check if Ollama is running and model '{model}' exists."
