import streamlit as st
from sentence_transformers import SentenceTransformer, util
from src.utils import get_logger

logger = get_logger(__name__)

@st.cache_resource
def get_embedding_model():
    logger.info("Loading sentence-transformers model...")
    return SentenceTransformer('all-MiniLM-L6-v2')

def calculate_semantic_similarity(text1, text2):
    """
    Calculate semantic similarity between two texts using cosine similarity.
    Returns a score between 0.0 and 1.0
    """
    if not text1 or not text2:
        return 0.0
        
    try:
        model = get_embedding_model()
        embeddings1 = model.encode(text1, convert_to_tensor=True)
        embeddings2 = model.encode(text2, convert_to_tensor=True)
        
        cosine_scores = util.cos_sim(embeddings1, embeddings2)
        score = cosine_scores[0][0].item()
        # Ensure score is within 0-1 (cosine similarity can be -1 to 1)
        return max(0.0, min(1.0, score))
    except Exception as e:
        logger.error(f"Error calculating semantic similarity: {e}")
        return 0.0
