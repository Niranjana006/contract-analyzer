"""
Model Loading Utilities
=====================
Functions to load and cache pre-trained models for contract analysis
"""

import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

@st.cache_resource
def load_qa_model(model_name="deepset/roberta-base-squad2"):
    """Load and cache QA model"""
    try:
        qa_pipeline = pipeline(
            "question-answering",
            model=model_name,
            tokenizer=model_name,
            device=0 if torch.cuda.is_available() else -1
        )
        return qa_pipeline
    except Exception as e:
        st.error(f"Error loading QA model {model_name}: {str(e)}")
        # Fallback to a simpler model
        return pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

@st.cache_resource
def load_classification_model(model_name="nlpaueb/legal-bert-base-uncased"):
    """Load and cache classification model"""
    try:
        # Define custom labels for legal contract clauses
        custom_labels = [
            "TERMINATION",
            "PAYMENT", 
            "LIABILITY",
            "INTELLECTUAL_PROPERTY",
            "CONFIDENTIALITY",
            "GENERAL",
            "DISPUTE_RESOLUTION",
            "WARRANTIES",
            "SCOPE_OF_WORK",
            "FORCE_MAJEURE"
        ]
        
        classifier = pipeline(
            "text-classification",
            model=model_name,
            tokenizer=model_name,
            device=0 if torch.cuda.is_available() else -1,
            truncation=True,
            max_length=512
        )
        return classifier
        
    except Exception as e:
        st.error(f"Error loading classification model {model_name}: {str(e)}")
        # Fallback to general BERT model
        try:
            fallback_classifier = pipeline(
                "text-classification",
                model="bert-base-uncased",
                truncation=True,
                max_length=512
            )
            return fallback_classifier
        except:
            return None

@st.cache_resource
def load_legal_ner_model():
    """Load Named Entity Recognition model for legal documents"""
    try:
        ner_pipeline = pipeline(
            "ner",
            model="dbmdz/bert-large-cased-finetuned-conll03-english",
            aggregation_strategy="simple"
        )
        return ner_pipeline
    except Exception as e:
        st.warning(f"Could not load NER model: {str(e)}")
        return None

def get_available_models():
    """Return dictionary of available models"""
    return {
        "qa_models": [
            "deepset/roberta-base-squad2",
            "distilbert-base-cased-distilled-squad",
            "bert-large-uncased-whole-word-masking-finetuned-squad"
        ],
        "classification_models": [
            "nlpaueb/legal-bert-base-uncased",
            "bert-base-uncased",
            "roberta-base"
        ]
    }