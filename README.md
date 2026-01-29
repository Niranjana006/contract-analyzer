# 🤖 Contract Analyzer

AI-powered contract analysis tool that extracts key terms, obligations, and risks from PDF contracts using NLP.

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live_Demo-brightgreen)](https://niranjana006-contract-analyzer.streamlit.app/)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Repo-blue?logo=github)](https://github.com/niranjana006/contract-analyzer)

## ✨ Features

- 📄 **PDF Contract Upload** - Drag & drop contracts
- ❓ **Smart Q&A** - Ask about payments, IP ownership, termination, liability
- 🧠 **NLP Models** - Transformers-based contract understanding
- 📊 **Key Terms Extraction** - Auto-identifies critical clauses
- ⚡ **Streamlit UI** - Clean, responsive interface

## 🎯 Demo

| Question | Answer |
|----------|--------|
| What is the payment amount? | `$50,000 quarterly` |
| Who owns the IP? | `Client retains all IP rights` |
| Termination notice? | `30 days written notice` |

## 🛠 Tech Stack

Frontend: Streamlit 1.49.1

Backend: Python 3.11

NLP: Transformers + HuggingFace

PDF: PyPDF2, python-docx

Data: pandas, numpy

Deploy: Streamlit Cloud


## 🚀 Quick Start (Local)

git clone https://github.com/niranjana006/contract-analyzer.git

cd contract-analyzer

pip install -r requirements.txt

streamlit run minimal_app.py

📈 Performance
Cold start: 15s (model loading)
Analysis time: 5-10s per contract
Supported formats: PDF, DOCX
Models: Legal-BERT fine-tuned
