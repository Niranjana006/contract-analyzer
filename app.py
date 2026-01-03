"""
🚀 Complete Contract Analyzer Streamlit App
==========================================
A full-featured contract analysis tool using pre-trained models
"""

import streamlit as st
import pandas as pd
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import re
import io
import os
import sys

# Add utils directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

try:
    from text_processing import clean_contract_text, split_into_clauses, extract_key_info
    from model_loader import load_qa_model, load_classification_model
except ImportError:
    st.error("Missing utility files. Please ensure all files are in the correct directories.")

# Page Configuration
st.set_page_config(
    page_title="Contract Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    
    .feature-box {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .clause-highlight {
        padding: 0.5rem;
        margin: 0.5rem 0;
        border-radius: 5px;
        border-left: 3px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'contract_text' not in st.session_state:
    st.session_state.contract_text = ""
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'qa_history' not in st.session_state:
    st.session_state.qa_history = []

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📄 Smart Contract Analyzer</h1>
        <p>AI-powered contract analysis using state-of-the-art legal NLP models</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.title("🔧 Controls")
        
        # Model Selection
        st.subheader("Model Settings")
        qa_model_choice = st.selectbox(
            "QA Model",
            ["deepset/roberta-base-squad2", "distilbert-base-cased-distilled-squad"],
            help="Choose the question-answering model"
        )
        
        classification_model_choice = st.selectbox(
            "Classification Model", 
            ["nlpaueb/legal-bert-base-uncased", "bert-base-uncased"],
            help="Choose the text classification model"
        )
        
        # Analysis Options
        st.subheader("Analysis Options")
        enable_key_extraction = st.checkbox("Extract Key Information", value=True)
        enable_clause_analysis = st.checkbox("Detailed Clause Analysis", value=True)
        confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5)
        
        # Clear button
        if st.button("🗑️ Clear All", type="secondary"):
            st.session_state.contract_text = ""
            st.session_state.analysis_results = None
            st.session_state.qa_history = []
            st.rerun()

    # Main Content Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📝 Upload Contract", "❓ Ask Questions", "🔍 Clause Analysis", "📊 Results Dashboard"])
    
    with tab1:
        handle_contract_upload()
    
    with tab2:
        handle_question_answering(qa_model_choice, confidence_threshold)
    
    with tab3:
        handle_clause_analysis(classification_model_choice, enable_clause_analysis, confidence_threshold)
    
    with tab4:
        handle_results_dashboard(enable_key_extraction)

def handle_contract_upload():
    st.header("📄 Contract Input")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # File upload
        uploaded_file = st.file_uploader(
            "Upload Contract File",
            type=['txt', 'pdf', 'docx'],
            help="Supported formats: TXT, PDF, DOCX"
        )
        
        if uploaded_file is not None:
            try:
                if uploaded_file.type == "text/plain":
                    contract_text = str(uploaded_file.read(), "utf-8")
                elif uploaded_file.type == "application/pdf":
                    contract_text = extract_text_from_pdf(uploaded_file)
                elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    contract_text = extract_text_from_docx(uploaded_file)
                else:
                    st.error("Unsupported file format")
                    return
                
                st.session_state.contract_text = clean_contract_text(contract_text)
                st.success(f"✅ Contract uploaded successfully! ({len(contract_text)} characters)")
                
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
        
        # Text area for manual input
        manual_text = st.text_area(
            "Or paste contract text here:",
            value=st.session_state.contract_text,
            height=300,
            placeholder="Paste your contract text here..."
        )
        
        if manual_text != st.session_state.contract_text:
            st.session_state.contract_text = clean_contract_text(manual_text)
    
    with col2:
        st.subheader("📊 Document Stats")
        
        if st.session_state.contract_text:
            text = st.session_state.contract_text
            
            # Display metrics
            st.markdown("""
            <div class="metric-card">
                <h3>{}</h3>
                <p>Characters</p>
            </div>
            """.format(len(text)), unsafe_allow_html=True)
            
            st.markdown("""
            <div class="metric-card">
                <h3>{}</h3>
                <p>Words</p>
            </div>
            """.format(len(text.split())), unsafe_allow_html=True)
            
            st.markdown("""
            <div class="metric-card">
                <h3>{}</h3>
                <p>Paragraphs</p>
            </div>
            """.format(len([p for p in text.split('\n\n') if p.strip()])), unsafe_allow_html=True)
        
        # Load sample contract
        if st.button("📋 Load Sample Contract"):
            sample_text = load_sample_contract()
            st.session_state.contract_text = sample_text
            st.success("Sample contract loaded!")
            st.rerun()

def handle_question_answering(model_choice, confidence_threshold):
    st.header("❓ Contract Question & Answer")
    
    if not st.session_state.contract_text:
        st.warning("⚠️ Please upload or paste a contract first!")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Question input
        question = st.text_input(
            "Ask a question about the contract:",
            placeholder="e.g., What is the termination notice period?"
        )
        
        # Suggested questions
        st.subheader("💡 Suggested Questions")
        suggested_questions = [
            "What is the payment deadline?",
            "How much notice is required for termination?",
            "What are the liability limitations?", 
            "Who owns the intellectual property?",
            "What is the contract duration?",
            "What are the confidentiality requirements?"
        ]
        
        cols = st.columns(2)
        for i, suggestion in enumerate(suggested_questions):
            col = cols[i % 2]
            if col.button(f"💬 {suggestion}", key=f"suggest_{i}"):
                question = suggestion
        
        # Answer question
        if st.button("🔍 Get Answer", type="primary") and question:
            with st.spinner("🤖 Analyzing contract..."):
                try:
                    qa_pipeline = load_qa_model(model_choice)
                    result = qa_pipeline({
                        'question': question,
                        'context': st.session_state.contract_text
                    })
                    
                    if result['score'] >= confidence_threshold:
                        st.success(f"**Answer:** {result['answer']}")
                        st.info(f"**Confidence:** {result['score']:.2%}")
                        
                        # Add to history
                        st.session_state.qa_history.append({
                            'question': question,
                            'answer': result['answer'],
                            'confidence': result['score'],
                            'timestamp': datetime.now()
                        })
                        
                        # Show answer in context
                        show_answer_in_context(st.session_state.contract_text, result['answer'])
                        
                    else:
                        st.warning(f"⚠️ Low confidence answer ({result['score']:.2%}). The answer might not be reliable.")
                        st.write(f"**Possible Answer:** {result['answer']}")
                
                except Exception as e:
                    st.error(f"Error processing question: {str(e)}")
    
    with col2:
        # QA History
        if st.session_state.qa_history:
            st.subheader("📜 Q&A History")
            
            for i, qa in enumerate(reversed(st.session_state.qa_history[-5:])):  # Show last 5
                with st.expander(f"Q: {qa['question'][:50]}..."):
                    st.write(f"**Answer:** {qa['answer']}")
                    st.write(f"**Confidence:** {qa['confidence']:.2%}")
                    st.write(f"**Time:** {qa['timestamp'].strftime('%H:%M:%S')}")

def handle_clause_analysis(model_choice, enable_detailed, confidence_threshold):
    st.header("🔍 Contract Clause Analysis")
    
    if not st.session_state.contract_text:
        st.warning("⚠️ Please upload or paste a contract first!")
        return
    
    if st.button("🚀 Analyze Contract Clauses", type="primary"):
        with st.spinner("🔬 Analyzing contract clauses..."):
            try:
                # Split contract into clauses
                clauses = split_into_clauses(st.session_state.contract_text)
                
                # Load classification model
                classifier = load_classification_model(model_choice)
                
                # Analyze each clause
                results = []
                progress_bar = st.progress(0)
                
                for i, clause in enumerate(clauses):
                    if len(clause.strip()) > 20:  # Skip very short clauses
                        result = classifier(clause)
                        
                        if isinstance(result, list):
                            result = result[0]
                        
                        results.append({
                            'clause_id': i + 1,
                            'text': clause,
                            'category': result['label'],
                            'confidence': result['score'],
                            'length': len(clause)
                        })
                    
                    progress_bar.progress((i + 1) / len(clauses))
                
                # Store results
                st.session_state.analysis_results = results
                
                # Display results
                display_clause_analysis_results(results, confidence_threshold)
                
            except Exception as e:
                st.error(f"Error analyzing clauses: {str(e)}")

def handle_results_dashboard(enable_key_extraction):
    st.header("📊 Analysis Dashboard")
    
    if not st.session_state.analysis_results and not st.session_state.qa_history:
        st.info("🔍 Run clause analysis or ask questions to see results here!")
        return
    
    # QA Summary
    if st.session_state.qa_history:
        st.subheader("❓ Question & Answer Summary")
        
        qa_df = pd.DataFrame(st.session_state.qa_history)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Confidence distribution
            fig = px.histogram(
                qa_df, x='confidence',
                title="Answer Confidence Distribution",
                nbins=10
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Questions over time
            qa_df['hour'] = qa_df['timestamp'].dt.hour
            questions_by_hour = qa_df.groupby('hour').size().reset_index()
            questions_by_hour.columns = ['hour', 'count']
            
            fig = px.bar(questions_by_hour, x='hour', y='count', 
                        title="Questions by Hour")
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed QA table
        st.dataframe(qa_df[['question', 'answer', 'confidence']], use_container_width=True)
    
    # Clause Analysis Summary
    if st.session_state.analysis_results:
        st.subheader("🔍 Clause Analysis Summary")
        
        results_df = pd.DataFrame(st.session_state.analysis_results)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Category distribution
            category_counts = results_df['category'].value_counts()
            
            fig = px.pie(
                values=category_counts.values,
                names=category_counts.index,
                title="Clause Categories Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Confidence by category
            fig = px.box(
                results_df, x='category', y='confidence',
                title="Confidence by Category"
            )
            fig.update_xaxis(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        
        # Key extraction
        if enable_key_extraction:
            st.subheader("🔑 Key Information Extracted")
            key_info = extract_key_info(st.session_state.contract_text)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**💰 Financial Terms**")
                for item in key_info.get('financial', []):
                    st.write(f"• {item}")
            
            with col2:
                st.markdown("**📅 Dates & Deadlines**")
                for item in key_info.get('dates', []):
                    st.write(f"• {item}")
            
            with col3:
                st.markdown("**👥 Parties & Entities**")
                for item in key_info.get('parties', []):
                    st.write(f"• {item}")
    
    # Export options
    st.subheader("📥 Export Results")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.session_state.qa_history:
            qa_csv = pd.DataFrame(st.session_state.qa_history).to_csv(index=False)
            st.download_button(
                "Download Q&A Results (CSV)",
                qa_csv,
                "contract_qa_results.csv",
                "text/csv"
            )
    
    with col2:
        if st.session_state.analysis_results:
            analysis_csv = pd.DataFrame(st.session_state.analysis_results).to_csv(index=False)
            st.download_button(
                "Download Clause Analysis (CSV)",
                analysis_csv,
                "contract_clause_analysis.csv",
                "text/csv"
            )
    
    with col3:
        if st.session_state.contract_text:
            st.download_button(
                "Download Original Contract (TXT)",
                st.session_state.contract_text,
                "contract_original.txt",
                "text/plain"
            )

# Utility Functions
def show_answer_in_context(text, answer):
    """Highlight answer in contract text"""
    if answer in text:
        highlighted = text.replace(answer, f"**{answer}**")
        st.markdown("**Answer in context:**")
        st.markdown(highlighted[:500] + "..." if len(highlighted) > 500 else highlighted)

def display_clause_analysis_results(results, confidence_threshold):
    """Display clause analysis results with filtering and visualization"""
    
    # Filter by confidence
    filtered_results = [r for r in results if r['confidence'] >= confidence_threshold]
    
    st.success(f"✅ Analyzed {len(results)} clauses ({len(filtered_results)} above confidence threshold)")
    
    # Category summary
    categories = {}
    for result in filtered_results:
        cat = result['category']
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(result)
    
    # Display by category
    for category, clauses in categories.items():
        with st.expander(f"📋 {category} ({len(clauses)} clauses)"):
            for clause in clauses:
                st.markdown(f"""
                <div class="clause-highlight">
                    <strong>Clause {clause['clause_id']}:</strong> {clause['text'][:200]}...
                    <br><small>Confidence: {clause['confidence']:.2%}</small>
                </div>
                """, unsafe_allow_html=True)

def extract_text_from_pdf(uploaded_file):
    """Extract text from PDF file"""
    try:
        from PyPDF2 import PdfReader
        pdf = PdfReader(uploaded_file)
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
        return text
    except ImportError:
        st.error("PyPDF2 not installed. Please install it to process PDF files.")
        return ""

def extract_text_from_docx(uploaded_file):
    """Extract text from DOCX file"""
    try:
        from docx import Document
        doc = Document(uploaded_file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except ImportError:
        st.error("python-docx not installed. Please install it to process DOCX files.")
        return ""

def load_sample_contract():
    """Load a sample contract for testing"""
    return """
    SOFTWARE LICENSE AGREEMENT

    This Software License Agreement ("Agreement") is entered into on January 1, 2024, between TechCorp Inc. ("Company") and Client Corp ("Client").

    1. PAYMENT TERMS
    Client agrees to pay the license fee of $50,000 within thirty (30) days of invoice date. Late payments will incur a penalty of 1.5% per month on the outstanding amount.

    2. TERMINATION
    Either party may terminate this Agreement upon sixty (60) days written notice. Upon termination, Client must cease all use of the software and return all materials.

    3. LIABILITY
    Company's total liability under this Agreement shall not exceed the total amount paid by Client. In no event shall Company be liable for indirect or consequential damages.

    4. INTELLECTUAL PROPERTY
    All intellectual property rights in the software remain with Company. Client receives only a license to use, not ownership.

    5. CONFIDENTIALITY
    Both parties agree to maintain confidentiality of proprietary information for a period of five (5) years after termination.
    """

if __name__ == "__main__":
    main()