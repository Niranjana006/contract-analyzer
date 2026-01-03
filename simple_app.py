"""
🚀 SIMPLIFIED Contract Analyzer - No External Dependencies
=========================================================
A working contract analysis tool that runs immediately without extra files
"""

import streamlit as st
import pandas as pd
import re
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Only import transformers when needed to avoid startup issues
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    st.error("Transformers not installed. Run: pip install transformers torch")

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
        text-align: center;
    }
    
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        border: 1px solid #dee2e6;
    }
    
    .clause-highlight {
        padding: 0.5rem;
        margin: 0.5rem 0;
        border-radius: 5px;
        border-left: 3px solid #667eea;
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'contract_text' not in st.session_state:
    st.session_state.contract_text = ""
if 'qa_history' not in st.session_state:
    st.session_state.qa_history = []
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

# Cached model loading
@st.cache_resource
def load_qa_model():
    """Load QA model with error handling"""
    try:
        if TRANSFORMERS_AVAILABLE:
            return pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
        else:
            return None
    except Exception as e:
        st.error(f"Error loading QA model: {e}")
        return None

@st.cache_resource  
def load_classification_model():
    """Load classification model with error handling"""
    try:
        if TRANSFORMERS_AVAILABLE:
            return pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
        else:
            return None
    except Exception as e:
        st.error(f"Error loading classification model: {e}")
        return None

def clean_contract_text(text):
    """Clean contract text"""
    if not text:
        return ""
    
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    return text.strip()

def split_into_clauses(text):
    """Split contract into clauses"""
    if not text:
        return []
    
    # Split by numbered sections or paragraphs
    clauses = []
    
    # Try numbered sections first
    numbered_sections = re.split(r'\n\d+\.', text)
    if len(numbered_sections) > 1:
        clauses = [clause.strip() for clause in numbered_sections[1:] if clause.strip()]
    else:
        # Fall back to paragraph splits
        paragraphs = text.split('\n\n')
        clauses = [p.strip() for p in paragraphs if len(p.strip()) > 50]
    
    return clauses if clauses else [text]

def extract_key_info(text):
    """Extract key information using regex"""
    key_info = {
        'financial': [],
        'dates': [],
        'parties': [],
        'deadlines': []
    }
    
    if not text:
        return key_info
    
    # Financial terms
    money_patterns = [
        r'\$[\d,]+(?:\.\d{2})?',
        r'[\d,]+\s*(?:dollars?|USD)',
    ]
    
    for pattern in money_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        key_info['financial'].extend(matches[:5])
    
    # Dates
    date_patterns = [
        r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
        r'\b\d{1,2}\s+days?\b',
        r'\b\d{1,2}\s+(?:months?|years?)\b'
    ]
    
    for pattern in date_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        key_info['dates'].extend(matches[:5])
    
    # Deadlines
    deadline_patterns = [
        r'(\d+)\s+days?\s+(?:notice|prior)',
        r'within\s+(\d+)\s+days?',
    ]
    
    for pattern in deadline_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        key_info['deadlines'].extend([f"{match} days" for match in matches[:5]])
    
    return key_info

def get_sample_contract():
    """Return sample contract text"""
    return """
    SOFTWARE LICENSE AGREEMENT

    This Software License Agreement ("Agreement") is entered into on January 1, 2024, between TechCorp Inc. ("Company") and Client Corp ("Client").

    1. PAYMENT TERMS
    Client agrees to pay the license fee of $50,000 within thirty (30) days of invoice date. Late payments will incur a penalty of 1.5% per month.

    2. TERMINATION
    Either party may terminate this Agreement upon sixty (60) days written notice. Upon termination, Client must cease all use of the software.

    3. LIABILITY
    Company's total liability under this Agreement shall not exceed the total amount paid by Client. Company shall not be liable for indirect damages.

    4. INTELLECTUAL PROPERTY
    All intellectual property rights in the software remain with Company. Client receives only a license to use.

    5. CONFIDENTIALITY
    Both parties agree to maintain confidentiality of proprietary information for five (5) years after termination.
    """

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📄 Contract Analyzer</h1>
        <p>AI-powered contract analysis tool</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.title("🔧 Controls")
        
        if st.button("🗑️ Clear All"):
            st.session_state.contract_text = ""
            st.session_state.qa_history = []
            st.session_state.analysis_results = None
            st.rerun()
        
        st.subheader("📊 App Status")
        if TRANSFORMERS_AVAILABLE:
            st.success("✅ AI Models Available")
        else:
            st.error("❌ AI Models Not Available")
            st.info("Install with: pip install transformers torch")

    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📝 Contract", "❓ Questions", "🔍 Analysis", "📊 Results"])
    
    with tab1:
        handle_contract_input()
    
    with tab2:
        handle_questions()
    
    with tab3:
        handle_analysis()
    
    with tab4:
        handle_results()

def handle_contract_input():
    st.header("📄 Contract Input")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # File upload
        uploaded_file = st.file_uploader(
            "Upload Contract File",
            type=['txt'],
            help="Upload a text file"
        )
        
        if uploaded_file is not None:
            try:
                contract_text = str(uploaded_file.read(), "utf-8")
                st.session_state.contract_text = clean_contract_text(contract_text)
                st.success("✅ Contract uploaded successfully!")
            except Exception as e:
                st.error(f"Error reading file: {e}")
        
        # Text area
        manual_text = st.text_area(
            "Or paste contract text here:",
            value=st.session_state.contract_text,
            height=300,
            placeholder="Paste your contract text here..."
        )
        
        if manual_text != st.session_state.contract_text:
            st.session_state.contract_text = clean_contract_text(manual_text)
        
        # Load sample
        if st.button("📋 Load Sample Contract"):
            st.session_state.contract_text = get_sample_contract()
            st.success("Sample contract loaded!")
            st.rerun()
    
    with col2:
        st.subheader("📊 Document Stats")
        
        if st.session_state.contract_text:
            text = st.session_state.contract_text
            
            st.markdown(f"""
            <div class="metric-card">
                <h3>{len(text)}</h3>
                <p>Characters</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="metric-card">
                <h3>{len(text.split())}</h3>
                <p>Words</p>
            </div>
            """, unsafe_allow_html=True)
            
            paragraphs = len([p for p in text.split('\n\n') if p.strip()])
            st.markdown(f"""
            <div class="metric-card">
                <h3>{paragraphs}</h3>
                <p>Paragraphs</p>
            </div>
            """, unsafe_allow_html=True)

def handle_questions():
    st.header("❓ Ask Questions")
    
    if not st.session_state.contract_text:
        st.warning("⚠️ Please add contract text first!")
        return
    
    if not TRANSFORMERS_AVAILABLE:
        st.error("❌ AI models not available. Please install transformers.")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        question = st.text_input(
            "Ask about the contract:",
            placeholder="e.g., What is the payment deadline?"
        )
        
        # Suggested questions
        st.subheader("💡 Try These Questions")
        suggestions = [
            "What is the payment deadline?",
            "How much notice is required for termination?",
            "What are the liability limitations?",
            "Who owns the intellectual property?",
            "What is the contract duration?"
        ]
        
        for i, suggestion in enumerate(suggestions):
            if st.button(f"💬 {suggestion}", key=f"suggest_{i}"):
                question = suggestion
        
        if st.button("🔍 Get Answer") and question:
            with st.spinner("🤖 Finding answer..."):
                qa_model = load_qa_model()
                
                if qa_model:
                    try:
                        result = qa_model({
                            'question': question,
                            'context': st.session_state.contract_text
                        })
                        
                        st.success(f"**Answer:** {result['answer']}")
                        st.info(f"**Confidence:** {result['score']:.2%}")
                        
                        # Add to history
                        st.session_state.qa_history.append({
                            'question': question,
                            'answer': result['answer'],
                            'confidence': result['score'],
                            'timestamp': datetime.now()
                        })
                        
                    except Exception as e:
                        st.error(f"Error: {e}")
                else:
                    st.error("QA model not loaded")
    
    with col2:
        if st.session_state.qa_history:
            st.subheader("📜 Recent Q&A")
            
            for qa in reversed(st.session_state.qa_history[-3:]):
                with st.expander(f"Q: {qa['question'][:30]}..."):
                    st.write(f"**A:** {qa['answer']}")
                    st.write(f"**Confidence:** {qa['confidence']:.2%}")

def handle_analysis():
    st.header("🔍 Contract Analysis")
    
    if not st.session_state.contract_text:
        st.warning("⚠️ Please add contract text first!")
        return
    
    if st.button("🚀 Analyze Contract"):
        with st.spinner("🔬 Analyzing contract..."):
            
            # Split into clauses
            clauses = split_into_clauses(st.session_state.contract_text)
            
            # Simple clause categorization based on keywords
            clause_types = {
                'Payment': ['payment', 'pay', 'fee', 'invoice', 'billing'],
                'Termination': ['termination', 'terminate', 'end', 'expire'],
                'Liability': ['liability', 'liable', 'damages', 'responsible'],
                'IP': ['intellectual property', 'copyright', 'patent', 'proprietary'],
                'Confidentiality': ['confidential', 'non-disclosure', 'proprietary']
            }
            
            results = []
            
            for i, clause in enumerate(clauses):
                clause_lower = clause.lower()
                
                # Determine category
                category = 'General'
                max_matches = 0
                
                for cat, keywords in clause_types.items():
                    matches = sum(1 for keyword in keywords if keyword in clause_lower)
                    if matches > max_matches:
                        max_matches = matches
                        category = cat
                
                results.append({
                    'clause_id': i + 1,
                    'text': clause,
                    'category': category,
                    'confidence': min(0.9, max_matches * 0.2 + 0.3),  # Simulated confidence
                    'length': len(clause)
                })
            
            st.session_state.analysis_results = results
            
            # Display results
            st.success(f"✅ Analyzed {len(results)} clauses")
            
            # Group by category
            categories = {}
            for result in results:
                cat = result['category']
                if cat not in categories:
                    categories[cat] = []
                categories[cat].append(result)
            
            for category, clauses in categories.items():
                with st.expander(f"📋 {category} ({len(clauses)} clauses)"):
                    for clause in clauses:
                        st.markdown(f"""
                        <div class="clause-highlight">
                            <strong>Clause {clause['clause_id']}:</strong> {clause['text'][:200]}...
                            <br><small>Confidence: {clause['confidence']:.1%}</small>
                        </div>
                        """, unsafe_allow_html=True)

def handle_results():
    st.header("📊 Analysis Dashboard")
    
    if not st.session_state.qa_history and not st.session_state.analysis_results:
        st.info("🔍 Run analysis to see results here!")
        return
    
    # Q&A Results
    if st.session_state.qa_history:
        st.subheader("❓ Q&A Summary")
        
        qa_df = pd.DataFrame(st.session_state.qa_history)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(qa_df, x='confidence', title="Answer Confidence Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.dataframe(qa_df[['question', 'answer', 'confidence']], use_container_width=True)
        
        # Export Q&A
        if st.button("📥 Download Q&A Results"):
            csv = qa_df.to_csv(index=False)
            st.download_button(
                "Download CSV",
                csv,
                "qa_results.csv",
                "text/csv"
            )
    
    # Clause Analysis Results
    if st.session_state.analysis_results:
        st.subheader("🔍 Clause Analysis")
        
        results_df = pd.DataFrame(st.session_state.analysis_results)
        
        col1, col2 = st.columns(2)
        
        with col1:
            category_counts = results_df['category'].value_counts()
            fig = px.pie(values=category_counts.values, names=category_counts.index, 
                        title="Clause Categories")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.box(results_df, x='category', y='confidence', 
                        title="Confidence by Category")
            st.plotly_chart(fig, use_container_width=True)
        
        # Key Information
        st.subheader("🔑 Key Information")
        
        if st.session_state.contract_text:
            key_info = extract_key_info(st.session_state.contract_text)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**💰 Financial Terms**")
                for item in key_info['financial']:
                    st.write(f"• {item}")
            
            with col2:
                st.markdown("**📅 Dates & Periods**")
                for item in key_info['dates']:
                    st.write(f"• {item}")
            
            with col3:
                st.markdown("**⏰ Deadlines**")
                for item in key_info['deadlines']:
                    st.write(f"• {item}")
        
        # Export Analysis
        if st.button("📥 Download Analysis Results"):
            csv = results_df.to_csv(index=False)
            st.download_button(
                "Download CSV",
                csv,
                "analysis_results.csv",
                "text/csv"
            )

if __name__ == "__main__":
    main()