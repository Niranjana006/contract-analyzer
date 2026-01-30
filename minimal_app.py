"""
🚀 MINIMAL Contract Analyzer - No Caching Issues
===============================================
Ultra-simplified version that works immediately
"""

import streamlit as st
import pandas as pd
import re
from datetime import datetime

# Simple global variables to avoid caching issues
QA_MODEL = None
CLASSIFICATION_MODEL = None

# Page config
st.set_page_config(
    page_title="Contract Analyzer",
    page_icon="📄",
    layout="wide"
)

# CSS
st.markdown("""
<style>
    .header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        border: 1px solid #dee2e6;
    }
    .clause {
        padding: 0.5rem;
        margin: 0.5rem 0;
        border-radius: 5px;
        border-left: 3px solid #667eea;
        background: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'contract_text' not in st.session_state:
    st.session_state.contract_text = ""
if 'qa_history' not in st.session_state:
    st.session_state.qa_history = []

def load_models():
    """Load models without caching"""
    global QA_MODEL, CLASSIFICATION_MODEL
    
    try:
        if QA_MODEL is None:
            from transformers import pipeline
            QA_MODEL = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
            st.success("✅ QA Model loaded!")
        return True
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        st.info("Install with: pip install transformers torch")
        return False

def get_sample_contract():
    return """SOFTWARE LICENSE AGREEMENT

This Agreement is entered into on January 1, 2024, between TechCorp Inc. and Client Corp.

1. PAYMENT TERMS
Client pays $50,000 within thirty (30) days of invoice. Late payments incur 1.5% monthly penalty.

2. TERMINATION  
Either party may terminate with sixty (60) days written notice. Client must cease software use upon termination.

3. LIABILITY
Company liability shall not exceed total amount paid by Client. No liability for indirect damages.

4. INTELLECTUAL PROPERTY
All IP rights remain with Company. Client receives usage license only.

5. CONFIDENTIALITY
Both parties maintain confidentiality for five (5) years after termination."""

def extract_key_info(text):
    """Simple regex-based extraction"""
    info = {'financial': [], 'dates': [], 'deadlines': []}
    
    # Money amounts
    money = re.findall(r'\$[\d,]+', text)
    info['financial'].extend(money[:3])
    
    # Day periods  
    days = re.findall(r'(\d+)\s+days?', text, re.IGNORECASE)
    info['deadlines'].extend([f"{d} days" for d in days[:3]])
    
    deadline_patterns = [
        r'\b(\d{1,3})\s*days?\b',                       # “30 days”, “60 day”
        r'\bwithin\s+(\d{1,3})\s*days?\b',              # “within 15 days”
        r'\b(\d{1,3})\s*calendar\s+days?\b',            # “45 calendar days”
        r'\b(?:thirty|forty|forty[- ]five|sixty|ninety)\s*'
        r'(?:\(\d{1,3}\))?\s*days?\b',                  # “sixty (60) days”, “forty-five days”
    ]

    for pattern in deadline_patterns:
        for match in re.findall(pattern, text, flags=re.IGNORECASE):
            # If match is a tuple (from the spelled-out variant) take first element
            value = match if isinstance(match, str) else match[0]
            info['deadlines'].append(f"{value.strip()} days")

    # deduplicate & limit
    info['deadlines'] = list(dict.fromkeys(info['deadlines']))[:5]
    
    # Dates
    dates = re.findall(r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}', text)
    info['dates'].extend(dates[:3])
    
    return info

def categorize_clause(text):
    """Simple keyword-based categorization"""
    text_lower = text.lower()
    
    categories = {
        'Payment': ['payment', 'pay', 'fee', 'invoice', 'billing', 'cost'],
        'Termination': ['terminate', 'termination', 'end', 'expire', 'cancel'],
        'Liability': ['liability', 'liable', 'damages', 'responsible', 'harm'],
        'IP': ['intellectual property', 'copyright', 'patent', 'proprietary', 'ip'],
        'Confidentiality': ['confidential', 'non-disclosure', 'proprietary', 'secret']
    }
    
    max_matches = 0
    best_category = 'General'
    
    for category, keywords in categories.items():
        matches = sum(1 for keyword in keywords if keyword in text_lower)
        if matches > max_matches:
            max_matches = matches
            best_category = category
    
    confidence = min(0.9, max_matches * 0.15 + 0.4)
    return best_category, confidence

def main():
    # Header
    st.markdown("""
    <div class="header">
        <h1>📄 Contract Analyzer</h1>
        <p>Simple AI-powered contract analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.title("🔧 Controls")
        if st.button("🗑️ Clear All"):
            st.session_state.contract_text = ""
            st.session_state.qa_history = []
            st.rerun()
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["📝 Contract", "❓ Questions", "📊 Results"])
    
    with tab1:
        st.header("📄 Contract Input")
        
        # File upload
        uploaded_file = st.file_uploader("Upload Text File", type=['txt'])
        
        if uploaded_file:
            try:
                content = str(uploaded_file.read(), "utf-8")
                st.session_state.contract_text = content.strip()
                st.success("✅ File uploaded!")
            except Exception as e:
                st.error(f"Error: {e}")
        
        # Text input
        text_input = st.text_area(
            "Or paste contract text:",
            value=st.session_state.contract_text,
            height=300
        )
        
        if text_input != st.session_state.contract_text:
            st.session_state.contract_text = text_input
        
        # Sample contract
        if st.button("📋 Load Sample"):
            st.session_state.contract_text = get_sample_contract()
            st.success("Sample loaded!")
            st.rerun()
        
        
    with tab2:
        st.header("❓ Ask Questions")
        
        if not st.session_state.contract_text:
            st.warning("⚠️ Please add contract text first!")
        else:
            # Question input
            question = st.text_input("Ask about the contract:")
            
            # Suggested questions
            suggestions = [
                "What is the payment amount?",
                "How much notice is required for termination?",
                "What are the liability limits?",
                "Who owns the intellectual property?"
            ]
            
            st.write("💡 **Try these questions:**")
            for suggestion in suggestions:
                if st.button(f"💬 {suggestion}"):
                    question = suggestion
            
            # Process question
            if st.button("🔍 Get Answer") and question:
                if load_models():
                    try:
                        with st.spinner("Finding answer..."):
                            global QA_MODEL
                            result = QA_MODEL(
                                question=question, 
                                context=st.session_state.contract_text
                            )
                            
                            st.success(f"**Answer:** {result['answer']}")
                            st.info(f"**Confidence:** {result['score']:.1%}")
                            
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
                    st.error("Models not available")
            
            # Show history
            if st.session_state.qa_history:
                st.subheader("📜 Recent Q&A")
                for qa in reversed(st.session_state.qa_history[-3:]):
                    with st.expander(f"Q: {qa['question'][:40]}..."):
                        st.write(f"**A:** {qa['answer']}")
                        st.write(f"**Confidence:** {qa['confidence']:.1%}")
    
    with tab3:
        st.header("📊 Analysis Results")
        
        if not st.session_state.contract_text:
            st.info("Add contract text to see analysis")
        else:
            if st.button("🔍 Analyze Contract"):
                with st.spinner("Analyzing..."):
                    
                    # Split into clauses (simple paragraph split)
                    paragraphs = [p.strip() for p in st.session_state.contract_text.split('\n\n') if len(p.strip()) > 30]
                    
                    # Categorize each clause
                    results = []
                    for i, paragraph in enumerate(paragraphs):
                        category, confidence = categorize_clause(paragraph)
                        results.append({
                            'id': i+1,
                            'text': paragraph,
                            'category': category,
                            'confidence': confidence
                        })
                    
                    st.success(f"✅ Analyzed {len(results)} clauses")
                    
                    # Show results by category
                    categories = {}
                    for result in results:
                        cat = result['category']
                        if cat not in categories:
                            categories[cat] = []
                        categories[cat].append(result)
                    # FIXED VERSION - Display actual clause content
                    for category, clauses in categories.items():
                        with st.expander(f"📋 {category} ({len(clauses)} clauses)"):
                            for clause in clauses:
            # Display the full clause text properly
                                clause_preview = clause['text'][:200] + "..." if len(clause['text']) > 200 else clause['text']
            
                                st.markdown(f"""
                                **Clause {clause['id']}:**
            
                                {clause_preview}
            
                                *Confidence: {clause['confidence']:.1%}*
                                """)
                                st.divider()  # Add separator between clauses

                    
                    
                    
                    # Key information
                    st.subheader("🔑 Key Information")
                    key_info = extract_key_info(st.session_state.contract_text)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown("**💰 Financial**")
                        for item in key_info['financial']:
                            st.write(f"• {item}")
                    
                    with col2:
                        st.markdown("**📅 Dates**")
                        for item in key_info['dates']:
                            st.write(f"• {item}")
                    
                    with col3:
                        st.markdown("**⏰ Deadlines**")
                        for item in key_info['deadlines']:
                            st.write(f"• {item}")
            
            # Q&A Summary
            if st.session_state.qa_history:
                st.subheader("❓ Q&A Summary")
                df = pd.DataFrame(st.session_state.qa_history)
                st.dataframe(df[['question', 'answer', 'confidence']])
                
                # Download
                if st.button("📥 Download Q&A Results"):
                    csv = df.to_csv(index=False)
                    st.download_button(
                        "Download CSV",
                        csv,
                        "qa_results.csv",
                        "text/csv"
                    )

if __name__ == "__main__":
    main()