# app.py
"""
Research Paper Reviewer Recommendation System
Streamlit Web Interface
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils import PDFExtractor, SimilarityCalculator, load_model
import os
import json

# Page config
st.set_page_config(
    page_title="Reviewer Recommendation System",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {padding: 2rem;}
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'calculator' not in st.session_state:
    st.session_state.calculator = None
if 'results' not in st.session_state:
    st.session_state.results = None

@st.cache_resource
def load_system():
    """Load model and create calculator"""
    try:
        author_profiles = load_model('data/author_profiles.pkl')
        if author_profiles:
            calculator = SimilarityCalculator(author_profiles)
            
            # Load metadata
            with open('data/metadata.json', 'r') as f:
                metadata = json.load(f)
            
            return calculator, metadata
        return None, None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Header
st.title("🔍 Research Paper Reviewer Recommendation System")
st.markdown("**Find the perfect reviewers using AI-powered semantic similarity**")

# Load model
with st.spinner("Loading model..."):
    calculator, metadata = load_system()

if calculator is None:
    st.error("❌ Failed to load model. Please check data/author_profiles.pkl exists.")
    st.info("📋 Make sure to place your downloaded files in the data/ folder")
    st.stop()

st.session_state.calculator = calculator

# Sidebar
with st.sidebar:
    st.header("⚙️ System Information")
    
    if metadata:
        st.metric("Total Authors", metadata['num_authors'])
        st.caption(f"Created: {metadata['created_at'][:10]}")
    
    st.markdown("---")
    
    st.subheader("Analysis Settings")
    
    method = st.selectbox(
        "Similarity Method",
        ["sbert", "tfidf", "jaccard"],
        help="""
        • SBERT: Most accurate (AI-powered)
        • TF-IDF: Fast, keyword-based
        • Jaccard: Simple baseline
        """
    )
    
    top_k = st.slider("Number of Reviewers", 3, 20, 10)
    
    compare_all = st.checkbox("Compare All Methods")
    
    st.markdown("---")
    st.success("🟢 System Ready")

# Main content
st.subheader("📄 Upload Research Paper")

uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=['pdf'],
    help="Upload the research paper to find reviewers for"
)

if uploaded_file is not None:
    # Save temporarily
    with open("temp_paper.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.success(f"✅ Uploaded: {uploaded_file.name}")
    
    # Analysis button
    col1, col2 = st.columns([3, 1])
    
    with col1:
        analyze_btn = st.button("🔍 Find Reviewers", type="primary")
    
    if analyze_btn:
        with st.spinner(f"Analyzing paper using {method.upper()}..."):
            
            # Extract text
            extractor = PDFExtractor()
            input_text = extractor.extract_text("temp_paper.pdf")
            
            if not input_text:
                st.error("❌ Failed to extract text from PDF")
            else:
                st.success(f"✅ Extracted {len(input_text):,} characters")
                
                if not compare_all:
                    # Single method
                    if method == 'sbert':
                        results = st.session_state.calculator.sbert_similarity(input_text, top_k)
                    elif method == 'tfidf':
                        results = st.session_state.calculator.tfidf_similarity(input_text, top_k)
                    else:
                        results = st.session_state.calculator.jaccard_similarity(input_text, top_k)
                    
                    st.session_state.results = {method: results}
                else:
                    # Compare all methods
                    st.session_state.results = {}
                    
                    with st.spinner("Running all methods..."):
                        st.session_state.results['jaccard'] = st.session_state.calculator.jaccard_similarity(input_text, top_k)
                        st.session_state.results['tfidf'] = st.session_state.calculator.tfidf_similarity(input_text, top_k)
                        st.session_state.results['sbert'] = st.session_state.calculator.sbert_similarity(input_text, top_k)
                
                st.success("🎉 Analysis complete!")
    
    # Clean up
    if os.path.exists("temp_paper.pdf"):
        os.remove("temp_paper.pdf")

# Display results
if st.session_state.results:
    st.markdown("---")
    st.header("📊 Results")
    
    results = st.session_state.results
    
    if len(results) == 1:
        # Single method results
        method_name = list(results.keys())[0]
        method_results = results[method_name]
        
        tab1, tab2, tab3 = st.tabs(["📋 Table", "📈 Charts", "💾 Export"])
        
        with tab1:
            st.subheader(f"Top {len(method_results)} Reviewers ({method_name.upper()})")
            
            df = pd.DataFrame(method_results)
            df.index = range(1, len(df) + 1)
            df.columns = ['Author', 'Similarity Score', 'Publications']
            
            styled_df = df.style.background_gradient(
                subset=['Similarity Score'],
                cmap='Greens'
            ).format({'Similarity Score': '{:.4f}'})
            
            st.dataframe(styled_df, use_container_width=True)
        
        with tab2:
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(
                    df,
                    x='Author',
                    y='Similarity Score',
                    color='Similarity Score',
                    color_continuous_scale='Viridis',
                    title='Similarity Scores'
                )
                fig.update_layout(xaxis_tickangle=-45, showlegend=False, height=500)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                top5 = method_results[:5]
                fig = go.Figure()
                
                for r in top5:
                    fig.add_trace(go.Scatterpolar(
                        r=[r['score'], r['papers']/20, r['score']*100],
                        theta=['Similarity', 'Publications (÷20)', 'Relevance %'],
                        fill='toself',
                        name=r['author'][:20]
                    ))
                
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    showlegend=True,
                    title="Top 5 Comparison",
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            csv = df.to_csv(index=True)
            st.download_button(
                "📥 Download CSV",
                csv,
                f"reviewers_{method_name}.csv",
                "text/csv"
            )
    
    else:
        # Multiple methods comparison
        st.subheader("🔬 Method Comparison")
        
        tab1, tab2, tab3 = st.tabs(["📊 Side-by-Side", "🎯 Overlap", "💾 Export"])
        
        with tab1:
            cols = st.columns(3)
            
            for idx, (method_name, method_results) in enumerate(results.items()):
                with cols[idx]:
                    st.markdown(f"### {method_name.upper()}")
                    for i, r in enumerate(method_results[:5], 1):
                        st.markdown(f"**{i}. {r['author']}**")
                        st.caption(f"Score: {r['score']:.4f} | Papers: {r['papers']}")
        
        with tab2:
            st.markdown("### Common Reviewers in Top 5")
            
            top5_sets = {
                m: set([r['author'] for r in res[:5]])
                for m, res in results.items()
            }
            
            all_common = set.intersection(*top5_sets.values())
            
            if all_common:
                st.success(f"✅ {len(all_common)} reviewers in ALL methods:")
                for author in all_common:
                    st.markdown(f"- **{author}** (High confidence)")
            else:
                st.info("No reviewers appear in all methods")
        
        with tab3:
            comparison_data = []
            for m, res in results.items():
                for rank, r in enumerate(res, 1):
                    comparison_data.append({
                        'Method': m.upper(),
                        'Rank': rank,
                        'Author': r['author'],
                        'Score': r['score'],
                        'Papers': r['papers']
                    })
            
            df = pd.DataFrame(comparison_data)
            csv = df.to_csv(index=False)
            st.download_button(
                "📥 Download Comparison",
                csv,
                "method_comparison.csv",
                "text/csv"
            )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Reviewer Recommendation System | Powered by Sentence-BERT & TF-IDF</p>
    <p>Built with Streamlit 🎈</p>
</div>
""", unsafe_allow_html=True)
