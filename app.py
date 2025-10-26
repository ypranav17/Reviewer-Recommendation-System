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

st.set_page_config(
    page_title="Reviewer Recommendation System",
    layout="wide"
)

# Custom styling
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
    """Load model and create similarity calculator"""
    try:
        author_profiles = load_model('data/author_profiles.pkl')
        if author_profiles:
            calculator = SimilarityCalculator(author_profiles)
            with open('data/metadata.json', 'r') as f:
                metadata = json.load(f)
            return calculator, metadata
        return None, None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None


# Header
st.title("Research Paper Reviewer Recommendation System")
st.markdown("Find academically suitable reviewers using AI-based semantic similarity methods.")


# Load pre-trained profiles
with st.spinner("Loading the pre-trained model..."):
    calculator, metadata = load_system()

if calculator is None:
    st.error("Model loading failed. Ensure that data/author_profiles.pkl is available.")
    st.stop()

st.session_state.calculator = calculator


# Sidebar configuration
with st.sidebar:
    st.header("System Information")
    
    if metadata:
        st.metric("Total Authors", metadata['num_authors'])
        st.caption(f"Model created on: {metadata['created_at'][:10]}")

    st.markdown("---")
    st.subheader("Analysis Settings")
    
    method = st.selectbox(
        "Select Similarity Method",
        ["sbert", "tfidf", "jaccard"],
        help="Choose a method: \n- SBERT: Most accurate using deep semantic embeddings.\n- TF-IDF: Fast, keyword-based approach.\n- Jaccard: Basic word overlap comparison."
    )
    
    top_k = st.slider("Number of Reviewers", 3, 20, 10)
    compare_all = st.checkbox("Compare All Methods")
    
    st.markdown("---")
    st.success("System is ready to run.")


# File uploader
st.subheader("Upload Research Paper")

uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=['pdf'],
    help="Upload a research paper in PDF format to receive reviewer recommendations."
)


if uploaded_file is not None:
    # Temporary save
    with open("temp_paper.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"Uploaded file: {uploaded_file.name}")
    
    col1, _ = st.columns([3, 1])
    analyze_btn = col1.button("Run Reviewer Analysis")

    if analyze_btn:
        with st.spinner(f"Analyzing the uploaded paper using {method.upper()} method..."):
            extractor = PDFExtractor()
            input_text = extractor.extract_text("temp_paper.pdf")

            if not input_text:
                st.error("Unable to extract readable text from the provided PDF file.")
            else:
                st.success(f"Text extraction complete. Extracted {len(input_text):,} characters.")
                
                if not compare_all:
                    if method == 'sbert':
                        results = st.session_state.calculator.sbert_similarity(input_text, top_k)
                    elif method == 'tfidf':
                        results = st.session_state.calculator.tfidf_similarity(input_text, top_k)
                    else:
                        results = st.session_state.calculator.jaccard_similarity(input_text, top_k)
                    st.session_state.results = {method: results}
                else:
                    st.session_state.results = {}
                    st.session_state.results['jaccard'] = st.session_state.calculator.jaccard_similarity(input_text, top_k)
                    st.session_state.results['tfidf'] = st.session_state.calculator.tfidf_similarity(input_text, top_k)
                    st.session_state.results['sbert'] = st.session_state.calculator.sbert_similarity(input_text, top_k)
                
                st.success("Analysis completed successfully.")

    if os.path.exists("temp_paper.pdf"):
        os.remove("temp_paper.pdf")


# Display results
if st.session_state.results:
    st.markdown("---")
    st.header("Analysis Results")

    results = st.session_state.results

    if len(results) == 1:
        method_name = list(results.keys())[0]
        method_results = results[method_name]

        tab1, tab2, tab3 = st.tabs(["Table", "Charts", "Export"])

        with tab1:
            st.subheader(f"Top {len(method_results)} Recommended Reviewers ({method_name.upper()})")

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
                    title='Reviewer Similarity Scores'
                )
                fig.update_layout(xaxis_tickangle=-45, showlegend=False, height=500)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                top5 = method_results[:5]
                fig = go.Figure()
                for r in top5:
                    fig.add_trace(go.Scatterpolar(
                        r=[r['score'], r['papers'] / 20, r['score'] * 100],
                        theta=['Similarity', 'Publications (scaled)', 'Relevance (%)'],
                        fill='toself',
                        name=r['author'][:20]
                    ))
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    showlegend=True,
                    title="Top 5 Reviewer Comparison",
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)

        with tab3:
            csv = df.to_csv(index=True)
            st.download_button(
                "Download as CSV",
                csv,
                f"reviewer_results_{method_name}.csv",
                "text/csv"
            )

    else:
        st.subheader("Method Comparison")

        tab1, tab2, tab3 = st.tabs(["Side-by-Side Comparison", "Common Reviewers", "Export Results"])

        with tab1:
            cols = st.columns(3)
            for idx, (method_name, method_results) in enumerate(results.items()):
                with cols[idx]:
                    st.markdown(f"### {method_name.upper()}")
                    for i, r in enumerate(method_results[:5], 1):
                        st.markdown(f"{i}. {r['author']} — Score: {r['score']:.4f}")

        with tab2:
            st.subheader("Reviewers appearing in multiple methods")

            top5_sets = {m: set([r['author'] for r in res[:5]]) for m, res in results.items()}
            all_common = set.intersection(*top5_sets.values())

            if all_common:
                st.success(f"{len(all_common)} common reviewers found across all methods:")
                for author in all_common:
                    st.markdown(f"- {author}")
            else:
                st.info("No common reviewers found across methods.")

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
                "Export Comparison Data",
                csv,
                "comparison_results.csv",
                "text/csv"
            )


# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Reviewer Recommendation System | Based on Semantic Similarity Analysis</p>
    <p>Developed with Streamlit and Python</p>
</div>
""", unsafe_allow_html=True)
