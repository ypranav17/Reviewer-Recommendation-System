# utils.py
"""
Utility functions for Reviewer Recommendation System
Matches your Colab implementation exactly
"""

import pickle
import pdfplumber
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


class PDFExtractor:
    """Extract text from PDF files"""
    
    def extract_text(self, pdf_path):
        """Extract all text from a PDF file"""
        try:
            text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + " "
            
            return self.clean_text(text)
            
        except Exception as e:
            print(f"❌ Error extracting {pdf_path}: {str(e)}")
            return ""
    
    def clean_text(self, text):
        """Clean extracted text"""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,;:!?\-]', '', text)
        text = text.lower()
        return text.strip()


class SimilarityCalculator:
    """Calculate similarity between papers and authors"""
    
    def __init__(self, author_profiles):
        self.author_profiles = author_profiles
        self.authors = list(author_profiles.keys())
        print(f"✅ Calculator initialized with {len(self.authors)} authors")
    
    def tfidf_similarity(self, input_text, top_k=10):
        """TF-IDF + Cosine Similarity"""
        print("📊 Calculating TF-IDF similarities...")
        
        corpus = [input_text]
        author_texts = [self.author_profiles[author]['combined_text'] 
                       for author in self.authors]
        corpus.extend(author_texts)
        
        vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )
        
        tfidf_matrix = vectorizer.fit_transform(corpus)
        input_vector = tfidf_matrix[0:1]
        author_vectors = tfidf_matrix[1:]
        
        similarities = cosine_similarity(input_vector, author_vectors)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                'author': self.authors[idx],
                'score': float(similarities[idx]),
                'papers': self.author_profiles[self.authors[idx]]['paper_count']
            })
        
        return results
    
    def sbert_similarity(self, input_text, top_k=10):
        """Sentence-BERT similarity"""
        print("🤖 Calculating SBERT similarities...")
        
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        print("   Converting input paper to AI embedding...")
        input_embedding = model.encode([input_text])
        
        print("   Converting author profiles to AI embeddings...")
        author_embeddings = []
        for author in self.authors:
            text = self.author_profiles[author]['combined_text']
            embedding = model.encode([text])
            author_embeddings.append(embedding[0])
        
        author_embeddings = np.array(author_embeddings)
        
        print("   Calculating similarities...")
        similarities = cosine_similarity(input_embedding, author_embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                'author': self.authors[idx],
                'score': float(similarities[idx]),
                'papers': self.author_profiles[self.authors[idx]]['paper_count']
            })
        
        return results
    
    def jaccard_similarity(self, input_text, top_k=10):
        """Jaccard Similarity"""
        print("📝 Calculating Jaccard similarities...")
        
        input_words = set(input_text.lower().split())
        
        similarities = []
        for author in self.authors:
            author_text = self.author_profiles[author]['combined_text']
            author_words = set(author_text.lower().split())
            
            intersection = len(input_words.intersection(author_words))
            union = len(input_words.union(author_words))
            
            score = intersection / union if union > 0 else 0
            similarities.append(score)
        
        similarities = np.array(similarities)
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                'author': self.authors[idx],
                'score': float(similarities[idx]),
                'papers': self.author_profiles[self.authors[idx]]['paper_count']
            })
        
        return results


def load_model(model_path='data/author_profiles.pkl'):
    """Load saved author profiles"""
    try:
        with open(model_path, 'rb') as f:
            author_profiles = pickle.load(f)
        return author_profiles
    except Exception as e:
        print(f"Error loading model: {e}")
        return None