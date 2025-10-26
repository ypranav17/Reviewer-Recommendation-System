***
# Research Paper Reviewer Recommendation System

This project is an **AI-powered Reviewer Recommendation System** that helps identify the most relevant reviewers for a research paper based on content similarity. It uses **TF-IDF** and **Sentence‑BERT (SBERT)** models to compare an uploaded research paper with a dataset of previously published papers from different authors.

Built with **Python**, **Streamlit**, and **Sentence Transformers**, it provides a clean and interactive web interface where you can upload a PDF paper and instantly get reviewer suggestions with similarity scores and insightful visualizations.

***

## Features

- **Upload PDFs**: Upload any research paper in PDF format.
- **Automatic Text Extraction**: Extracts and cleans text using `pdfplumber`.
- **Semantic Similarity Matching**:
  - **TF‑IDF + Cosine Similarity** (Fast and efficient)
  - **Sentence‑BERT (SBERT)** (Deep contextual AI model)
  - **Jaccard Similarity** (Simple word overlap for comparison)
- **Top Reviewer Recommendations**: See the top‑N recommended reviewers.
- **Interactive Visualizations**:
  - Reviewers’ similarity bar chart
  - Radar chart comparison for top reviewers
- **Export Results**: Download reviewer recommendations as a CSV file.

***

## How It Works

1. **Dataset Setup**
   - Each author has a folder containing their published research papers.
   - The system preprocesses all PDFs to create **author profiles** that capture their research themes.

2. **Profile Building**
   - Each author’s papers are processed using `pdfplumber` to extract text.
   - The extracted text is cleaned, tokenized, and combined into a single author text profile.
   - These profiles are saved as a serialized pickle file (`author_profiles.pkl`) for faster reuse.

3. **Recommendation Process**
   - A user uploads a new paper.
   - The text from the uploaded paper is extracted and compared with all author profiles.
   - The top‑K most similar authors are displayed with similarity scores.

***

## Getting Started

### **1. Clone the Repository**
```bash
git clone https://github.com/M-Yaswanth-Reddy/Reviewer-Recommendation-System.git
cd Reviewer-Recommendation-System
```

### **2. Create and Activate a Virtual Environment**
**Windows (Anaconda Prompt):**
```bash
conda create -n reviewer-env python=3.10 -y
conda activate reviewer-env
```


***

### **3. Install Dependencies**
```bash
pip install -r requirements.txt
```

This will install all required libraries including:
- Streamlit
- Sentence‑Transformers
- Scikit‑Learn
- pdfplumber
- Plotly
- PyTorch  
and more.

***

### **4. Run the App**
```bash
streamlit run app.py
```
Once started, Streamlit will launch in your browser at:
```
http://localhost:8501
```

***

## Project Structure

```
Reviewer-Recommendation-System/
│
├── app.py                # Streamlit web app (main entry point)
├── utils.py              # Helper functions for text extraction & similarity
├── requirements.txt      # Required packages
├── data/
│   ├── author_profiles.pkl   # Preprocessed author profile data
│   └── metadata.json         # Metadata about authors and papers
```

***

## Configuration

I have built the model in Google Colab:
1. Save the `author_profiles.pkl` and `metadata.json` files to the `data/` folder.
2. Edit the path in `app.py` (default: `data/author_profiles.pkl`).
3. Restart the Streamlit app.

***

## Example Usage

1. Launch the app.
2. Upload your paper (PDF).
3. Choose the similarity method (TF‑IDF, SBERT, or Jaccard).
4. Get top recommended reviewers visually ranked by similarity.
5. Export the results as a CSV file.

***

## Technologies Used

| Category | Technologies |
|-----------|---------------|
| **Frontend** | Streamlit 1.28.0, Plotly 5.17 |
| **Backend** | Python 3.10, Sentence‑Transformers, Scikit‑Learn |
| **Data Processing** | pdfplumber, pandas, numpy |
| **Embedding Models** | SBERT (`all-MiniLM-L6-v2`) |
| **Visualization** | Plotly interactive charts |
| **Storage** | Pickle (`author_profiles.pkl` for fast loading) |

***

## Deployment Options

You can deploy this project for free using any of the following platforms:

### **1. Streamlit Cloud **
1. Push your code to a public GitHub repository.
2. Go to [https://share.streamlit.io](https://share.streamlit.io).
3. Log in with GitHub → Click “New App”.
4. Choose your repo and `app.py`.
5. Deploy — your web app will be live instantly.


***

## Author
**Name:** M Yaswanth Reddy  
**Project:** Reviewer Recommendation System  
**Tech Stack:** Streamlit | Python | Sentence‑BERT | Scikit‑Learn

***

## License
This project is released under the MIT License.  
You are free to modify, distribute, and use it for academic or research purposes.
