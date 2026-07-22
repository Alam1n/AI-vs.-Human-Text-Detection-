# AI vs. Human Text Detection Engine
An end-to-end, explainable Machine Learning pipeline engineered to distinguish between Human-Written prose and AI-Generated text across academic, social, and journalistic domains.

The framework extracts and horizontally concatenates n-gram TF-IDF representations, structural linguistic metrics, and 971-dimensional deep contextual embeddings via DistilBERT, followed by model scaling, cross-domain sensitivity analysis (local token perturbation shifts), and high-resolution metric exports.

## Hybrid Tri-Vector Feature Architecture:
- Statistical N-Grams: 200-dimensional TF-IDF sparse matrix tracking structural lexicon frequency.

- Linguistic Stylometrics: Explicit metrics including lexical diversity (TTR), average sentence length, and sentence length variance.

- Deep Contextual Embeddings: 768-dimensional dense vector extractions from Transformer layer activations (distilbert-base-uncased).

- Cross-Domain Generalization: Built-in validation suite testing models across student essays, online articles, and informal social media prose.

- Thesis Asset Generation: Automated export pipeline saving publication-ready, 300 DPI visualization figures (roc_curve_output.png and comparative_metrics_300dpi.png).

- Explainable AI (XAI) Diagnostics: Includes a local token perturbation engine (LIME-style sensitivity shifts) that measures the directional probability shift when specific tokens are omitted.

## Empirical Results
The pipeline achieved an overall System Accuracy of 88.33% on out-of-sample academic test evaluation instances.

Comparative Metric Breakdown


| Class Domain Profile | Precision | Recall | F1-Score | Overall Accuracy |
|----------------------|----------:|-------:|---------:|-----------------:|
| Human-Written (Class 0) | 92.00% | 89.00% | 90.00% | 88.33% |
| AI-Generated (Class 1) | 83.00% | 87.00% | 85.00% | 88.33% | 

Repository Structure
Code snippet
├── thesis_final_assets/                 # Saved model artifacts & figure exports
│   ├── weighted_binary_rf_model.pkl    # Trained Random Forest classifier
│   ├── feature_scaler.pkl              # Fitted StandardScaler weights
│   ├── tfidf_vectorizer.pkl            # Fitted TF-IDF vectorizer dictionary
│   ├── roc_curve_output.png            # High-resolution ROC curve plot
│   └── comparative_metrics_300dpi.png  # Grouped metrics visualization
├── app.py                              # Standalone Flask web application
├── main_pipeline.py                    # Complete training, cross-validation & XAI script
├── requirements.txt                    # Python environment dependencies
└── README.md                           # Documentation
Quickstart Guide
1. Installation
Clone the repository and set up a virtual environment:

Bash
# Clone repository
git clone https://github.com/your-username/hybrid-ai-text-detector.git
cd hybrid-ai-text-detector

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
2. Launching the Flask Web App
Ensure your saved artifacts are inside the thesis_final_assets/ directory, then start the web server:

python app.py
Navigate to [http://127.0.0.1:5000](http://127.0.0.1:5000) in your web browser to test essay snippets live.

3. Running Pipeline Evaluation & Local Token Shift Diagnostics
To train, evaluate, and test token perturbation shifts directly via CLI:

from main_pipeline import test_custom_input

sample_text = """
Furthermore, artificial intelligence technology presents a paradigm shift 
in modern educational frameworks. Consequently, academic institutions must adapt.
"""

test_custom_input(
    raw_text=sample_text,
    label_hint="Sample AI Input",
    model=binary_model,
    extractor=extractor,
    scaler=scaler,
)

- Requirements (requirements.txt)

flask>=3.0.0
torch>=2.0.0
transformers>=4.25.0
scikit-learn>=1.2.0
numpy>=1.22.0
pandas>=1.4.0
joblib>=1.2.0
matplotlib>=3.5.0
seaborn>=0.12.0

