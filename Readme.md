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


|Class Domain Profile| Precision| Recall | F1-Score| Overall Accuracy| 
|--------------|-------------|----------------|----------------|
|Human-Written (Class 0)| 92.00% | 89.00% | 0.90| 88.33% |
|AI-Generated (Class 1)| 83.00% | 87.00% | 0.85 | 88.33% | 

