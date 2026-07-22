# ==========================================
# FINAL APP.PY - INDEPENDENT FLASK DEPLOYMENT
# ==========================================
from flask import Flask, request, render_template_string
import joblib
import numpy as np
import re
import torch
from transformers import AutoTokenizer, AutoModel

app = Flask(__name__)

# --- Re-initialize the Laptop-Friendly Extractors Locally inside Flask ---
print("[*] Launching Web Server ... Initializing Text Preprocessing Engine...")
STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "if", "because", "as", "what", "which", 
    "this", "that", "these", "those", "is", "are", "was", "were", "be", "been", "being"
}

print("[*] Loading DistilBERT Embedding Weights into App Sandbox...")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased")
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# --- Load the Saved Classifier and Scaler Weights ---
print("[*] Loading Trained Machine Learning Pipeline Artifacts...")
trained_rf_model = joblib.load("thesis_final_assets/weighted_binary_rf_model.pkl")
fitted_scaler = joblib.load("thesis_final_assets/feature_scaler.pkl")
tfidf_vectorizer = joblib.load("thesis_final_assets/tfidf_vectorizer.pkl")
print("[+] System Ready! Web app running securely on:", device)

# --- Native Feature Processing Core Functions ---
def local_preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    cleaned = [w for w in words if w not in STOP_WORDS]
    return " ".join(cleaned)

def local_extract_linguistic_features(text):
    raw_text = str(text)
    sentences = [s.strip() for s in re.split(r'[.!?]', raw_text) if s.strip()]
    words = raw_text.split()
    if len(words) == 0:
        return [0.0, 0.0, 0.0]
    lexical_diversity = len(set(words)) / len(words)
    avg_sentence_len = len(words) / max(len(sentences), 1)
    sent_lengths = [len(s.split()) for s in sentences]
    variance = np.std(sent_lengths) if len(sent_lengths) > 1 else 0.0
    return [lexical_diversity, avg_sentence_len, variance]

def local_extract_distilbert_embeddings(text):
    inputs = tokenizer(text, padding=True, truncation=True, max_length=64, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()


# --- HTML UI Template Definition ---
HTML_UI = """
<!DOCTYPE html>
<html>
<head>
    <title>Thesis AI Text Detector</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background-color: #f4f6f9; color: #333; }
        .container { max-width: 700px; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 4px 10px rgba(0,0,0,0.1); margin: auto; }
        h2 { color: #1F77B4; }
        textarea { width: 100%; height: 180px; padding: 12px; border-radius: 4px; border: 1px solid #ccc; font-size: 14px; box-sizing: border-box; resize: none;}
        button { background-color: #1F77B4; color: white; border: none; padding: 12px 20px; font-size: 16px; border-radius: 4px; cursor: pointer; margin-top: 10px; width: 100%;}
        button:hover { background-color: #155B8A; }
        .result-box { margin-top: 25px; padding: 15px; border-radius: 4px; font-weight: bold; font-size: 18px; text-align: center;}
        .ai { background-color: #F8D7DA; color: #721C24; border: 1px solid #F5C6CB; }
        .human { background-color: #D4EDDA; color: #155724; border: 1px solid #C3E6CB; }
    </style>
</head>
<body>
<div class="container">
    <h2>Hybrid AI Text Classification System</h2>
    <p>Paste an essay snippet below to generate real-time structural predictions.</p>
    <form method="POST" action="/">
        <textarea name="user_text" placeholder="Type or paste text content here..." required>{{ input_text }}</textarea>
        <button type="submit">Execute Classification Matrix Pipeline</button>
    </form>

    {% if prediction %}
    <div class="result-box {{ 'ai' if prediction == 'AI-Generated' else 'human' }}">
        System Result: {{ prediction.upper() }} <br>
        <span style="font-size:14px; font-weight:normal;">Model Confidence Level: {{ confidence }}%</span>
    </div>
    {% endif %}
</div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home_index():
    if request.method == "POST":
        raw_input_text = request.form["user_text"]
        
        # 1. Transform raw web input string into preprocessed text
        cleaned = local_preprocess_text(raw_input_text)
        
        # 2. Rebuild the exact 971-feature vector shape expected by the Random Forest model
        v_tfidf = tfidf_vectorizer.transform([cleaned]).toarray()
        v_ling = np.array([local_extract_linguistic_features(raw_input_text)])
        v_deep = np.array([local_extract_distilbert_embeddings(raw_input_text)])
        
        # 3. Combine features horizontally and apply your normalization weights
        hybrid_vector = np.hstack((v_tfidf, v_ling, v_deep))
        scaled_vector = fitted_scaler.transform(hybrid_vector)
        
        # 4. Run model prediction
        prediction = trained_rf_model.predict(scaled_vector)[0]
        probs = trained_rf_model.predict_proba(scaled_vector)[0]
        
        res_string = "AI-Generated" if prediction == 1 else "Human-Written"
        conf_string = f"{probs[prediction]*100:.2f}"
        
        return render_template_string(HTML_UI, prediction=res_string, confidence=conf_string, input_text=raw_input_text)
    
    return render_template_string(HTML_UI, prediction=None, input_text="")

if __name__ == "__main__":
    app.run(debug=True)