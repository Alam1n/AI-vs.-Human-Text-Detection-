import pandas as pd
import numpy as np
import re

def load_and_transform_daigt_dataset(file_path):
    """
    Loads your manual DAIGT CSV file and maps binary labels into a 
    custom Multi-Class structure spanning Human vs Specific AI sources vs Edited Text.
    """
    print(f"[*] Reading dataset from: {file_path}")
    # Load data
    df = pd.read_csv(file_path)
    
    # ----------------------------------------------------
    # Proposed Target Schema:
    # 0 -> Human Text
    # 1 -> ChatGPT Generated
    # 2 -> Gemini/Bard Generated
    # 3 -> Claude Generated
    # 4 -> Edited / Paraphrased AI Text
    # ----------------------------------------------------
    
    print("[*] Grouping and engineering multi-class labels using source metadata...")
    conditions = [
        (df['label'] == 0),                                      # Human
        (df['source'].str.contains('gpt|chatgpt', case=False, na=False)), # ChatGPT
        (df['source'].str.contains('gemini|palm|bard', case=False, na=False)), # Gemini
        (df['source'].str.contains('claude', case=False, na=False))      # Claude
    ]
    choices = [0, 1, 2, 3]
    
    # Assign our mapped classes, default everything else to Class 1 (General AI) temporarily
    df['multiclass_label'] = np.select(conditions, choices, default=1)
    
    # Filter the dataset down to just rows matching our targeted project setup
    # This keeps your project highly focused and computationally lighter
    target_sources = (df['multiclass_label'].isin([0, 1, 2, 3]))
    df_filtered = df[target_sources].copy()
    
    # ----------------------------------------------------
    # Engineering "Class 4: Edited AI Text" (Addressing Gap iv)
    # We take a subset of AI text and inject rule-based paraphrasing noise
    # ----------------------------------------------------
    print("[*] Artificially engineering real-world noise (Class 4: Edited AI)...")
    ai_indices = df_filtered[df_filtered['multiclass_label'].isin([1, 2, 3])].index
    # Sample 15% of the AI text to serve as edited/manipulated cheating text
    edited_indices = np.random.choice(ai_indices, size=int(len(ai_indices) * 0.15), replace=False)
    
    def inject_noise(text):
        # Simulates a student running text through a light paraphraser like Quillbot
        text = text.replace("However,", "On the other hand,")
        text = text.replace("Therefore,", "Consequently,")
        text = text.replace("Furthermore,", "In addition to this,")
        text = text.replace("for example", "instance by instance")
        return text

    df_filtered.loc[edited_indices, 'text'] = df_filtered.loc[edited_indices, 'text'].apply(inject_noise)
    df_filtered.loc[edited_indices, 'multiclass_label'] = 4  # Remap to Class 4
    
    print("\n[+] Final Multi-Class Transformation Distribution:")
    class_names = {0: "Human", 1: "ChatGPT", 2: "Gemini", 3: "Claude", 4: "Edited AI"}
    distribution = df_filtered['multiclass_label'].value_counts().rename(index=class_names)
    print(distribution)
    
    return df_filtered[['text', 'multiclass_label']]

# Verification Block
if __name__ == "__main__":
    # Change this to match your real CSV file name in your directory!
    csv_file_name = "train_v2_drcat_02.csv" 
    try:
        processed_df = load_and_transform_daigt_dataset(csv_file_name)
    except FileNotFoundError:
        print(f"[-] Could not find your file '{csv_file_name}'. Please verify the exact filename in your directory.")
        