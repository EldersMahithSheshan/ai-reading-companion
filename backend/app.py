import os
import joblib
import pandas as pd
import numpy as np
import spacy
import textstat
from wordfreq import zipf_frequency
import nltk
from nltk.corpus import stopwords
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import torch
import re

# --- 1. CONFIGURATION ---
app = Flask(__name__)
CORS(app)

# Load NLTK
nltk.download('stopwords', quiet=True)
STOP_WORDS = set(stopwords.words("english"))

# --- 2. LOAD AI MODELS ---
print("⏳ Loading Models... This might take a minute.")

# A. Load SpaCy (Medium model required for Vectors)
try:
    nlp = spacy.load("en_core_web_md")
    print("✅ SpaCy (Medium) Loaded.")
except OSError:
    print("❌ ERROR: Run 'python -m spacy download en_core_web_md'")
    exit()

# B. Load CWI Model (XGBoost)
try:
    cwi_model = joblib.load('models/best_cwi_model.pkl')
    print("✅ CWI XGBoost Model Loaded.")
except Exception as e:
    print(f"❌ Error loading CWI Model: {e}")

# C. Load Sinhala Definition Model (mBART)
try:
    model_path = "models/sinhala-model-v2"
    tokenizer = MBart50TokenizerFast.from_pretrained(model_path)
    llm_model = MBartForConditionalGeneration.from_pretrained(model_path)
    tokenizer.src_lang = "en_XX"
    tokenizer.tgt_lang = "si_LK"
    print("✅ Sinhala mBART Model Loaded.")
except Exception as e:
    print(f"❌ Error loading mBART Model: {e}")

# ---  HELPER FUNCTIONS ---

def fix_sinhala_rendering(text):
    """
    Restores Zero Width Joiners (ZWJ) to broken Sinhala conjunct characters.
    Ensures complex characters like 'ක්‍ර' and 'ද්‍ය' render correctly in browsers.
    """
    replacements = {
        # Yansaya (්‍ය)
        "ද් ය": "ද්‍ය",    # Fixes විද් යාව -> විද්‍යාව
        "ක් ය": "ක්‍ය",    # Fixes ක් යාවලිය -> ක්‍රියාවලිය
        "ත් ය": "ත්‍ය",
        
        # Rakaransaya (්‍ර)
        "ප් ර": "ප්‍ර",    # Fixes ප් රධාන -> ප්‍රධාන
        "ක් ර": "ක්‍ර",    # Fixes ක් රියා -> ක්‍රියා
        "ත් ර": "ත්‍ර",
        "ද් ර": "ද්‍ර",
        "බ් ර": "බ්‍ර",
        
        # Sanyaka/Special Conjuncts
        "ද්‍ධ": "ද්ධ",     # Fixes බුද්‍ධි -> බුද්ධි
        "න්‍ද": "න්ද",
        "න්‍ත": "න්ත",
    }
    for broken, fixed in replacements.items():
        text = text.replace(broken, fixed)
    return text


# --- 3. FEATURE ENGINEERING (Must match Training Code EXACTLY) ---
def get_token_features(row, doc):
    target = None
    for token in doc:
        # Find the exact token in the sentence
        if token.text == row['target'] or row['target'] in token.text:
            target = token
            break
    
    if not target:
        return [0, 0, 0.0, 'UNK', 'UNK']

    return [
        1 if target.ent_type_ else 0,    # is_ent
        1 if target.is_title else 0,     # is_title
        target.vector_norm,              # vector_norm (Crucial!)
        target.pos_,                     # pos
        target.dep_                      # dep
    ]

def extract_features(df):
    # 1. Statistical Features
    features = pd.DataFrame(index=df.index)
    features['char_len'] = df['target'].str.len()
    features['syllables'] = df['target'].apply(textstat.syllable_count)
    features['zipf_freq'] = df['target'].apply(lambda x: zipf_frequency(x, 'en'))
    features['is_stopword'] = df['target'].str.lower().isin(STOP_WORDS).astype(int)
    features['vowel_count'] = df['target'].str.count(r'[aeiouAEIOU]')
    features['consonant_count'] = features['char_len'] - features['vowel_count']

    # 2. Linguistic Features (Using SpaCy)
    # We process one by one for the API (slower than batch, but safer for requests)
    linguistic_rows = []
    for _, row in df.iterrows():
        doc = nlp(row['sentence'])
        linguistic_rows.append(get_token_features(row, doc))
    
    linguistic_df = pd.DataFrame(linguistic_rows, columns=['is_ent', 'is_title', 'vector_norm', 'pos', 'dep'])
    
    features = pd.concat([features, linguistic_df], axis=1)
    return features

# --- 4. API ROUTES ---

@app.route('/analyze', methods=['POST'])
def analyze_text():
    """ Step 1: Identify Complex Words """
    data = request.json
    text = data.get('text', '')
    
    if not text:
        return jsonify({"error": "No text"}), 400

  # 1. Parse text once
    doc = nlp(text)
    
    # 2. FAST FILTER: Only keep words that MIGHT be complex
    # We skip stopwords, punctuation, and short words (< 5 chars) immediately.
    candidates = []
    seen_words = set()

    for sent in doc.sents:
        for token in sent:
            # OPTIMIZATION IS HERE:
            # - Ignore words shorter than 5 characters (usually simple)
            # - Ignore stopwords (is, the, and)
            # - Ignore non-alphabetic tokens
            # - Ignore duplicates we've already processed
            if (len(token.text) < 5 or 
                token.is_stop or 
                not token.is_alpha or 
                token.text.lower() in seen_words):
                continue

            seen_words.add(token.text.lower())
            
            candidates.append({
                'target': token.text,
                'sentence': sent.text
            })
    
    if not candidates:
        return jsonify({"complex_words": []})

    df = pd.DataFrame(candidates)
    
    try:
        # 3. Extract features only for the "hard" candidates
        X_features = extract_features(df)
        
        # 4. Predict probability
        probs = cwi_model.predict_proba(X_features)[:, 1]
        df['complexity_score'] = probs
        
        # 5. Filter & Sort
        THRESHOLD = 0.35 
        WORD_LIMIT = 500 

        df_filtered = df[df['complexity_score'] > THRESHOLD] 
        df_sorted = df_filtered.sort_values(by='complexity_score', ascending=False)
        complex_words = df_sorted['target'].unique().tolist()[:WORD_LIMIT]
        
        return jsonify({"complex_words": complex_words})
        
    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({"error": str(e)}), 500
    
@app.route('/define', methods=['POST'])
def define_word():
    """ Step 2: Generate Sinhala Definition with Forced Formatting """
    data = request.json
    word = data.get('word', '')
    context = data.get('context', '')
    
# --- NEW SCRAPING CLEANUP LAYER ---
    # Removes fragments like '">', '">...', or leftover HTML tags
    def sanitize_scraped_text(text):
        text = re.sub(r'/">', '', text) # Removes /">
        text = re.sub(r'">', ' ', text)  # Removes ">
        text = re.sub(r'<[^>]+>', '', text) # Removes any actual HTML tags
        text = re.sub(r'">', ' ', text)
        text = re.sub(r'[a-zA-Z]+="[^"]*">', '', text)
        text = re.sub(r'<[^>]+>', '', text)

        
        return text.strip()

    word = sanitize_scraped_text(word)
    context = sanitize_scraped_text(context)
    
    if not word or not context:
        return jsonify({"error": "Missing word or context"}), 400

    # 1. KEYWORD GUARD: Fix for the "Interest" bias found during testing
    if word.lower() == "interest" and any(k in context.lower() for k in ["bank", "loan", "rate", "money", "percent"]):
        return jsonify({"definition": f"{word} යනු ණය මුදලක් සඳහා ගෙවනු ලබන අතිරේක මුදල (පොලිය) වේ."})

    # 2. PREPARE INPUT
    input_text = f"Explain the word '{word}' in Sinhala.\nContext: {context}"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=128, truncation=True)
    
    # 3. GENERATE (Using optimized stability parameters)
    with torch.no_grad():
        outputs = llm_model.generate(
            **inputs,
            max_length=256,
            num_beams=4,
            repetition_penalty=2.5,  # Stops repetitive bracket hallucinations
            length_penalty=1.0,      # Keeps definitions concise
            early_stopping=True,
            forced_bos_token_id=tokenizer.lang_code_to_id["si_LK"]
        )
    
    # 1. DECODE
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    
    # 2. THE "CLEANING" LAYER
    # A. Remove brackets like [17], [18], etc.
    prediction = re.sub(r'\[\d+\]', '', prediction)
    
    # B. Remove any English in parentheses like (intelligently)
    prediction = re.sub(r'\s*\([^)]*\)', '', prediction)
    
    # C. Cut off extra headers if the model starts babbling
    # This stops the text if the model starts explaining "Original" or "Reason"
    headers_to_cut = ["මුලාරම්භය:", "හේතුව:", "යථාර්ථය:", "මූලාශ්රය:", 
        "අන්තර්ගතය:", "අන්තර්ගතය", "ප්‍රධාන", "විස්තරය:", "සටහන:"]
    
    for header in headers_to_cut:
        # Split by header and take only the first part (everything before it)
        prediction = prediction.split(header)[0]

    # 3. APPLY YOUR "යනු" FORMATTING
    prefix = f"{word} යනු "
    if "යනු" in prediction:
        definition_part = prediction.split("යනු", 1)[1].strip()
        final_definition = f"{prefix}{definition_part}"
    else:
        clean_pred = prediction.replace(word, "").replace(word.lower(), "").strip()
        clean_pred = re.sub(r'^යනු\s*', '', clean_pred)
        final_definition = f"{prefix}{clean_pred}"

    # 4. FINAL TOUCH: Ensure the sentence ends cleanly with a full stop
    final_definition = final_definition.strip().rstrip('.') + "."

    final_definition = fix_sinhala_rendering(final_definition)

    return jsonify({"definition": final_definition})

    

if __name__ == '__main__':
    print("🚀 Server starting on Port 5000...")
    app.run(debug=True, port=5000)