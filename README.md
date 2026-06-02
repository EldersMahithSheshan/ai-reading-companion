# 📖 AI-Powered Reading Companion

A proactive Chrome browser extension that assists EFL learners by automatically identifying complex English vocabulary and generating rich, descriptive Sinhala explanations in real-time.

## 🚀 The Problem & Solution
Traditional digital reading aids rely on a "stop-and-search" reactive method, which increases cognitive load. They also often return superficial, single-word translations. This project solves this by using a dual-model AI pipeline to proactively highlight difficult words and instantly provide full-sentence, dictionary-style Sinhala explanations.

## 🛠️ Architecture & Tech Stack
This project utilizes a 3-tier architecture to ensure low latency and frontend stability:

* **Frontend (Chrome Extension):** JavaScript (ES6), Manifest V3. Utilizes the native `TreeWalker` API for non-destructive DOM manipulation.
* **Backend (REST API):** Python 3.12, Flask. Handles CORS, model initialization, and Sinhala Unicode (ZWJ) post-processing.
* **Machine Learning (AI Engine):**
    * **Filter (CWI):** XGBoost Classifier utilizing a custom 11-feature linguistic pipeline (spaCy, textstat).
    * **Generator:** Fine-Tuned mBART-large-50 (HuggingFace, PyTorch) trained on a custom 8K verified dataset.

## ✨ Key Features
* **Proactive Assistance:** Flags difficult words automatically without user intervention.
* **Rich Explanatory Definitions:** Generates descriptive Sinhala sentences rather than basic synonyms.
* **Seamless Integration:** Does not break modern SPA frameworks (React/Angular) on host websites.
* **Sub-Second Latency:** Aggressive filtering reduces end-to-end inference to <0.6 seconds.

## ⚙️ Local Setup & Installation
1. Clone the repository: `git clone https://github.com/EldersMahithSheshan/ai-reading-companion.git`
2. Install backend dependencies: `pip install -r requirements.txt`
3. Download the fine-tuned model `.bin` files and place them in the `/models` directory.
4. Start the Flask server: `python app.py`
5. Load the `/extension` folder into Chrome via `chrome://extensions/` (Developer Mode).

## 📊 Evaluation & Metrics
* **XGBoost CWI Model:** 90.46% AUC-ROC, 82% Recall.
* **mBART-50 Generative Model:** 0.9692 ROUGE-L score.
