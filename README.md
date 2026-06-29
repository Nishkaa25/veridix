# Veridix — AI-Powered News Verification Engine

A hybrid fake news detection system combining classical NLP with LLM-based evidence retrieval. Classifies news headlines as **REAL**, **FAKE**, or **PARTIAL** with confidence scores, source-backed evidence links, and XAI-style explanations.

---

## Architecture

```
Headline Input
      │
      ▼
Preprocessing (NLTK lemmatization + TF-IDF vectorization)
      │
      ▼
ML Classifier (PassiveAggressiveClassifier — 99.5% on ISOT)
      │
      ├── Partial Verdict Detection (confidence < 80% or hedging language)
      │
      ├── Evidence Retrieval (Serper Dev API → Google Search)
      │
      └── LLM Explanation (Gemini 2.5 Flash → Groq LLaMA 3 fallback)
            │
            ▼
      FastAPI Backend → React Frontend
```

**Key design decision:** The ML model determines the verdict. The LLM only generates the human-readable explanation. This avoids hallucination affecting classification accuracy.

---

## Results

| Metric | Score |
|--------|-------|
| Accuracy (ISOT test set, n=8,980) | **99.5%** |
| Avg confidence | 99.9% |
| F1 — FAKE | 1.00 |
| F1 — REAL | 0.99 |

Trained and evaluated on the [ISOT Fake News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset).

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| ML model | scikit-learn `PassiveAggressiveClassifier` + isotonic calibration |
| Feature extraction | TF-IDF (50k features, bigrams, sublinear TF) |
| Preprocessing | NLTK (lemmatization, stopwords) |
| XAI - explanation | Google Gemini 2.5 Flash |
| XAI - Fallback LLM | Groq LLaMA 3 70B |
| Evidence search | Serper Dev API (Google Search) |
| Backend | FastAPI + Uvicorn |
| Frontend | React + TypeScript + Tailwind CSS + Chart.js |

---

## Setup

### 1. Clone & install

```bash
git clone https://github.com/YOUR_USERNAME/veridix.git
cd veridix
```

### 2. Backend

```bash
cd src
pip install -r requirements.txt
```

Create a `.env` file in `src/`:

```env
GEMINI_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
SERPER_API_KEY=your_key_here
```

### 3. Download dataset

Download from [Kaggle — ISOT Fake News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) and place `True.csv` and `Fake.csv` in `News_dataset/`.

### 4. Train the model

Run in order:

```bash
python preprocess.py       # clean + lemmatize
python vectorize.py        # TF-IDF feature extraction
python baseline_model.py   # train + save model
```

### 5. Start backend

```bash
python main.py
# Running at http://127.0.0.1:8000
```

### 6. Frontend

```bash
# from project root
npm install
npm run dev
```

---

## Project Structure

```
veridix/
├── src/
│   ├── main.py                  # FastAPI entry point
│   ├── verification_engine.py   # Core 3-layer pipeline
│   ├── preprocess.py            # NLTK cleaning pipeline
│   ├── vectorize.py             # TF-IDF vectorization
│   ├── baseline_model.py        # PAC classifier training
│   └── evaluate_engine.py       # Evaluation script
├── News_dataset/                # Add True.csv + Fake.csv here (not tracked)
├── models/                      # Saved model files (not tracked)
├── src/                         # React frontend
│   ├── App.tsx
│   ├── services/api.ts
│   └── types/verification.ts
├── .gitignore
├── requirements.txt
└── README.md
```
## Example Outputs

### REAL — Ukraine-Russia Peace Talks
<img width="1158" height="882" alt="Screenshot 2026-06-28 221744" src="https://github.com/user-attachments/assets/dd62054c-a65f-4802-a159-e8b8b57640b9" />


### FAKE — Sensational Political Clickbait
<img width="1162" height="887" alt="Screenshot 2026-06-28 221811" src="https://github.com/user-attachments/assets/a59209bc-f1e0-4030-aff2-432140870f93" />

### PARTIAL — Unverified Claims with Low Confidence
<img width="1033" height="873" alt="Screenshot 2026-06-28 221845" src="https://github.com/user-attachments/assets/5f2c554c-41b1-418c-bce2-c7331622820b" />


> These examples demonstrate the three verdict types Veridix returns —
> each with a credibility score, analysis metrics, LLM-generated summary,
> and retrieved evidence links.
---

## API

### `POST /api/verify`

```json
{
  "title": "Ukraine and Russia hold peace talks",
  "body": ""
}
```

Response:

```json
{
  "verdict": "REAL",
  "confidence": 99,
  "truth_percentage": 100,
  "fake_percentage": 0,
  "partial_reason": "",
  "breakdown": [{"label": "Verified true", "pct": 100, "color": "#2ecc71"}],
  "explanation": "...",
  "evidence_url": "https://reuters.com/...",
  "evidence_snippet": "...",
  "metrics": {
    "language_model_score": 99,
    "structural_accuracy": 97,
    "factual_correlation": 95
  }
}
```

### `GET /health`

```json
{"status": "ok", "model_loaded": true}
```

## Dataset

ISOT Fake News Dataset — University of Victoria  
~44,000 articles (21,417 real + 23,481 fake)  
[Kaggle link](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
