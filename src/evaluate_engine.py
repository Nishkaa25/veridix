import os
import sys
import json
import requests
import joblib
import numpy as np
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import accuracy_score, classification_report
from dotenv import load_dotenv

load_dotenv()

nltk.download('stopwords', quiet=True)
nltk.download('wordnet',   quiet=True)

# ── Config ────────────────────────────────────────────────────────────────────
TRUE_CSV_PATH = "News_dataset/True.csv"
FAKE_CSV_PATH = "News_dataset/Fake.csv"
MODEL_PATH    = "models/baseline_classifier.pkl"
VEC_PATH      = "models/tfidf_vectorizer.pkl"
SAMPLE_SIZE   = 10   # keep small while Serper has rate limits


# ── Text cleaning (must match preprocess.py exactly) ─────────────────────────

def clean_for_inference(title: str, body: str = "") -> str:
    def _clean(text: str, strip_prefix: bool = False) -> str:
        if not isinstance(text, str):
            return ""
        text = text.lower()
        if strip_prefix:
            text = re.sub(r'^[a-z\s,\.]+\([^)]*\)\s*-\s*', '', text)
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'<.*?>', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))
        words = text.split()
        return " ".join([lemmatizer.lemmatize(w) for w in words
                         if w not in stop_words and len(w) > 1])

    cleaned_title = _clean(title, strip_prefix=False)
    cleaned_body  = _clean(body,  strip_prefix=True)
    return f"{cleaned_title} {cleaned_title} {cleaned_body}".strip()


# ── Serper Dev API ────────────────────────────────────────────────────────────

def search_serper(query: str, num_results: int = 3) -> list:
    """
    Calls Serper Dev API to get real Google Search results for a headline.
    Returns list of {title, link, snippet} dicts.
    """
    api_key = os.environ.get("SERPER_API_KEY")
    if not api_key:
        print("  ⚠️  SERPER_API_KEY not set in .env — skipping web search")
        return []

    try:
        response = requests.post(
            "https://google.serper.dev/search",
            headers={
                "X-API-KEY": api_key,
                "Content-Type": "application/json"
            },
            json={
                "q": query,
                "num": num_results,
                "gl": "us",
                "hl": "en"
            },
            timeout=8
        )
        response.raise_for_status()
        data = response.json()

        results = []

        # Top stories first (most relevant for news)
        for item in data.get("topStories", [])[:num_results]:
            results.append({
                "title":   item.get("title", ""),
                "link":    item.get("link", ""),
                "snippet": item.get("snippet", ""),
                "source":  "topStory"
            })

        # Organic results as fallback / supplement
        for item in data.get("organic", [])[:num_results]:
            if item.get("link") not in [r["link"] for r in results]:
                results.append({
                    "title":   item.get("title", ""),
                    "link":    item.get("link", ""),
                    "snippet": item.get("snippet", ""),
                    "source":  "organic"
                })

        return results[:num_results]

    except requests.exceptions.Timeout:
        print("  ⚠️  Serper timeout")
        return []
    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response else "?"
        if status == 429:
            print("  ⚠️  Serper rate limit hit")
        else:
            print(f"  ⚠️  Serper HTTP {status}: {e}")
        return []
    except Exception as e:
        print(f"  ⚠️  Serper error: {e}")
        return []


def pick_best_evidence_link(verdict: str, web_context: list, title: str):
    """
    Picks the most useful evidence URL depending on verdict:
    - FAKE → prefer fact-checking domains that debunk the claim
    - REAL → prefer credible original reporting sources
    Returns (url, snippet)
    """
    if not web_context:
        fallback_query = requests.utils.quote(title[:80])
        return f"https://news.google.com/search?q={fallback_query}", "No search results — Google News fallback."

    FACT_CHECK_DOMAINS = [
        "snopes.com", "politifact.com", "factcheck.org",
        "reuters.com/fact-check", "apnews.com/hub/ap-fact-check",
        "fullfact.org", "leadstories.com", "checkyourfact.com",
        "verifythis.com", "africacheck.org"
    ]

    CREDIBLE_NEWS = [
        "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk",
        "nytimes.com", "washingtonpost.com", "theguardian.com",
        "npr.org", "pbs.org", "nbcnews.com", "abcnews.go.com"
    ]

    if verdict == "FAKE":
        for result in web_context:
            link = result.get("link", "")
            if any(domain in link for domain in FACT_CHECK_DOMAINS):
                return link, result.get("snippet", "Fact-check source.")
        for result in web_context:
            link = result.get("link", "")
            if any(domain in link for domain in CREDIBLE_NEWS):
                return link, result.get("snippet", "Credible source (no direct fact-check found).")
    else:
        for result in web_context:
            link = result.get("link", "")
            if any(domain in link for domain in CREDIBLE_NEWS):
                return link, result.get("snippet", "Story confirmed by credible source.")

    first = web_context[0]
    return first.get("link", "https://news.google.com"), first.get("snippet", "")


# ── Dataset loading ───────────────────────────────────────────────────────────

def load_balanced_sample(n_per_class: int = SAMPLE_SIZE) -> pd.DataFrame:
    for path in [TRUE_CSV_PATH, FAKE_CSV_PATH]:
        if not os.path.exists(path):
            print(f"❌ Missing: {path}")
            sys.exit(1)

    df_true = pd.read_csv(TRUE_CSV_PATH)
    df_fake = pd.read_csv(FAKE_CSV_PATH)
    df_true['actual'] = 1
    df_fake['actual'] = 0

    combined = pd.concat([
        df_true.sample(n=n_per_class, random_state=None),
        df_fake.sample(n=n_per_class, random_state=None)
    ]).sample(frac=1).reset_index(drop=True)

    return combined


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    for path, label in [(MODEL_PATH, "model"), (VEC_PATH, "vectorizer")]:
        if not os.path.exists(path):
            print(f"❌ {label} not found at '{path}'.")
            print("   Run: preprocess.py → vectorize.py → baseline_model.py")
            sys.exit(1)

    print("🔧 Loading trained model and vectorizer...")
    model      = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VEC_PATH)

    df = load_balanced_sample(SAMPLE_SIZE)
    serper_enabled = bool(os.environ.get("SERPER_API_KEY"))
    print(f"🚀 Evaluating {len(df)} articles | "
          f"Web search: {'✅ Serper active' if serper_enabled else '❌ disabled — add SERPER_API_KEY to .env'}\n")

    y_true, y_pred, confidences = [], [], []

    for idx, (_, row) in enumerate(df.iterrows(), 1):
        title  = str(row.get('title', ''))
        body   = str(row.get('text',  ''))
        actual = row['actual']

        # ── ML classification ─────────────────────────────────────────────
        cleaned = clean_for_inference(title, body)
        vec     = vectorizer.transform([cleaned])
        pred    = model.predict(vec)[0]
        proba   = model.predict_proba(vec)[0]
        conf    = round(max(proba) * 100, 1)

        pred_label   = "REAL" if pred   == 1 else "FAKE"
        actual_label = "REAL" if actual == 1 else "FAKE"
        correct      = "✅" if pred == actual else "❌"

        y_true.append(actual)
        y_pred.append(pred)
        confidences.append(conf)

        print(f"{correct} [{idx:>3}] Actual: {actual_label:<4} | Pred: {pred_label:<4} | Conf: {conf:>5.1f}%")
        print(f"       Title: \"{title[:80]}\"")

        # ── Evidence via Serper ───────────────────────────────────────────
        search_results = search_serper(title, num_results=3)
        evidence_url, snippet = pick_best_evidence_link(pred_label, search_results, title)

        print(f"       🔗 Evidence: {evidence_url}")
        print(f"       📄 Context:  {snippet[:120]}")
        print()

    # ── Summary ───────────────────────────────────────────────────────────
    acc = accuracy_score(y_true, y_pred)
    print(f"{'='*65}")
    print(f"  Accuracy:        {acc*100:.2f}%")
    print(f"  Avg Confidence:  {np.mean(confidences):.1f}%")
    print(f"  Correct:         {sum(p==t for p,t in zip(y_pred,y_true))}/{len(y_true)}")
    print(f"{'='*65}")
    print(classification_report(
        y_true, y_pred,
        target_names=["FAKE", "REAL"],
        labels=[0, 1],
        zero_division=0
    ))


if __name__ == "__main__":
    main()