import os
import pandas as pd
import numpy as np
import scipy.sparse as sparse
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib


def export_vectorized_data():
    path = 'News_dataset/processed_news.csv'
    print(f"Reading processed data from {path}...")
    df = pd.read_csv(path)
    df.dropna(subset=['cleaned_content'], inplace=True)

    X = df['cleaned_content']
    y = df['label'].values

    print(f"Total samples: {len(X)} | Label distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    print("Splitting into train (80%) / test (20%)...")

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Fitting TF-IDF Vectorizer...")
    # FIX: Increased max_features — 5000 was leaving a lot of signal on the table
    # for a dataset this size (~45k articles). sublinear_tf reduces the dominance
    # of very frequent terms. ngram_range=(1,2) captures "white house", "fake news" etc.
    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_df=0.7,
        min_df=3,               # ignore very rare terms (typos, noise)
        max_features=50000,
        sublinear_tf=True,      # log(1+tf) instead of raw tf
        ngram_range=(1, 2),     # unigrams + bigrams
    )

    X_train_tfidf = vectorizer.fit_transform(X_train_text)
    X_test_tfidf  = vectorizer.transform(X_test_text)

    os.makedirs('models', exist_ok=True)
    os.makedirs('News_dataset/vectorized', exist_ok=True)

    joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')

    np.save('News_dataset/vectorized/y_train.npy', y_train)
    np.save('News_dataset/vectorized/y_test.npy',  y_test)
    sparse.save_npz('News_dataset/vectorized/X_train_tfidf.npz', X_train_tfidf)
    sparse.save_npz('News_dataset/vectorized/X_test_tfidf.npz',  X_test_tfidf)

    print("\nFeature engineering complete!")
    print(f"  Training matrix: {X_train_tfidf.shape}")
    print(f"  Testing matrix:  {X_test_tfidf.shape}")
    print("  Outputs saved to 'News_dataset/vectorized/'")


if __name__ == "__main__":
    export_vectorized_data()