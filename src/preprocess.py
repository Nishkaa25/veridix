import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

def clean_article_text(text: str, strip_source_prefix: bool = True) -> str:
    """
    Core text cleaning pipeline for ISOT dataset articles.
    
    Args:
        strip_source_prefix: Only strip "CITY (AGENCY) -" style prefixes from
                             article bodies, NOT from titles.
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()

    if strip_source_prefix:
        # FIX: Was `^.*?\s*-\s*` which ate the ENTIRE article body up to the
        # last dash. Now only strips a real source prefix like:
        # "WASHINGTON (Reuters) - " or "NEW YORK (AP) -"
        text = re.sub(r'^[a-z\s,\.]+\([^)]*\)\s*-\s*', '', text)

    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    words = text.split()
    cleaned = [lemmatizer.lemmatize(w) for w in words if w not in stop_words and len(w) > 1]

    return " ".join(cleaned)


def clean_title_only(text: str) -> str:
    """
    Lighter cleaning for titles — no source prefix stripping,
    preserves more signal since titles are already short.
    """
    return clean_article_text(text, strip_source_prefix=False)


def load_and_prepare_isot(true_path: str, fake_path: str) -> pd.DataFrame:
    print("Loading raw ISOT datasets...")
    true_df = pd.read_csv(true_path)
    fake_df = pd.read_csv(fake_path)

    true_df['label'] = 1   # REAL
    fake_df['label'] = 0   # FAKE

    combined_df = pd.concat([true_df, fake_df], ignore_index=True)
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Combine title + body — both get cleaned separately to avoid prefix
    # stripping from clobbering the title
    print(f"Loaded {len(combined_df)} total rows. Starting text cleaning pipeline...")

    combined_df['cleaned_title'] = combined_df['title'].apply(clean_title_only)
    combined_df['cleaned_body']  = combined_df['text'].apply(
        lambda t: clean_article_text(t, strip_source_prefix=True)
    )

    # Weight title slightly by repeating it — titles are high-signal for ISOT
    combined_df['cleaned_content'] = (
        combined_df['cleaned_title'] + " " +
        combined_df['cleaned_title'] + " " +   # title repeated = soft upweight
        combined_df['cleaned_body']
    )

    print("Data processing complete!")
    return combined_df


if __name__ == "__main__":
    import os
    os.makedirs('News_dataset', exist_ok=True)
    df = load_and_prepare_isot('News_dataset/True.csv', 'News_dataset/Fake.csv')
    df[['cleaned_content', 'cleaned_title', 'label']].to_csv(
        'News_dataset/processed_news.csv', index=False
    )
    print("Saved processed data to News_dataset/processed_news.csv")
    print(f"Class distribution:\n{df['label'].value_counts()}")