import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os

# Set plotting style
sns.set_theme(style="whitegrid")

def load_and_inspect_data():
    path = 'News_dataset/processed_news.csv'  # It correctly caught your file name!
    print(f"Loading unified dataset from: {path}")
    df = pd.read_csv(path)
    
    df.dropna(subset=['cleaned_content'], inplace=True)
    
    print("\n--- Dataset Info ---")
    print(df.info())
    
    # Auto-detect text column
    text_col = None
    for col in ['text', 'cleaned_text', 'cleaned_content', 'title']:
        if col in df.columns:
            text_col = col
            break
            
    # Auto-detect label column
    label_col = None
    for col in ['label', 'target', 'class', 'is_fake']:
        if col in df.columns:
            label_col = col
            break
            
    if not text_col or not label_col:
        raise ValueError(f"Could not auto-detect text or label columns. Found columns: {list(df.columns)}")
        
    print(f"\nUsing '{text_col}' for text analysis and '{label_col}' for labels.")
    return df, text_col, label_col

def analyze_lengths(df, text_col, label_col):
    print("\nCalculating word counts...")
    # Clean non-string elements just in case
    df[text_col] = df[text_col].astype(str)
    df['word_count'] = df[text_col].apply(lambda x: len(x.split()))
    
    plt.figure(figsize=(10, 6))
    
    # Plot length distributions grouped by label
    sns.histplot(data=df, x='word_count', hue=label_col, element='step', 
                 stat='density', common_norm=False, kde=True, bins=50, palette='Set2')
    
    plt.title('Article Length Distribution by Class', fontsize=14, fontweight='bold')
    plt.xlabel('Word Count per Article', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.xlim(0, 1500)  # Focus on the primary bulk of articles
    
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/length_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved layout profile to: plots/length_distribution.png")

def get_top_words(series, top_n=15):
    all_words = []
    for text in series:
        words = text.lower().split()
        # Drop extremely short tokens/punctuation artifacts
        all_words.extend([w for w in words if len(w) > 2])
    return Counter(all_words).most_common(top_n)

def plot_word_frequencies(df, text_col, label_col):
    print("Extracting top words per class...")
    distinct_labels = df[label_col].unique()
    
    if len(distinct_labels) != 2:
        print(f"Warning: Found {len(distinct_labels)} unique labels instead of 2: {distinct_labels}")
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    colors = ['Blues_r', 'Reds_r']
    
    for i, label_val in enumerate(distinct_labels[:2]):
        subset = df[df[label_col] == label_val][text_col]
        top_words = get_top_words(subset)
        
        words, counts = zip(*top_words)
        sns.barplot(x=list(counts), y=list(words), ax=axes[i], palette=colors[i % 2])
        axes[i].set_title(f'Top 15 Common Words (Label: {label_val})', fontsize=13, fontweight='bold')
        axes[i].set_xlabel('Frequencies')

    plt.tight_layout()
    plt.savefig('plots/word_frequencies.png', dpi=300)
    plt.close()
    print("Saved word frequency metrics to: plots/word_frequencies.png")

if __name__ == "__main__":
    try:
        df, text_col, label_col = load_and_inspect_data()
        analyze_lengths(df, text_col, label_col)
        plot_word_frequencies(df, text_col, label_col)
        print("\nEDA completed! Check the 'plots' folder for your generated figures.")
    except Exception as e:
        print(f"\nExecution failed: {e}")