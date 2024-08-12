import os
import random
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk import pos_tag, ngrams
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import string

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')


def load_documents(folder_path):
    """
    Load all documents from the specified folder.

    Args:
    folder_path (str): Path to the folder containing documents

    Returns:
    list: A list of strings, where each string is the content of a document
    """
    documents = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                documents.append(f.read())
    return documents


def process_document(text):
    """
    Process a single document: tokenize, remove stopwords and punctuation, lemmatize, and POS tag.

    Args:
    text (str): The text content of a document

    Returns:
    tuple: (tokens, lemmas, pos_tags) where each is a list
    """
    # Initialize tokenizer, lemmatizer, and stopwords
    tokenizer = RegexpTokenizer(r'\w+')
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    # Tokenize and convert to lowercase
    tokens = tokenizer.tokenize(text.lower())

    # Remove stopwords
    tokens = [token for token in tokens if token not in stop_words]

    # Lemmatize tokens
    lemmas = [lemmatizer.lemmatize(token) for token in tokens]

    # POS tagging
    pos_tags = pos_tag(tokens)

    return tokens, lemmas, pos_tags


def calculate_statistics(documents):
    """
    Calculate various statistics for the document collection.

    Args:
    documents (list): List of document strings

    Returns:
    dict: A dictionary containing various statistics
    """
    all_tokens = []
    all_lemmas = []
    doc_stats = []
    all_pos_tags = []

    for doc in tqdm(documents, desc="Processing documents"):
        tokens, lemmas, pos_tags = process_document(doc)
        all_tokens.extend(tokens)
        all_lemmas.extend(lemmas)
        all_pos_tags.extend(pos_tags)
        doc_stats.append({
            'tokens': len(tokens),
            'unique_tokens': len(set(tokens)),
            'lemmas': len(lemmas),
            'unique_lemmas': len(set(lemmas))
        })

    return {
        'number_documents': len(documents),
        'total_tokens': len(all_tokens),
        'unique_tokens': len(set(all_tokens)),
        'total_lemmas': len(all_lemmas),
        'unique_lemmas': len(set(all_lemmas)),
        'avg_tokens': sum(d['tokens'] for d in doc_stats) / len(doc_stats),
        'avg_unique_tokens': sum(d['unique_tokens'] for d in doc_stats) / len(doc_stats),
        'avg_lemmas': sum(d['lemmas'] for d in doc_stats) / len(doc_stats),
        'avg_unique_lemmas': sum(d['unique_lemmas'] for d in doc_stats) / len(doc_stats),
        'max_tokens': max(d['tokens'] for d in doc_stats),
        'min_tokens': min(d['tokens'] for d in doc_stats),
        'max_lemmas': max(d['lemmas'] for d in doc_stats),
        'min_lemmas': min(d['lemmas'] for d in doc_stats),
        'most_common_unigrams': get_most_common(all_tokens),
        'most_common_bigrams': get_most_common(list(ngrams(all_tokens, 2))),
        'most_common_trigrams': get_most_common(list(ngrams(all_tokens, 3))),
        'most_common_nouns': get_most_common([word for word, pos in all_pos_tags if pos.startswith('NN')]),
        'most_common_verbs': get_most_common([word for word, pos in all_pos_tags if pos.startswith('VB')]),
        'most_common_adjectives': get_most_common([word for word, pos in all_pos_tags if pos.startswith('JJ')]),
        'most_common_adverbs': get_most_common([word for word, pos in all_pos_tags if pos.startswith('RB')])
    }


def get_most_common(items, n=10):
    """
    Get the n most common items from a list.

    Args:
    items (list): List of items
    n (int): Number of most common items to return

    Returns:
    list: List of tuples (item, frequency)
    """
    return FreqDist(items).most_common(n)


def apply_zipf_law(tokens):
    """
    Apply Zipf's law to a list of tokens.

    Args:
    tokens (list): List of tokens

    Returns:
    tuple: (ranks, frequencies) where each is a list
    """
    freq_dist = FreqDist(tokens)
    frequencies = sorted(freq_dist.values(), reverse=True)
    ranks = range(1, len(frequencies) + 1)
    return ranks, frequencies


def plot_zipf(ranks, frequencies, title):
    """
    Plot Zipf's law graph.

    Args:
    ranks (list): List of ranks
    frequencies (list): List of frequencies
    title (str): Title of the plot
    """
    plt.figure(figsize=(10, 6))
    plt.loglog(ranks, frequencies, 'b.')
    plt.title(title)
    plt.xlabel('Rank')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(f'{title.lower().replace(" ", "_")}.png')
    plt.close()


def save_statistics_to_table(stats, filename='statistics.csv'):
    """
    Save statistics to a CSV file.

    Args:
    stats (dict): Dictionary containing statistics
    filename (str): Name of the file to save the statistics
    """
    df = pd.DataFrame(list(stats.items()), columns=['Statistic', 'Value'])
    df.to_csv(filename, index=False)
    print(f"Statistics saved to {filename}")


def main():
    """
    Main execution function.
    """
    # Load all documents
    documents = load_documents('./gov/documents')

    # Calculate statistics for the full collection
    print("Calculating statistics for full collection...")
    stats = calculate_statistics(documents)

    # Calculate subset based on university ID
    uni_id = 57725171  # Your university ID
    sample_percentage = 10000000 / uni_id
    subset_size = int(len(documents) * sample_percentage)
    subset = random.sample(documents, subset_size)

    # Calculate statistics for the subset
    print("Calculating statistics for subset...")
    subset_stats = calculate_statistics(subset)

    # Zipf's law analysis
    print("Applying Zipf's law...")
    all_tokens = [token for doc in tqdm(documents, desc="Processing full collection") for token in
                  process_document(doc)[0]]
    subset_tokens = [token for doc in tqdm(subset, desc="Processing subset") for token in process_document(doc)[0]]

    # Apply Zipf's law to full collection
    ranks, frequencies = apply_zipf_law(all_tokens)
    plot_zipf(ranks, frequencies, 'Zipf Law - Full Collection')

    # Apply Zipf's law to subset
    subset_ranks, subset_frequencies = apply_zipf_law(subset_tokens)
    plot_zipf(subset_ranks, subset_frequencies, 'Zipf Law - Subset')

    # Save statistics to CSV
    save_statistics_to_table(stats, 'full_collection_statistics.csv')
    save_statistics_to_table(subset_stats, 'subset_statistics.csv')

    print("Analysis complete. Check the CSV files and PNG images for results.")


if __name__ == "__main__":
    main()