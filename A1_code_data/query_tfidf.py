import math
from collections import defaultdict
from string_processing import (
    process_tokens,
    tokenize_text,
)
from query import (
    get_query_tokens,
    count_query_tokens,
    query_main,
)


def get_doc_to_norm(index, doc_freq, num_docs):
    """Pre-compute the norms for each document vector in the corpus using tfidf.

    Args:
        index (dict(str : list(tuple(int, int)))): The index aka dictionary of posting lists
        doc_freq (dict(str : int)): document frequency for each term
        num_docs (int): number of documents in the corpus

    Returns:
        dict(int: float): a dictionary mapping doc_ids to document norms
    """

    # TODO: Implement this function using tfidf
    # Hint: This function is similar to the get_doc_to_norm function in query.py
    #       but should use tfidf instead of term frequency
    doc_norm = defaultdict(float)

    # calculate square of norm for all docs
    for term, postings in index.items():
        # calculate idf
        idf = math.log2(num_docs / (1 + doc_freq[term]))
        for doc_id, tf in postings:
            tfidf = tf * idf  # tf-idf
            doc_norm[doc_id] += tfidf ** 2

    # take square root
    for docid in doc_norm.keys():
        doc_norm[docid] = math.sqrt(doc_norm[docid])

    return doc_norm


def run_query(query_string, index, doc_freq, doc_norm, num_docs):
    """ Run a query on the index and return a sorted list of documents. 
    Sorted by most similar to least similar.
    Documents not returned in the sorted list are assumed to have 0 similarity.

    Args:
        query_string (str): the query string
        index (dict(str : list(tuple(int, int)))): The index aka dictionary of posting lists
        doc_freq (dict(str : int)): document frequency for each term
        doc_norm (dict(int : float)): a map from doc_ids to pre-computed document norms
        num_docs (int): number of documents in the corpus

    Returns:
        list(tuple(int, float)): a list of document ids and the similarity scores with the query
        sorted so that the most similar documents to the query are at the top.
    """

    # pre-process the query string
    query_tokens = get_query_tokens(query_string)
    query_counts = count_query_tokens(query_tokens)

    # calculate the norm of the query vector
    query_vector = {}
    query_norm = 0
    for term, count in query_counts:
        if term in index:
            idf = math.log2(num_docs / (1 + doc_freq[term]))
            tfidf = count * idf
            query_vector[term] = tfidf
            query_norm += tfidf ** 2
    query_norm = math.sqrt(query_norm)

    # calculate cosine similarity for all relevant documents
    similarities = defaultdict(float)
    for term, query_tfidf in query_vector.items():
        for doc_id, tf in index[term]:
            idf = math.log2(num_docs / (1 + doc_freq[term]))
            doc_tfidf = tf * idf
            similarities[doc_id] += query_tfidf * doc_tfidf

    # Normalize similarities
    for doc_id in similarities:
        similarities[doc_id] /= (query_norm * doc_norm[doc_id])

    # Sort by similarity (descending order)
    sorted_docs = sorted(similarities.items(), key=lambda x: -x[1])

    return sorted_docs


if __name__ == '__main__':
    queries = [
        'Is nuclear power plant eco-friendly?',
        'How to stay safe during severe weather?',
    ]
    query_main(queries=queries, query_func=run_query, doc_norm_func=get_doc_to_norm)
    
