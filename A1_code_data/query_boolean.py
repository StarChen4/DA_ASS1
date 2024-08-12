import pickle
from string_processing import (
    process_tokens,
    tokenize_text,
)


def intersect_query(doc_list1, doc_list2):
    # TODO: you might like to use a function like this in your run_boolean_query implementation
    # for full marks this should be the O(n + m) intersection algorithm for sorted lists
    # using data structures such as sets or dictionaries in this function will not score full marks

    return res


def union_query(doc_list1, doc_list2):
    # TODO: you might like to use a function like this in your run_boolean_query implementation
    # for full marks this should be the O(n + m) union algorithm for sorted lists
    # using data structures such as sets or dictionaries in this function will not score full marks

    return res


def run_boolean_query(query_string, index):
    """Runs a boolean query using the index.

    Args:
        query_string (str): boolean query string
        index (dict(str : list(tuple(int, int)))): The index aka dictionary of posting lists

    Returns:
        list(int): a list of doc_ids which are relevant to the query
    """

    # TODO: implement this function

    return relevant_docs


if __name__ == '__main__':
    # load the stored index
    (index, doc_freq, doc_ids, num_docs) = pickle.load(open("stored_index.pkl", "rb"))

    print("Index length:", len(index))
    if len(index) != 808777:
        print("Warning: the length of the index looks wrong.")
        print("Make sure you are using `process_tokens_original` when you build the index.")
        raise Exception()

    # the list of queries asked for in the assignment text
    queries = [
        "Workbooks",
        "Australasia OR Airbase",
        "Warm AND WELCOMING",
        "Global AND SPACE AND economies",
        "SCIENCE OR technology AND advancement AND PLATFORM",
        "Wireless OR Communication AND channels OR SENSORY AND INTELLIGENCE",
    ]

    # run each of the queries and print the result
    ids_to_doc = {docid: path for (path, docid) in doc_ids.items()}
    for query_string in queries:
        print(query_string)
        doc_list = run_boolean_query(query_string, index)
        res = sorted([ids_to_doc[docid] for docid in doc_list])
        for path in res:
            print(path)

