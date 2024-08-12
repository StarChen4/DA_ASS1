import pickle
from string_processing import (
    process_tokens,
    tokenize_text,
)


def get_doc_id(item):
    """Helper function to get doc_id from either an int or a tuple.

    Args:
        item (int or tuple): The item which could either be an integer (doc_id)
                             or a tuple (doc_id, term_frequency).

    Returns:
        int: The doc_id extracted from the item.
    """
    return item if isinstance(item, int) else item[0]


def intersect_query(doc_list1, doc_list2):
    """Perform intersection of two sorted document lists.

    This function finds the common document IDs between two sorted lists of documents.

    Args:
        doc_list1 (list): The first sorted list of document items (either int or tuple).
        doc_list2 (list): The second sorted list of document items (either int or tuple).

    Returns:
        list: A list of document IDs that are present in both doc_list1 and doc_list2.
    """
    result = []  # List to store the intersected document IDs
    i, j = 0, 0  # Pointers for doc_list1 and doc_list2
    while i < len(doc_list1) and j < len(doc_list2):
        id1 = get_doc_id(doc_list1[i])  # Extract doc_id from doc_list1[i]
        id2 = get_doc_id(doc_list2[j])  # Extract doc_id from doc_list2[j]
        if id1 == id2:
            result.append(id1)  # If IDs match, add to the result
            i += 1  # Move both pointers forward
            j += 1
        elif id1 < id2:
            i += 1  # Move pointer i forward if id1 is smaller
        else:
            j += 1  # Move pointer j forward if id2 is smaller
    return result


def union_query(doc_list1, doc_list2):
    """Perform union of two sorted document lists.

    This function combines two sorted lists of document items and returns all unique document IDs.

    Args:
        doc_list1 (list): The first sorted list of document items (either int or tuple).
        doc_list2 (list): The second sorted list of document items (either int or tuple).

    Returns:
        list: A list of unique document IDs from both doc_list1 and doc_list2.
    """
    result = []  # List to store the union of document IDs
    i, j = 0, 0  # Pointers for doc_list1 and doc_list2
    while i < len(doc_list1) and j < len(doc_list2):
        id1 = get_doc_id(doc_list1[i])  # Extract doc_id from doc_list1[i]
        id2 = get_doc_id(doc_list2[j])  # Extract doc_id from doc_list2[j]
        if id1 == id2:
            result.append(id1)  # If IDs match, add to the result
            i += 1  # Move both pointers forward
            j += 1
        elif id1 < id2:
            result.append(id1)  # Add id1 to result if it's smaller
            i += 1  # Move pointer i forward
        else:
            result.append(id2)  # Add id2 to result if it's smaller
            j += 1  # Move pointer j forward
    # Add any remaining items from either list to the result
    result.extend([get_doc_id(doc) for doc in doc_list1[i:]])
    result.extend([get_doc_id(doc) for doc in doc_list2[j:]])
    return result


def run_boolean_query(query_string, index):
    """Runs a boolean query using the index.

    Args:
        query_string (str): boolean query string
        index (dict(str : list(tuple(int, int)))): The index aka dictionary of posting lists

    Returns:
        list(int): a list of doc_ids which are relevant to the query
    """
    tokens = query_string.split()  # Split the query string into tokens
    if not tokens:
        return []  # If no tokens, return an empty list

    # Initialize result with documents for the first token
    result = [get_doc_id(doc) for doc in index.get(tokens[0].lower(), [])]

    # Process each token and operator in the query
    for i in range(1, len(tokens), 2):
        operator = tokens[i]  # Get the operator (AND/OR)
        term = tokens[i + 1].lower()  # Get the next term
        doc_list = index.get(term, [])  # Retrieve the posting list for the term

        if operator == "AND":
            result = intersect_query(result, doc_list)  # Perform intersection
        elif operator == "OR":
            result = union_query(result, doc_list)  # Perform union

    return result  # Return the final list of document IDs


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
