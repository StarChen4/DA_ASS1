import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re

# Download required NLTK data
nltk.download('stopwords')
nltk.download('wordnet')


def process_tokens(toks):
    # TODO: fill in the functions: process_tokens_1, 
    # process_tokens_2, process_tokens_3 functions and
    # uncomment the one you want to test below 
    # make sure to rebuild the index

    # return process_tokens_1(toks)
    # return process_tokens_2(toks)
    return process_tokens_3(toks)
    # return process_tokens_original(toks)


# get the nltk stopwords list
stopwords = set(nltk.corpus.stopwords.words("english"))

def process_tokens_original(toks):
    """ Perform processing on tokens. This is the Linguistics Modules
    phase of index construction

    Args:
        toks (list(str)): all the tokens in a single document

    Returns:
        list(str): tokens after processing
    """
    new_toks = []
    for t in toks:
        # ignore stopwords
        if t in stopwords or t.lower() in stopwords:
            continue
        # lowercase token
        t = t.lower()
        new_toks.append(t)
    return new_toks


def process_tokens_1(toks):
    """First modification: Stemming

    Args:
        toks (list(str)): all the tokens in a single document

    Returns:
        list(str): tokens after processing
    """
    stemmer = PorterStemmer()
    new_toks = []
    for t in toks:
        # ignore stopwords
        if t in stopwords or t.lower() in stopwords:
            continue
        # lowercase and stem token
        t = stemmer.stem(t.lower())  # Apply porterStemming on the tokens
        new_toks.append(t)
    return new_toks


def process_tokens_2(toks):
    """ Second modification: Lemmatization

    Args:
        toks (list(str)): all the tokens in a single document

    Returns:
        list(str): tokens after processing
    """
    lemmatizer = WordNetLemmatizer()
    new_toks = []
    for t in toks:
        # ignore stopwords
        if t in stopwords or t.lower() in stopwords:
            continue
        # lowercase and lemmatize token
        t = lemmatizer.lemmatize(t.lower())

        new_toks.append(t)
    return new_toks


def process_tokens_3(toks):
    """ Third modification: Lemmatization + Number handling + Special character removal

    Args:
        toks (list(str)): all the tokens in a single document

    Returns:
        list(str): tokens after processing
    """
    lemmatizer = WordNetLemmatizer()
    new_toks = []
    for t in toks:
        # ignore stopwords
        if t in stopwords or t.lower() in stopwords:
            continue
        # lowercase token
        t = t.lower()
        # remove special characters
        t = re.sub(r'[^a-zA-Z0-9]', '', t)
        # handle numbers
        if t.isdigit():
            if len(t) == 4:  # Possible year
                new_toks.append('year')
            else:
                new_toks.append('number')
        else:
            # lemmatize token
            t = lemmatizer.lemmatize(t)
            if t:  # only add non-empty tokens
                new_toks.append(t)
    return new_toks


def tokenize_text(data):
    """Convert a document as a string into a document as a list of
    tokens. The tokens are strings.

    Args:
        data (str): The input document

    Returns:
        list(str): The list of tokens.
    """
    # split text on spaces
    tokens = data.split()
    return tokens

