from typing import Sequence, Type, List, Dict, Tuple, NamedTuple
from tqdm import tqdm
import time
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from string import punctuation
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Initialize the NLTK lemmatizer and stop words
LEMMATIZER = WordNetLemmatizer()
STOPWORDS = set(stopwords.words('english'))


Document: Type = Dict[str, str]
"""
{
    title: <title>
    text: <text>
}
"""
Documents: Type = Dict[str, Document]
"""
  {
    doc_id: {
      title: <title>
      text: <text>
    },
  }
"""
Queries: Type = Dict[str, str]
GroundTruth: Type = Dict[str, Dict[str, int]]
Dataset: Type = Tuple[Documents, Queries, GroundTruth]
Tokens: Type = Sequence[str]


class TokenizedText(NamedTuple):
    text_id: str
    tokens: Tokens


TokenizedDocuments: Type = List[TokenizedText]


def get_tokenized_documents(docs: Documents) -> TokenizedDocuments:
    """
    The function tokenizes each document using NLTK.
    (lowercase conversion, stopword removal, punctuation removal, lemmatization)
    :param docs:
    :return:
    """
    cleaned_documents = []

    for doc_id, doc in tqdm(docs.items(), desc="Tokenizing documents"):
        # Tokenize the document, remove punctuation and stop words, and lemmatize
        tokens = lemmatization(tokenize(compact_document(doc)))
        cleaned_doc = ' '.join(tokens)
        cleaned_documents.append(TokenizedText(text_id=doc_id, tokens=cleaned_doc))

    return cleaned_documents


def compact_document(doc: Document) -> str:
    """
    Compact form of the document
    :param doc:
    :return: string representation of the document where the title and text fields are concatenated.
    """
    return f"{doc['title']} {doc['text']}"


def tokenize(text: str) -> Tokens:
    tokens = word_tokenize(text.lower())
    tokens = remove_punctuation(remove_stopwords(tokens))
    return tokens


def remove_stopwords(tokens: Tokens) -> Tokens:
    return [t for t in tokens if t not in STOPWORDS]


def remove_punctuation(tokens: Tokens) -> Tokens:
    return [t for t in tokens if t not in punctuation]


def lemmatization(tokens: Tokens) -> Tokens:
    return [LEMMATIZER.lemmatize(token) for token in tokens]


def vectorize(tokenized_docs: TokenizedDocuments) -> Tuple[csr_matrix, float]:
    """
    Vectorize documents using TfidfVectorizer and sort by IDF weights and docs length, also returns execution time
    :param tokenized_docs:
    :return:
    """
    # extract docs
    cleaned_docs = [doc.tokens for doc in tokenized_docs]

    vectorizer = TfidfVectorizer()
    start_time = time.time()
    tfidf_matrix = vectorizer.fit_transform(cleaned_docs)
    time_taken = time.time() - start_time

    # sort documents by IDF weights
    tfidf_mat_sort_idf = sort_by_idf(tfidf_matrix, vectorizer)
    # sort documents by length
    tfidf_mat_sort_length = sort_by_doc_length(tfidf_mat_sort_idf)

    return tfidf_mat_sort_length, time_taken


def sort_by_idf(tfidf_matrix: csr_matrix, vectorizer: TfidfVectorizer) -> csr_matrix:
    """
    Sort the documents by IDF weights in ascending order
    :param tfidf_matrix:
    :param vectorizer:
    :return:
    """
    idf_permutation = np.argsort(vectorizer.idf_)
    tfidf_matrix_sorted_idf = tfidf_matrix[:, idf_permutation]

    return tfidf_matrix_sorted_idf


def sort_by_doc_length(tfidf_matrix: csr_matrix) -> csr_matrix:
    """
    Sort the documents by length in descending order
    :param tfidf_matrix:
    :return:
    """
    doc_lengths = np.sum(tfidf_matrix, axis=1).flatten()
    length_permutation = np.argsort(doc_lengths)[::-1]
    tfidf_matrix_sorted_length = tfidf_matrix[length_permutation, :]

    return tfidf_matrix_sorted_length
