from typing import Dict, List, Tuple
import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.utilities.tokenization import TokenizedDocuments, Tokens

Similarities = np.ndarray
DocumentCounts = Dict[int, int]
SimilarDocumentPairs = List[Tuple[int, int]]


# TODO first
def perform_all_pairs_docs_sim(tokenized_docs: TokenizedDocuments, threshold: float):
    # Extract the documents
    cleaned_docs = [doc.tokens for doc in tokenized_docs]
    # call functions
    classic_all_pairs_docs_sim(cleaned_docs, threshold)
    npargwhere_all_pairs_docs_sim(cleaned_docs, threshold)


# TODO count the number of similar documents for each document
#     for i in range(num_docs):
#         count = 0
#         for j in range(num_docs):
#             if similarities[i][j] > threshold:
#                 count += 1
#         counts[i] = count
#     return counts


def classic_all_pairs_docs_sim(docs_list: List[Tokens], threshold: float):
    count = 0
    doc_similaritis = []
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(docs_list)

    start = time.time()
    similarities = cosine_similarity(tfidf_matrix)
    for doc_1, doc_sims in enumerate(similarities):
        for doc_2, doc_sim in enumerate(doc_sims[(doc_1+1):], start=doc_1+1):
            if doc_sim >= threshold:  # d1 != d2 and threshold
                count += 1
                doc_similaritis.append((doc_1, doc_2, doc_sim))
    end = time.time()

    return doc_similaritis, {'threshold': threshold,
                             'similar_doc': count,
                             'elapsed': end-start}
# maybe use (doc_1, doc_2) pair ?


def npargwhere_all_pairs_docs_sim(docs_list: List[Tokens], threshold: float):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(docs_list)

    start = time.time()
    similarities = cosine_similarity(tfidf_matrix)
    np.fill_diagonal(similarities, 0.0)
    idx_doc_similaritis = np.argwhere(similarities > threshold)
    end = time.time()

    return [(similar.tolist(), similarities[similar[0], similar[1]])
            for similar in idx_doc_similaritis
            ], {'threshold': threshold,
                'similar_doc': int(len(idx_doc_similaritis) / 2),
                'elapsed': end - start}


# # TODO delete, Function to compute cosine similarity between each document:
# def compute_cosine_similarities(tokenized_documents: tkn.TokenizedDocuments) -> Similarities:
#     """
#     Computes the cosine similarity between each pair of documents in tokenized_documents.
#     takes in the list of TokenizedText objects, extracts the TF-IDF vectors from each object,
#     and computes the cosine similarity between each pair of documents using sklearn's cosine_similarity function.
#     :param tokenized_documents:
#     :return:
#     """
#     num_docs = len(tokenized_documents)
#     similarities = np.zeros((num_docs, num_docs))
#     tfidf_vectors = [doc.tfidf for doc in tokenized_documents]
#
#     for i in range(num_docs):
#         for j in range(num_docs):
#             similarities[i][j] = cosine_similarity(tfidf_vectors[i], tfidf_vectors[j])[0][0]
#
#     return similarities
#
#
# # TODO delete, Function to return all document pairs with a similarity greater than a given threshold:
# def get_similar_documents(similarities: Similarities, threshold: float) -> SimilarDocumentPairs:
#     """
#     Given the computed cosine similarities, returns all pairs of documents with a similarity greater than the threshold.
#     takes in the cosine similarity matrix and a threshold value, and returns a list of document pairs (i, j)
#     where the similarity between documents i and j is greater than the threshold.
#     Note that we only need to consider pairs (i, j) where i < j to avoid duplicating pairs.
#     :param similarities:
#     :param threshold:
#     :return:
#     """
#     num_docs = similarities.shape[0]
#     pairs = []
#
#     for i in range(num_docs):
#         for j in range(i + 1, num_docs):
#             if similarities[i][j] > threshold:
#                 pairs.append((i, j))
#
#     return pairs
