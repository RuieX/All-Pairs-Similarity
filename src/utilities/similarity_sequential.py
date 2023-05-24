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


# TODO fare fit_transform(docs_list) di tfidfvectorizer prima perche prende tempo (stampare tempo), e salvare
# todo count pairs
# TODO  Get a permutation that sorts the IDF weights in ascending order
# self._idf_permutation = np.argsort(tfidf_vectorizer.idf_)


# bozza
#         # Compute TF-IDF matrix and similarity matrix
#         vectorizer = TfidfVectorizer()
#         tfidf_matrix = vectorizer.fit_transform(corpus)
#         similarity_matrix = cosine_similarity(tfidf_matrix)
#
#         # Find pairs of similar documents
#         n_docs = len(corpus)
#         similar_pairs = []
#         for i, j in combinations(range(n_docs), 2):
#             if similarity_matrix[i, j] > threshold:
#                 similar_pairs.append((i, j))
#
#         # Print similar pairs
#         for pair in similar_pairs:
#             print(pair)

# TODO fai struttura dati con risultato e tempo impiegato

# TODO stampa alcune coppie

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


# variante figa similseba:
#     # Find pairs of similar documents
#     n_docs = len(corpus)
#     similar_pairs = []
#     for i, j in combinations(range(n_docs), 2):
#         if similarity_matrix[i, j] > threshold:
#             similar_pairs.append((i, j))
#
#     # Print similar pairs
#     for pair in similar_pairs:
#         print(pair)



# figo TODO

def npargwhere_all_pairs_docs_sim(docs_list: List[Tokens], threshold: float):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(docs_list)

    start = time.time()
    similarities = cosine_similarity(tfidf_matrix)
    np.fill_diagonal(similarities, 0.0)
    idx_doc_similarities = np.argwhere(similarities > threshold)
    idx_doc_similarities = set(pair for pair in idx_doc_similarities if pair[0] != pair[1])
    end = time.time()

    similar_docs = [(pair, similarities[pair[0], pair[1]]) for pair in idx_doc_similarities]

    return similar_docs, {'threshold': threshold,
                          'similar_doc': len(similar_docs),
                          'elapsed': end - start}


def get_docIDs(similar_docs, doc_idx_to_id):
    docid_similar_docs = []
    for pair, similarity in similar_docs:
        doc1_id = doc_idx_to_id[pair[0]]
        doc2_id = doc_idx_to_id[pair[1]]
        docid_pair = (doc1_id, doc2_id)
        docid_similar_docs.append((docid_pair, similarity))
    return docid_similar_docs

# end TODO

# TODO come usare
    # similar_docs, metadata = npargwhere_all_pairs_docs_sim(tokenized_docs, 0.5)
    # docid_similar_docs = get_docIDs(similar_docs, doc_idx_to_id)
    #
    # for pair, similarity in docid_similar_docs:
    #     doc1_id, doc2_id = pair
    #     print(f"Similarity between documents '{doc1_id}' and '{doc2_id}': {similarity}")



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
