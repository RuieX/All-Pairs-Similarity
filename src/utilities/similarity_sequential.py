from typing import Dict, List, Tuple
import time
import numpy as np
from itertools import combinations
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.utilities.preprocess import TokenizedDocuments, Tokens

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



# TODO fai struttura dati con risultato e tempo impiegato
# TODO stampa alcune coppie

# todo count pairs



def classic_all_pairs_docs_sim(docs_list: List[Tokens], threshold: float):
    count = 0
    similar_pairs = []
    # Compute TF-IDF matrix and similarity matrix
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(docs_list)
    n_docs = len(docs_list)


    # TODO fare fit_transform(docs_list) di tfidfvectorizer prima perche prende tempo (stampare tempo), e salvare


    start = time.time()
    similarity_matrix = cosine_similarity(tfidf_matrix)
    end = time.time()
    # todo save this time

    # Find pairs of similar documents
    start = time.time()
    for doc_1, doc_2 in combinations(range(n_docs), 2):
        docs_sim = similarity_matrix[doc_1, doc_2]
        if docs_sim > threshold:
            count += 1
            similar_pairs.append((doc_1, doc_2, docs_sim))
    end = time.time()
    # todo save this time

    return similar_pairs, {'threshold': threshold,
                             'similar_doc': count,
                             'elapsed': end-start}


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


def get_doc_ids(similar_docs, doc_idx_to_id):
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
