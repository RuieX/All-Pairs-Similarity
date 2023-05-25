import time
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


def compute_all_pairs_docs_sim(tfidf_matrix: csr_matrix, sample_name: str, threshold: float):
    """
    Compute all pairs document similarity using cosine similarity and thresholding
    :param tfidf_matrix:
    :param sample_name:
    :param threshold:
    :return:
    """
    similar_pairs = []
    n_docs = tfidf_matrix.shape[0]

    # Compute pairwise cosine similarities
    start = time.time()
    similarity_matrix = cosine_similarity(tfidf_matrix)
    cosine_time = time.time() - start

    # Find pairs of similar documents
    start = time.time()
    for i in range(n_docs):
        for j in range(i+1, n_docs):
            docs_sim = similarity_matrix[i, j]
            if docs_sim >= threshold:
                similar_pairs.append((i, j, docs_sim))
    find_time = time.time() - start

    return similar_pairs, {'sample_name': sample_name,
                           'threshold': threshold,
                           'pairs_count': len(similar_pairs),
                           'cosine_time': cosine_time,
                           'find_time': find_time}


def map_doc_idx_to_id(similar_pairs, doc_idx_to_id):
    mapped_pairs = []
    for pair in similar_pairs:
        i, j, sim = pair
        mapped_pairs.append((doc_idx_to_id[i], doc_idx_to_id[j], sim))
    return mapped_pairs
