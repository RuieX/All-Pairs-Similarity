import time
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations


def sequential_DAPS(tfidf_matrix: csr_matrix, sample_name: str, threshold: float, tfidf_time, heuristic: bool = False):
    """
    Compute all pairs document similarity using cosine similarity and threshold
    :param tfidf_matrix:
    :param sample_name:
    :param threshold:
    :param tfidf_time:
    :param heuristic:
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
    for i, j in combinations(range(n_docs), 2):
        # Skip comparison if the heuristic is enabled and the length difference is too large
        if heuristic:
            len_i = len(tfidf_matrix[i].toarray()[0])
            len_j = len(tfidf_matrix[j].toarray()[0])
            if abs(len_i - len_j) >= max(len_i, len_j) / 2:
                continue
        docs_sim = similarity_matrix[i, j]
        if docs_sim >= threshold:
            similar_pairs.append((i, j, docs_sim))
    find_time = time.time() - start

    return similar_pairs, {'sample_name': sample_name,
                           'threshold': threshold,
                           'pairs_count': len(similar_pairs),
                           'tfidf_time': tfidf_time,
                           'cosine_time': cosine_time,
                           'find_time': find_time,
                           'total_time': tfidf_time+cosine_time+find_time}


def map_doc_idx_to_id(similar_pairs, doc_idx_to_id):
    mapped_pairs = []
    for pair in similar_pairs:
        i, j, sim = pair
        mapped_pairs.append((doc_idx_to_id[i], doc_idx_to_id[j], sim))
    return mapped_pairs


def print_results(all_seq_results, sample_name):
    for (sim_pairs, sp_id, sp_info) in all_seq_results[sample_name]:
        print("--Run info--")
        for key, value in sp_info.items():
            print(f"{key}: {value}")
        # print("--Similarity pairs--")
        # if len(sp_id) < 5:
        #     for doc1, doc2, similarity in sp_id:
        #         print(f"{doc1} and {doc2} have {similarity:.4f} similarity")
        # else:
        #     for i in range(5):
        #         doc1, doc2, similarity = sp_id[i]
        #         print(f"{doc1} and {doc2} have {similarity:.4f} similarity")
        print()


def print_first_pair(all_seq_results, sample_name, tokenized_samples):
    tokenized_docs = []
    for name, docs, _ in tokenized_samples:
        if name == sample_name:
            tokenized_docs = docs
    for (sim_pairs, sp_id, sp_info) in all_seq_results[sample_name]:
        if sp_id:
            doc1, doc2, similarity = sp_id[0]
            doc1_idx, doc2_idx, _ = sim_pairs[0]
            print(f"{doc1} and {doc2} have {similarity:.4f} similarity")
            print()
            print(f"{doc1}:")
            print(tokenized_docs[doc1_idx].tokens)
            print()
            print(f"{doc2}:")
            print(tokenized_docs[doc2_idx].tokens)
            print()
        else:
            print("There are 0 pairs")
    print()
