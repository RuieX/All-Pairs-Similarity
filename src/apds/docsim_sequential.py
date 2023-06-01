import os
import csv
import time
import pickle
from typing import Tuple, List, Dict
from tqdm import tqdm
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations


def exec_seq_apds(
        tfidf_docs: List[Tuple[str, csr_matrix, float, Dict]],
        thresholds: List[float],
        seq_results_path: str,
        samples_dir: str,
        heuristic: bool = False) -> Dict:
    """
    wrapper function to increase readability of the notebook
    execute the sequential algorithm
    :param tfidf_docs:
    :param thresholds:
    :param seq_results_path:
    :param samples_dir:
    :param heuristic:
    :return:
    """
    all_seq_results = {}

    if not os.path.exists(seq_results_path):
        for name, vectorized_docs, tfidf_time, idx_to_id in tqdm(tfidf_docs):
            seq_results = []
            for threshold in thresholds:
                similar_pairs, information = sequential_apds(tfidf_matrix=vectorized_docs,
                                                             sample_name=name,
                                                             threshold=threshold,
                                                             tfidf_time=tfidf_time,
                                                             heuristic=heuristic)
                mapped_pairs = map_doc_idx_to_id(similar_pairs, idx_to_id)
                seq_results.append((similar_pairs, mapped_pairs, information))
                save_seq_result_csv(all_pairs=mapped_pairs,
                                    ds_name=name,
                                    threshold=threshold,
                                    samples_dir=samples_dir,
                                    heuristic=heuristic)
            all_seq_results[name] = seq_results

        with open(seq_results_path, "wb") as f:
            pickle.dump(all_seq_results, f)
    else:
        with open(seq_results_path, "rb") as f:
            all_seq_results = pickle.load(f)

    return all_seq_results


def sequential_apds(
        tfidf_matrix: csr_matrix,
        sample_name: str,
        threshold: float,
        tfidf_time: float,
        heuristic: bool = False
) -> Tuple[List[Tuple[int, int, float]], Dict[str, object]]:
    """
    compute all pairs document similarity using cosine similarity and threshold
    :param tfidf_matrix:
    :param sample_name:
    :param threshold:
    :param tfidf_time:
    :param heuristic:
    :return:
    """
    similar_pairs = []
    n_docs = tfidf_matrix.shape[0]

    # compute pairwise cosine similarities
    start = time.time()
    similarity_matrix = cosine_similarity(tfidf_matrix)
    cosine_time = time.time() - start

    # all document lengths
    doc_lengths = tfidf_matrix.sum(axis=1).A1

    # find pairs
    start = time.time()
    for i, j in combinations(range(n_docs), 2):
        # skip comparison if the heuristic is enabled and the length difference is too large
        if heuristic:
            len_i = doc_lengths[i]
            len_j = doc_lengths[j]
            if abs(len_i - len_j) >= max(len_i, len_j) / 2:
                continue
        docs_sim = similarity_matrix[i, j]
        if docs_sim >= threshold:
            similar_pairs.append((i, j, docs_sim))
    find_time = time.time() - start

    result_dict = {
        'sample_name': sample_name,
        'threshold': threshold,
        'pairs_count': len(similar_pairs),
        'tfidf_time': tfidf_time,
        'cosine_time': cosine_time,
        'find_time': find_time,
        'total_time': tfidf_time + cosine_time + find_time
    }

    return similar_pairs, result_dict


def map_doc_idx_to_id(similar_pairs, doc_idx_to_id):
    mapped_pairs = []
    for pair in similar_pairs:
        i, j, sim = pair
        mapped_pairs.append((doc_idx_to_id[i], doc_idx_to_id[j], sim))
    return mapped_pairs


def save_seq_result_csv(
        all_pairs: List[Tuple[str, str, float]],
        ds_name: str,
        threshold: float,
        samples_dir: str,
        heuristic: bool = False) -> None:
    """
    save the document pairs and their similarity sorted by their similarity as a .csv file
    :param all_pairs: list of unique similar pair with the similarity
    :param ds_name: dataset name
    :param threshold: threshold used
    :param samples_dir: directory path of the ds_name sample
    :param heuristic: if the heuristic was used in computing the pairs
    :return:
    """
    save_dir = os.path.join(samples_dir, "seq_result", ds_name, f'{threshold}')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    filename = 'results_h.csv' if heuristic else 'results.csv'
    path = os.path.join(save_dir, filename)
    if not os.path.exists(path):
        with open(path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(all_pairs)


def load_results(seq_results_path: str) -> Dict:
    if not os.path.exists(seq_results_path):
        raise FileNotFoundError(f"Sequential results file not found at path: {seq_results_path}")
    else:
        with open(seq_results_path, "rb") as f:
            all_seq_results = pickle.load(f)

    return all_seq_results


def print_results(all_seq_results, sample_name):
    for (sim_pairs, sp_id, sp_info) in all_seq_results[sample_name]:
        print("--Run info--")
        for key, value in sp_info.items():
            print(f"{key}: {value}")
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
