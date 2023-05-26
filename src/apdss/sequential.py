import itertools
import time
from typing import Tuple, List, Iterable

from pyspark import sql as psql

import src.tokenization as tok
from src.apdss.core import APDSS, Results, DocId, SparseScores, Document, DocsSimilarity, cosine_sim
from src.utilities.utils import AVAILABLE_CORES


class SequentialAPDSS(APDSS):
    def apdss(
            self, 
            spark: psql.SparkSession, 
            docs_scores_df: psql.DataFrame, 
            threshold: float,
            num_partitions: int = AVAILABLE_CORES
    ) -> Results:
        docs_scores_df_pandas = docs_scores_df.toPandas()
        ids: Iterable[DocId] = docs_scores_df_pandas[tok.ID_COL].array
        scores: Iterable[SparseScores] = docs_scores_df_pandas[tok.SCORES_COL].array

        # Start timer after conversion to pandas and other pre-processing
        # Just want to measure pure sequential apdss, also simulating an input
        #   that didn't need any transformation to make it sequential
        start_time = time.time()

        doc_id_scores_pairs: Iterable[Tuple[Document]] = zip(ids, scores)
        all_pairs: Iterable[Tuple[Document]] = itertools.combinations(doc_id_scores_pairs, 2)
        
        similar_docs: List[DocsSimilarity] = []
        for a, b in all_pairs:
            a_id, a_scores = a
            b_id, b_scores = b
        
            sim = cosine_sim(a_scores, b_scores)
            if sim >= threshold:
                similar_docs.append((a_id, b_id, sim))

        end_time = time.time()

        return Results(
            similar_docs=similar_docs,
            time=end_time - start_time
        )
