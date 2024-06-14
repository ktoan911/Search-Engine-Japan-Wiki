import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sentence_process import raw_process_query
import rawSearch
import reranker



rerank = reranker.ReRanker()
query = 'にほんご'
query = raw_process_query(query)

tf = rawSearch.TFIDF()
print(rerank.rank(query, tf.search(query, k=10), k=1))
