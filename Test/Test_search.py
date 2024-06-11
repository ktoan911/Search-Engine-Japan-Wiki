import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import reranker_dpr
import TF_IDF_Search
import reranker_vibert_faiss
import rawSearch
from sentence_process import process_query


rerank = reranker_vibert_faiss.ReRanker_Vibert()
query = 'miền bắc là gì?'
query = process_query(query)

tf = rawSearch.TFIDF()
print(rerank.rank(query, tf.search(query, k=10), k=1))
