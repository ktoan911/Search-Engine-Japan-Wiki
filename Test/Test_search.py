import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import rawSearch
import reranker_vibert
import TF_IDF_Search
import reranker_dpr



rerank = reranker_vibert.ReRanker_Vibert()
query = 'Dân chủ là gì?'

tfidf = TF_IDF_Search.TF_IDF_Init()
query = tfidf.preprocessing([query])[0]
print(query)

tf = rawSearch.TFIDF()
print(rerank.rank(query, tf.search(query, k=20), k=1))
