import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import rawSearch
import reranker
import TF_IDF_Search

rerank = reranker.ReRanker()
query = 'Dân chủ là gì?'

tfidf = TF_IDF_Search.TF_IDF_Init()
query = tfidf.preprocessing([query])[0]
print(query)

tf = rawSearch.TFIDF()
print(len(tf.search(query, k=20)))
#print(rerank.rank(query, tf.search(query, k=20), k=1))
