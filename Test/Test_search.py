import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import rawSearch
import reranker_vibert_faiss
import TF_IDF_Search
import reranker_dpr

rerank = reranker_vibert_faiss.ReRanker_Vibert()
query = 'miền bắc'

tfidf = TF_IDF_Search.TF_IDF_Init()
query = tfidf.preprocessing([query])[0]
print(query)

tf = rawSearch.TFIDF()
print(rerank.rank(query, tf.search(query, k=10), k=1))
