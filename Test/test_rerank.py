import reranker_vibert
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

rerank = reranker_vibert.ReRanker_Vibert()
query = "What is the capital of France?"
docs = ["Paris is the capital of France", "France is in Europe"]
print(rerank.rank(query, docs, k=1))
