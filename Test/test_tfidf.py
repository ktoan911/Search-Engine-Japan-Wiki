import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import rawSearch


tf = rawSearch.TFIDF()
print(tf.search('Miền bắc', k=1))
