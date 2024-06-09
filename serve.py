from flask import Flask, request, jsonify

app = Flask(__name__)

from reranker import ReRanker
from rawSearch import TFIDF

tf_idf = TFIDF()


@app.route('/api/search', methods=['POST'])
def handle_query():
    data = request.get_json()
    query = data.get('query')

    if not query:
        return jsonify({'error': 'No query provided'}), 400

    # Example response based on the query
    clean_query = process_query(query)

    # Search using TF-IDF
    filtered_results = tf_idf.search(clean_query, 10)

    # Re-rank using AI 
    re_ranker = ReRanker()

    scores = re_ranker.rank(query, filtered_results)

    return jsonify({'response': scores})

def process_query(query):
    return query

if __name__ == '__main__':
    app.run(debug=True)
