from rawSearch import TFIDF
from reranker import ReRanker
from flask import Flask, request, jsonify
from sentence_process import raw_process_query

app = Flask(__name__)

tf_idf = TFIDF()
re_ranker = ReRanker()

# chunking to upadte document
@app.route('/api/search', methods=['POST'])
def handle_query():
    data = request.get_json()
    query = data.get('query')

    if not query:
        return jsonify({'error': 'No query provided'}), 400

    # Example response based on the query

    # Search using TF-IDF
    filtered_results = tf_idf.search(raw_process_query(query), 10)

    # Re-rank using AI
    scores = re_ranker.rank(query, filtered_results)

    return jsonify({'response': scores[0]})


if __name__ == '__main__':
    app.run(debug=True)
