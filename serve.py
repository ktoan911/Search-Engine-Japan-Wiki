import TF_IDF_Search
from rawSearch import TFIDF
from reranker_vibert import ReRanker_Vibert
from flask import Flask, request, jsonify

app = Flask(__name__)


tf_idf = TFIDF()
tfidf = TF_IDF_Search.TF_IDF_Init()
re_ranker = ReRanker_Vibert()


@app.route('/api/search', methods=['POST'])
def handle_query():
    data = request.get_json()
    query = data.get('query')

    if not query:
        return jsonify({'error': 'No query provided'}), 400

    # Example response based on the query
    clean_query = tfidf.preprocessing([query])[0]

    # Search using TF-IDF
    filtered_results = tf_idf.search(clean_query, 10)

    # Re-rank using AI
    scores = re_ranker.rank(query, filtered_results)

    return jsonify({'response': scores})


def process_query(query):
    return query


if __name__ == '__main__':
    app.run(debug=True)
