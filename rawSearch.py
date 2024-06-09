import json


class TFIDF():
    def __init__(self, tf_idf_path, ds_path, doc_path):
        # Load
        with open(tf_idf_path, 'r') as infile:
            self.tf_idf_dict = json.load(infile)
        print("Load tf-idf done!")

        with open(ds_path, 'r') as infile:
            self.ds = json.load(infile)
        print("Load ds done!")

        with open(doc_path, 'r') as infile:
            self.docs = json.load(infile)
        print("Load docs done!")

    def search(self, q, k=10):
        def tf_idf_score(word, doc_idx):
            if word in self.tf_idf_dict:
                return self.tf_idf_dict[word][doc_idx]
            else:
                return 0
        # Search documents using TF-IDF
        finals = []
        # Lặp qua những văn bản
        for i in range():
            if self.ds[i] == 0:
                continue
            score = 0
            # Lặp qua các từ trong truy vấn
            for t in q.split():
                t = t.lower()
            score += tf_idf_score(t, i) / self.ds[i]
            finals.append((score, i))
        sorted(finals, key=lambda x: -x[0])
        results = [i for _, i in finals[:k]]
        return results
