import json
import math
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict


class TF_IDF_Init():
    def __init__(self, folder_path):
        # Load data
        self.data = self.load_processed_data(folder_path)

    def load_processed_data(self, folder_path):
        data = []
        for filename in os.listdir(folder_path):
            with open(os.path.join(folder_path, filename), 'r') as f:  # open in readonly mode
                # do your stuff
                with open(filename, 'r') as f:
                    text = f.read()
                    data.append(self.preprocessing(text))
        return data

    def preprocessing(self, docs):
        letters = set(
            'aáàảãạăaáàảãạăắằẳẵặâấầẩẫậbcdđeéèẻẽẹêếềểễệfghiíìỉĩịjklmnoóòỏõọôốồổỗộơớờởỡợpqrstuúùủũụưứừửữựvwxyýỳỷỹỵz0123456789')

        def clean_word(w):
            new_w = ''
            for letter in w:
                if letter.lower() in letters or letter == '.':
                    new_w += letter.lower()

            return new_w
        new_docs = []
        new_doc = ''
        for i in range(len(docs)):
            doc = docs[i]
            doc = doc.replace('\n', ' ').replace('==', ' ')
            words = doc.split()
            for j in range(len(words)):
                word = clean_word(words[j])
                words[j] = word
            new_doc = ' '.join(words)
            new_docs.append(new_doc)
        return new_docs

    def compute_tf_idf(self, num_docs):
        def rounding(num):
            return math.floor(num * 1000) / 1000

        documents = self.data[:num_docs]

        # Tạo TfidfVectorizer
        vectorizer = TfidfVectorizer()

        # Huấn luyện mô hình TF-IDF và chuyển đổi tập văn bản
        tfidf_matrix = vectorizer.fit_transform(documents)

        # Lấy ra từ điển các từ và chỉ số của chúng
        feature_names = vectorizer.get_feature_names_out()
        tf_idf_dict = {key: defaultdict(float) for key in feature_names}

        # Chuyển đổi ma trận TF-IDF thành dạng dễ đọc
        dense = tfidf_matrix.todense()
        denselist = dense.tolist()

        # Danh sách mẫu số bên dưới
        ds = defaultdict(float)

        # In kết quả
        for doc_idx, doc in enumerate(denselist):
            d = 0
            for word_idx, score in enumerate(doc):
                tf_idf_dict[feature_names[word_idx]][doc_idx] = score
                d += score**2
            d_ = d ** (1/2)
            ds[doc_idx] = rounding(d_)
        return tf_idf_dict, ds

    def save_data(self, tf_idf_dict, ds):
        # Lưu dữ liệu
        with open('tf_idf.json', 'w') as outfile:
            json.dump(tf_idf_dict, outfile)
        print('tf_idf.json saved!')

        with open('../ds.json', 'w') as outfile:
            json.dump(ds, outfile)
        print('ds.json saved!')

        document = {
            "docs": self.data
        }
        with open('/content/docs.json', 'w') as outfile:
            json.dump(document, outfile)
        print('docs.json saved!')

    def get_file(self, num_docs):
        tf_idf_dict, ds = self.compute_tf_idf(num_docs)
        self.save_data(tf_idf_dict, ds)
        return tf_idf_dict, ds


tf_idf = TF_IDF_Init('/content/data')
tf_idf_dict, ds = tf_idf.get_file(1500)
