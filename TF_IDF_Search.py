import json
import math
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from datasets import load_datase
import MeCab
import re
from datasets import load_dataset


class TF_IDF_Init():
    def __init__(self, folder_path=None, retrain=False):
        # Load data
        if retrain:
            doc_temp = self.load_processed_data()
            self.data = [self.preprocess_text(doc) for doc in doc_temp]

    def load_processed_data(self, name_dataset="fujiki/llm-japanese-dataset_wikipedia", num_docs=7000):
        dataset = load_dataset(name_dataset)
        return dataset['train']['output'][:num_docs]

    def preprocess_text(self, text):
        # Loại bỏ các ký tự không mong muốn
        # Loại bỏ ký tự đặc biệt nhưng giữ lại các ký tự tiếng Nhật
        text = re.sub(r'[^\w\sぁ-んァ-ン一-龥々ー]', '', text)
        text = re.sub(r'\d+', '', text)  # Loại bỏ số

        # Tách từ sử dụng MeCab
        mecab = MeCab.Tagger('-Owakati')
        tokenized_text = mecab.parse(text).strip()

        return tokenized_text

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


# tf_idf = TF_IDF_Init('/content/data')
# tf_idf_dict, ds = tf_idf.get_file(1500)
