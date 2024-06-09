from transformers import DPRQuestionEncoder, DPRContextEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoderTokenizer
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

import os

# Set environment variable to allow duplicate OpenMP runtime
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

question_encoder = DPRQuestionEncoder.from_pretrained(
    "facebook/dpr-question_encoder-single-nq-base")
context_encoder = DPRContextEncoder.from_pretrained(
    "facebook/dpr-ctx_encoder-single-nq-base")

question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
    "facebook/dpr-question_encoder-single-nq-base")
context_tokenizer = DPRContextEncoderTokenizer.from_pretrained(
    "facebook/dpr-ctx_encoder-single-nq-base")


class ReRanker():
    def __init__(self):
        self.question_encoder = question_encoder
        self.context_encoder = context_encoder

        self.question_tokenizer = question_tokenizer
        self.context_tokenizer = context_tokenizer

    def get_embedding(self, contexts, type='query'):
        # Hàm thực hiện map
        # Nhiệm vụ:
        # tokenizer mẫu hiện tại
        if (type == 'query'):
            question_inputs = self.question_tokenizer(
                contexts, return_tensors="pt")
            question_embedding = self.question_encoder(
                **question_inputs).pooler_output
            # shape(1, embedding size)
            return question_embedding.detach().numpy()

        if (type == 'context'):
            # Tạo batch các inputs cho các context
            batch_context_inputs = self.context_tokenizer(
                contexts, padding=True, truncation=True, return_tensors="pt")
            # Tokenize batch context
            with torch.no_grad():
                batch_context_embeddings = self.context_encoder(
                    **batch_context_inputs).pooler_output
            # shape: (number of sentence, embedding size)
            return batch_context_embeddings.detach().numpy()

    def rank(self, query, docs):

        # Load các docs từ TF-IDF và chuyển thành Hugging Dataset
        # Thực hiện tìm vector tương đồng với query
        # Trả về kết quả

        # Tính embedding cho câu query
        query_embedding = self.get_embedding(query, type='query')

        # Tính embedding cho các context
        context_embeddings = self.get_embedding(docs, type='context')

        # Tính toán độ tương đồng
        similarities = [cosine_similarity(query_embedding, np.expand_dims(
            context_embedding_np, axis=0)).flatten()[0] for context_embedding_np in context_embeddings]

        # Sắp xếp list theo thứ tự giảm dần và bao gồm cả số thứ tự
        sorted_list_with_indices = sorted(
            enumerate(similarities), key=lambda x: x[1], reverse=True)

        # Trích xuất chỉ số và giá trị từ danh sách đã sắp xếp
        result_docs = [docs[index] for index, _ in sorted_list_with_indices]

        return result_docs
