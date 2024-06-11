import os
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import torch
import numpy as np


# Set environment variable to allow duplicate OpenMP runtime
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

model = AutoModel.from_pretrained('vinai/phobert-base')
tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')


class ReRanker_Vibert():
    def __init__(self):
        self.model = model
        self.tokenizer = tokenizer

    def get_embedding(self, contexts, max_length=256, type='query'):
        # Hàm thực hiện map
        # Nhiệm vụ:
        # tokenizer mẫu hiện tại
        if (max_length > self.model.config.max_position_embeddings):
            raise ValueError(
                f"Max length of input ({max_length}) exceeds the maximum length for the model ({self.model.config.max_position_embeddings})."
            )
        if (type == 'query'):
            contexts = [contexts]

        tokens = tokenizer(contexts,
                           max_length=max_length,
                           truncation=True,
                           padding='max_length',
                           return_tensors='pt')

        # output: last_hidden_state ([num, max_length, 768]), pooler_output([num, 768])
        outputs = self.model(**tokens)

        embeddings = outputs.last_hidden_state
        # get mask max length
        mask = tokens['attention_mask'].unsqueeze(
            -1).expand(embeddings.size()).float()  # shape = ([num, max_length, 768])

        masked_embeddings = embeddings * mask

        # cộng tổng các embedding
        # shape: (number of sentence, embedding size)
        summed = torch.sum(masked_embeddings, 1)

        # Chuẩn hóa độ dài các câu bằng cách chia mỗi phần tử cho độ dài câu
        counted = torch.clamp(mask.sum(1), min=1e-9)
        mean_pooled = summed / counted

        return mean_pooled.detach().numpy()

    def rank(self, query, docs, k=1):

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
        result_docs = [docs[index]
                       for index, _ in sorted_list_with_indices[:k]]

        return result_docs
