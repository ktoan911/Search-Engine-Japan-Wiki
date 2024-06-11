import os
from transformers import AutoTokenizer, AutoModel
import torch
from datasets import Dataset


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

        ds = Dataset.from_dict({"context": docs})

        # Tính embedding cho câu query
        query_embedding = self.get_embedding(query, type='query')

        # Tính embedding cho các context
        context_embeddings = self.get_embedding(docs, type='context')

        ds_with_embeddings = ds.add_column(
            'embeddings', context_embeddings.tolist())
        
        # Đánh index cho cột embeddings
        ds_with_embeddings.add_faiss_index(column='embeddings')

        # Tính toán độ tương đồng
        scores, retrieved_examples = ds_with_embeddings.get_nearest_examples(
            'embeddings', query_embedding, k=k)
        
        return retrieved_examples['context']
