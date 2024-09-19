from transformers import MLukeTokenizer, LukeModel
import torch
from datasets import Dataset
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


class SentenceLukeJapanese:
    def __init__(self, model_name_or_path, device=None):
        self.tokenizer = MLukeTokenizer.from_pretrained(model_name_or_path)
        self.model = LukeModel.from_pretrained(model_name_or_path)
        self.model.eval()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(device)

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    @torch.no_grad()
    def encode(self, sentences, batch_size=8):
        all_embeddings = []
        iterator = range(0, len(sentences), batch_size)
        for batch_idx in iterator:
            batch = sentences[batch_idx:batch_idx + batch_size]

            encoded_input = self.tokenizer.batch_encode_plus(batch, padding="longest", 
                                           truncation=True, return_tensors="pt").to(self.device)
            model_output = self.model(**encoded_input)
            sentence_embeddings = self._mean_pooling(model_output, encoded_input["attention_mask"]).to('cpu')

            all_embeddings.extend(sentence_embeddings)

        return torch.stack(all_embeddings)


MODEL_NAME = "sonoisa/sentence-luke-japanese-base-lite"
model = SentenceLukeJapanese(MODEL_NAME)

class ReRanker:
    def __init__(self):
        self.model = model

    def get_embedding(self, contexts, batch_size=8):
        # Hàm thực hiện map
        # Nhiệm vụ:
        # tokenizer mẫu hiện tại
        if isinstance(contexts, str):
            contexts = [contexts]
        sentence_embeddings = model.encode(contexts, batch_size=batch_size)
        return sentence_embeddings.detach().numpy()

    def rank(self, query, docs, k=1):

        # Load các docs từ TF-IDF và chuyển thành Hugging Dataset
        # Thực hiện tìm vector tương đồng với query
        # Trả về kết quả

        ds = Dataset.from_dict({"context": docs})

        # Tính embedding cho câu query
        query_embedding = self.get_embedding([query])

        # Tính embedding cho các context
        context_embeddings = self.get_embedding(docs)

        ds_with_embeddings = ds.add_column(
            'embeddings', context_embeddings.tolist())

        # Đánh index cho cột embeddings
        ds_with_embeddings.add_faiss_index(column='embeddings')

        # Tính toán độ tương đồng
        scores, retrieved_examples = ds_with_embeddings.get_nearest_examples(
            'embeddings', query_embedding, k=k)

        return retrieved_examples['context']
