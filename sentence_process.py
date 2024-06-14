import re
import MeCab


def raw_process_query(text):
    # Loại bỏ các ký tự không mong muốn
    # Loại bỏ ký tự đặc biệt nhưng giữ lại các ký tự tiếng Nhật
    text = re.sub(r'[^\w\sぁ-んァ-ン一-龥々ー]', '', text)
    text = re.sub(r'\d+', '', text)  # Loại bỏ số

    # Tách từ sử dụng MeCab
    mecab = MeCab.Tagger('-Owakati')
    tokenized_text = mecab.parse(text).strip()

    return tokenized_text
