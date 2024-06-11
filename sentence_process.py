import TF_IDF_Search


tfidf = TF_IDF_Search.TF_IDF_Init()


def process_query(query):
    words_to_remove = ["là gì", "cái gì", "nghĩa là", "tại sao", "vì sao", "làm sao", "cách", "hỏi", "biết", "giải thích",
                       "khái niệm", "mô tả", "được gọi là", "thuộc về", "ở đâu", "tại đâu", "khi nào", "nếu", "cho", "nhờ", "tìm hiểu",
                       "điều gì", "kể cho", "giúp", "cho biết", "được định nghĩa là", "nên", "có thể", "muốn", "sao", "được xem là",
                       "chú ý", "thế nào", "là cách nào", "đưa ra", "tìm", "phải", "hỏi", "biết", "giúp", "đúng không", "nên",
                       "được xem như", "sao", "rằng", "để làm gì", "làm như thế nào", "có nghĩa là", "có phải là", "thành thật mà nói",
                       "không phải là", "đúng không", "như thế nào", "mục đích của", "để làm gì", "thế nào", "nghĩa là gì", "có nghĩa là gì",
                       "sự khác biệt giữa", "so sánh với", "ví dụ về", "cách để", "phần nào", "có ý nghĩa là", "kỹ thuật làm thế nào",
                       "cách mà", "không biết", "phát hiện ra", "có thể biết", "phân biệt giữa", "mục đích của", "đặc điểm chính của",
                       "điều quan trọng nhất là", "mô tả của", "được sử dụng để", "phương pháp nào", "một số", "giải thích là",
                       "tại sao lại", "làm thế nào để", "khi nào thì", "để làm", "tại sao lại không", "tại sao lại cần", "là gì vậy",
                       "làm như thế nào để", "như thế nào là", "muốn biết", "khám phá", "có nghĩa là gì", "chúng ta có thể",
                       "cách mà chúng ta có thể", "vấn đề là gì", "nắm bắt được", "sự khác biệt là gì", "cần phải làm gì", "tìm hiểu về",
                       "tại sao không nên", "có ý nghĩa gì", "cần biết về", "được xác định là", "tìm hiểu thêm về", "tác động của", "hiểu rõ",
                       "tính chất của"]

    query = tfidf.preprocessing([query])[0]
    for word in words_to_remove:
        query = query.replace(word, '')
    return query.strip()
