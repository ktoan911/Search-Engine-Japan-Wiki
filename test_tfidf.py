import rawSearch

tf_path = r'D:\Python\Search-Engine-Wiki\Json_file\tf_idf_dict.json'
ds_path = r'D:\Python\Search-Engine-Wiki\Json_file\ds.json'
doc_path = r'D:\Python\Search-Engine-Wiki\Json_file\docs.json'

tf = rawSearch.TFIDF(tf_path, ds_path, doc_path)
print(tf.search('Miền bắc', k=1))
