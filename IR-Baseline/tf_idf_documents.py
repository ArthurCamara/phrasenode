import os
import json
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from document_cleaner import get_documents_from_file
import pickle

class TF_IDF_Documents:
    def __init__(self, data_home, force_train=False):
        self.alpha = 3
        #alpha is a downweigth for tokens from
        #Check if trained IDF already exists
        self._data_home = data_home
        self._trained = False
        self._train_queries_path = os.path.join(data_home, 
                                                                        "phrase-node-dataset",
                                                                        "data",
                                                                        "combined-v2-cleaned.train.jsonl")

        if  os.path.exists("IDF_Counter.pkl"):
            self._trained = True
            self.load_tf_idf_documents()
            return
        else:
            self.generate_tf_idf_documents()
    
    #Train IDF vectors
    def generate_tf_idf_documents(self):
        all_documents = []
        procesed_urls = set()
        self.all_attributes = set()
        for counter, line in enumerate(open(self._train_queries_path)):
            line_data = json.loads(line)
            query = line_data["phrase"]
            document = line_data["webpage"]
            target = line_data['equiv']
            if document in procesed_urls:
                continue
            procesed_urls.add(document)
            if len(procesed_urls) %100 ==0:
                print(document, len(all_documents))
            doc_path = os.path.join(self._data_home, "phrase-node-dataset",
                                                    "infos", "v6", "info-"+document+".gz") 
            page_docs, attributes = get_documents_from_file(doc_path)
            self.all_attributes.update(attributes)
            all_documents += page_docs
        
        #generate a count vector and train the IDF vectors
        count_vector = CountVectorizer(encoding="utf-8", strip_accents="ascii")
        X = count_vector.fit_transform(all_documents)
        self.transformer = TfidfTransformer()
        self.transformer.fit(X)
        pickle.dump(self.transformer, open("IDF_Counter.pkl", "wb"))
        pickle.dump(self.all_attributes, open("attributes.pkl", "wb"))
        


    def load_tf_idf_documents(self):
        self.all_attributes = pickle.load(open("attributes.pkl", "rb"))
        self.transformer = pickle.load(open("IDF_Counter.pkl", "rb"))