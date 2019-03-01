import os
import json
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, TfidfVectorizer
from document_cleaner import get_documents_from_file
import pickle

class TF_IDF_Documents():
    def __init__(self, data_home, force_train=False):
        self.alpha = 3
        #alpha is a downweigth for tokens from
        #Check if trained IDF already exists
        self._data_home = data_home
        self._trained = False
        self.loaded_pages = dict()
        self._train_queries_path = os.path.join(data_home, 
                                                                        "phrase-node-dataset",
                                                                        "data",
                                                                        "combined-v2-cleaned.train.jsonl")

        if  os.path.exists("IDF_Counter.pkl") and os.path.exists("Counter.pkl") and not force_train:
            self._trained = True
            self.load_tf_idf_documents()
            self.reverse_index = {feature: idx for idx, feature in enumerate(self.transformer.get_feature_names())}
            return
        else:
            self.reverse_index = {feature: idx for idx, feature in enumerate(self.transformer.get_feature_names())}
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
            all_documents += [x.replace("-", " ") for x in page_docs]
        
        #generate a count vector and train the IDF vectors
        # self.count_vector = CountVectorizer(encoding="utf-8", strip_accents="ascii")
        # X = self.count_vector.fit_transform(all_documents)
        self.transformer = TfidfVectorizer()
        self.transformer.fit(all_documents)
        pickle.dump(self.transformer, open("IDF_Counter.pkl", "wb"))
        # pickle.dump(self.count_vector, open("Counter.pkl", "wb"))

#Load pre-computed TF_IDF transformer        
    def load_tf_idf_documents(self):
        # self.count_vector = pickle.load(open("Counter.pkl", "rb"))
        self.transformer = pickle.load(open("IDF_Counter.pkl", "rb"))


    def load_docs_for_page(self, page):
        if page in self.loaded_pages:
            return self.loaded_pages[page]
        doc_path = os.path.join(self._data_home, "phrase-node-dataset",
                                                 "infos", "v6", "info-"+page+".gz")
        page_docs, attributes = get_documents_from_file(doc_path)
        clear_page_docs = [x.replace("-", " ") for x in page_docs]
        # count_vectors = self.count_vector.transform(clear_page_docs)
        tf_idfs = self.transformer.transform(clear_page_docs)
        #decrease score by alpha for attributes
        for counter, doc in enumerate(page_docs):
            if "-" not in doc:
                continue
            for attr in attrs_to_demote:
                if len(attr) <1  or attr not in self.reverse_index:
                    continue
                idx = self.reverse_index[attr.lower()]
                tf_idfs[counter, idx] = tf_idfs[counter, idx]/self.alpha
        return tf_idfs


    def get_query_vector(self, query):
        return self.transformer.transform([query]])