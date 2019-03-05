# -*- coding:utf-8 -*-
import os
import argparse
import json
from  document_cleaner import query_cleaner
from collections import Counter
import pickle
from sklearn.feature_extraction.text import TfidfTransformer
from tf_idf_documents import TF_IDF_Documents
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path',)
parser.add_argument("-t", "--train", action="store_true")

if __name__=="__main__":

    #First, calculate IDFs for trainning datasets
    args = parser.parse_args()
    TF_IDF_transformer = TF_IDF_Documents(args.path, force_train=args.train)

    # For each document in the test dataset, get the TF-IDF representation of the query and the documents

    matches = []    
    for line in tqdm(open(TF_IDF_transformer.test_path).readlines()):
        line_data = json.loads(line)
        query = query_cleaner(line_data['phrase'])
        page = line_data["webpage"]
        docs_vectors, page_docs = TF_IDF_transformer.load_docs_for_page(page)
        query_vector = TF_IDF_transformer.get_query_vector(query)
        similarity_matrix = cosine_similarity(query_vector, docs_vectors)
        best_match = similarity_matrix.argmax()
        target = line_data['xid']
        matches.append(target==best_match)
        if len(matches) % 100 ==0:
            print("current accuracy {}".format((sum(matches)*1.0)/len(matches)))
    print("current accuracy {}".format((sum(matches)*1.0)/len(matches)))
    
    # #for each query in the TEST dataset, go to the target page, compute TF-IDF for each document, store the document on dictionary
    # for counter, line in enumerate(open(train_queries_path)):
    #     if counter > 100:
    #         break
    # # documenst_path = os.path.join(args.path, "phrase-node-dataset","infos", "v6")