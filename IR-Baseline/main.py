# -*- coding:utf-8 -*-
import os
import argparse
import json
from  document_cleaner import get_documents_from_file
from collections import Counter
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path',)

if __name__=="__main__":

    #First, calculate IDFs for trainning datasets
    #For each document in the test dataset, get the TF-IDF representation of the query and the 
    args = parser.parse_args()
    train_queries_path = os.path.join(args.path, "phrase-node-dataset","data","combined-v2-cleaned.train.jsonl")
    IDF_counter = Counter()
    processed_urls = set()
    documents = dict()
    need_IDF = True
    if not os.path.exists("IDF_Counter.pkl"):
        need_IDF = False

    for counter, line in enumerate(open(train_queries_path)):
        if len(documents) >100:
            break
        line_data = json.loads(line)
        query = line_data["phrase"]
        document = line_data["webpage"]
        target = line_data['equiv']
        if document in processed_urls:
            continue
        processed_urls.add(document)
        doc_path = os.path.join(args.path, "phrase-node-dataset", "infos", "v6", "info-"+document+".gz")
        page_docs = get_documents_from_file(doc_path)
        documents[document] = page_docs
        if need_IDF:
            for doc in page_docs:
                for term in doc.split(" "):
                    IDF_counter[term]+=1
        #dump IDF counter
    if need_IDF:
        pickle.dump(IDF_counter, open("IDF_Counter.pkl", "wb"))
    else:
        IDF_counter = pickle.load(open("IDF_Counter.pkl"))

    
    # documenst_path = os.path.join(args.path, "phrase-node-dataset","infos", "v6")