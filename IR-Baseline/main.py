# -*- coding:utf-8 -*-
import os
import argparse
import json
from  document_cleaner import get_documents_from_file
from collections import Counter
import pickle
from sklearn.feature_extraction.text import TfidfTransformer
from tf_idf_documents import TF_IDF_Documents

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path',)
parser.add_argument("-t", "--train", action="store_true")

if __name__=="__main__":

    #First, calculate IDFs for trainning datasets
    args = parser.parse_args()
    TF_IDF_transformer = TF_IDF_Documents(args.path, args.force_train)

    # For each document in the test dataset, get the TF-IDF representation of the query and the 
    
    
    # #for each query in the TEST dataset, go to the target page, compute TF-IDF for each document, store the document on dictionary
    # for counter, line in enumerate(open(train_queries_path)):
    #     if counter > 100:
    #         break
    # # documenst_path = os.path.join(args.path, "phrase-node-dataset","infos", "v6")