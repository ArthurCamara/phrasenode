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
import multiprocessing
from contextlib import closing

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path',)
parser.add_argument("-t", "--train", action="store_true")
parser.add_argument("-m", "--multiprocess", default="2")

def process_line(line):
    line_data = json.loads(line)
    query = query_cleaner(line_data['phrase'])
    page = line_data["webpage"]
    docs_vectors, _ = TF_IDF_transformer.load_docs_for_page(page)
    query_vector = TF_IDF_transformer.get_query_vector(query)
    similarity_matrix = cosine_similarity(query_vector, docs_vectors)
    best_match = similarity_matrix.argmax()
    target = line_data['xid']
    return target==best_match

if __name__=="__main__":

    args = parser.parse_args()
    TF_IDF_transformer = TF_IDF_Documents(args.path, force_train=args.train)

    matches = [] 
    if args.multiprocess>1:
        lines_to_process = [x for x in open(TF_IDF_transformer.test_path).readlines()]
        with closing(multiprocessing.Pool(args.multiprocess)) as pool:
            matches = pool.map(process_line, lines_to_process)
            pool.terminate()
    else:
        for line in tqdm(open(TF_IDF_transformer.test_path).readlines()):
            matches.append(process_line(line))
            if len(matches) % 100 ==0:
                print("current accuracy {}".format((sum(matches)*1.0)/len(matches)))
    print("current accuracy {}".format((sum(matches)*1.0)/len(matches)))
    