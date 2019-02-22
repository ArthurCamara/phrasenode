import json
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from re import finditer
import gzip as gz

def camel_case_and_tokenizer_split(identifier):
    matches = finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    partial = " ".join([m.group(0) for m in matches])
    tokenizer = RegexpTokenizer(r'\w+')
    return " ".join(tokenizer.tokenize(partial))

def generate_document(page):
    #step 1- tokenize and stem text
    final_doc = ""
    tokenizer = RegexpTokenizer(r'\w+')
    ps = SnowballStemmer('english')
    if 'text' in page:
        final_doc += " ".join([ps.stem(x) for x in tokenizer.tokenize(page['text'])])
    
    #step 2 - tokenize atributes
    if 'attributes' in page:
        attributes_to_tokenize_and_stem = ['id', 'class', 'placeholder', 'label', 'tooltip', 'aria-text', 'name', 'src', 'href']
        for attr in attributes_to_tokenize_and_stem:
            if attr in page['attributes'] and page['attributes'][attr]:
                clean_attr = camel_case_and_tokenizer_split(page['attributes'][attr])
                final_doc+=" " + clean_attr
    return final_doc.strip()


def get_documents_from_file(raw):
    page = json.load(gz.open(raw))
    clean_docs = []
    for doc in page['info']:
        clean_doc = generate_document(doc) 
        if len(clean_doc) == 0:
            continue
        clean_docs.append(clean_doc)
    return  clean_docs
    