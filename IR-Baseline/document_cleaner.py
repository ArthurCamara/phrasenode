import json
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from re import finditer
import gzip as gz
import re

def camel_case_and_tokenizer_split(identifier):
    matches = finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    partial = " ".join([m.group(0) for m in matches])
    tokenizer = RegexpTokenizer(r'\w+')
    return " ".join(tokenizer.tokenize(partial))

TOKENIZER = re.compile(r'[^\W_]+|[^\w\s-]', re.UNICODE | re.MULTILINE | re.DOTALL)

def word_tokenize(text):
    """Tokenize without keeping the mapping to the original string.

    Args:
        text (str or unicode)
    Return:
        list[unicode]
    """
    return TOKENIZER.findall(text)


TOKENIZER2 = re.compile(r"[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\'\w]+", re.UNICODE | re.MULTILINE | re.DOTALL)

# courtesy https://stackoverflow.com/questions/6202549/word-tokenization-using-python-regular-expressions
def word_tokenize2(text):
    """Tokenize without keeping the mapping to the original string.
    Removes punctuation, keeps dashes, and splits on capital letters correctly.
    Returns tokenized words in lower case.
    E.g.
    Jeff's dog is un-American SomeTimes! BUT NOTAlways
    ['jeff's', 'dog', 'is', 'un', 'american', 'some', 'times', 'but', 'not', 'always']


    Args:
        text (str or unicode)
    Return:
        list[unicode]
    """
    return [s.lower() for s in TOKENIZER2.findall(text)]


def generate_document(page):
    #step 1- tokenize and stem text
    final_doc = ""
    tokenizer = RegexpTokenizer(r'\w+')
    ps = SnowballStemmer('english')
    if 'text' in page:
        final_doc += " ".join([ps.stem(x) for x in word_tokenize2(page['text'])])
        # final_doc += " ".join([ps.stem(x) for x in word_tokenize2(page['text']] ))
    
    #step 2 - tokenize atributes
    #these will be downweighted by alpha later. Thus, they need to be returned.
    attributes_to_donweight = set()
    if 'attributes' in page:
        clean_attr = u""
        attributes_to_tokenize_and_stem = ['id', 'class', 'placeholder', 'label', 'tooltip', 'aria-text', 'name', 'src', 'href']
        for attr in attributes_to_tokenize_and_stem:
            if attr in page['attributes'] and page['attributes'][attr]:
                clean_attr += " ".join(word_tokenize(page['attributes'][attr]))
                attributes_to_donweight.update(set(clean_attr.split()))
        final_doc+="-" + clean_attr
    return (final_doc.strip(), attributes_to_donweight)


def get_documents_from_file(raw):
    page = json.load(gz.open(raw))
    clean_docs = []
    attributes_to_donweight = set()
    for doc in page['info']:
        clean_doc, attr = generate_document(doc) 
        attributes_to_donweight.update(attr)
        if len(clean_doc) == 0:
            continue
        clean_docs.append(clean_doc)
    return  clean_docs, attributes_to_donweight
    
def query_cleaner(query):
    final_query = []
    tokenizer = RegexpTokenizer(r'\w+')
    ps = SnowballStemmer('english')
    final_query = " ".join([ps.stem(x) for x in word_tokenize2(query)])
    return final_query
