import os
from docx import Document
import os.path
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import spacy
data_path = '/home/user/workspace/data/training'
file_names = [file.split('.')[0] for file in os.listdir(data_path)]
files = [file for file in os.listdir(data_path)]
def load_data():
    
    documents_list = []
    titles=[]
    for f in range(0,len(files)):
        full_text = []
        document = Document(data_path+'/'+files[f])
        for para in document.paragraphs:
            full_text.append(str(para.text.encode('utf-8').strip())[1:])
            full_text = [line for line in full_text if line != "''"]
        documents_list.append(full_text)
    # print("Total Number of Documents:",len(documents_list))
    return documents_list
# print (load_data())

def preprocess_data(doc_set):
    tokenizer = RegexpTokenizer(r'\w+')
    en_stop = set(stopwords.words('english'))
    # p_stemmer = PorterStemmer()
    full_texts = []
    for dl in doc_set:
        texts = []
        for i in dl:
            # raw = i.lower()
            tokens = tokenizer.tokenize(i)
            stopped_tokens = [i for i in tokens if not i in en_stop]
            # stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
            texts.append(' '.join(stopped_tokens))
        full_texts.append(texts)
    return full_texts
# print(preprocess_data(load_data()))
en = spacy.load('en_core_web_md')
# # person , dates = [] , []
for doc in preprocess_data(load_data()):
    p ,d = [], []
    for txt in doc:
        txt = txt.title().strip()
        sents = en(txt)
        people = [ee for ee in sents.ents if ee.label_ == 'PERSON']
        date = [ee for ee in sents.ents if ee.label_ == 'DATE']

    p.append(people)
    d.append(date)
print (p)
print (d)
#     person = [name for name in p]
#     dates = [da for da in d]
# print (len(person))
# print (len(dates))

    