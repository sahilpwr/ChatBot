import os
import logging
import tempfile
import nltk
import sys
import time
from gensim import corpora, models, similarities
from six import iteritems
from nltk.corpus import stopwords
from nltk.tokenize import *
import os
import logging
import tempfile
import nltk
import sys
import time
from gensim import corpora, models, similarities
from six import iteritems
from nltk.corpus import stopwords
from nltk.tokenize import *

def create_dict_and_corpus (q_text, a_text):
    #usage: "questions.txt", "answers.txt", "dictionary.dict", "corpus.mm"
    #Create dictionary
    nltk.extract_rels
    lemma = nltk.wordnet.WordNetLemmatizer()
    stoplist = set(stopwords.words('english'))
    #TAG NOT COMPLETE!!!!!
    tags=['WRB', 'WP', 'WP$', 'WPT', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'NNP', 'NNPS', 'NN', 'NNS', 'JJ', 'JJS', 'JJ']
    tokenizer = RegexpTokenizer(r'\w+')
    text_cleaned = []
    for q_line, a_line in zip(open(q_text), open(a_text)):
        line_cleaned = []
        tokens = tokenizer.tokenize(q_line.lower()+a_line.lower())
        pos = nltk.pos_tag(tokens)
        for i, j in pos:
            if (i not in stoplist):
                if (j in tags):
                    line_cleaned.append(lemma.lemmatize(i))
                    line_cleaned.append(i)
        text_cleaned.append(line_cleaned)
    dictionary=corpora.Dictionary(text_cleaned)
    dictionary.compactify()
    dictionary.save(os.path.join(TEMP_FOLDER, 'dictionary.dict'))
    #Create corpus
    class Corpus(object):
        def __iter__(self):
            for q_line, a_line in zip(open(q_text), open(a_text)):
                line = q_line.lower().split()+a_line.lower().split()
                yield dictionary.doc2bow(line)
    corpus = Corpus()
    corpora.MmCorpus.serialize(os.path.join(TEMP_FOLDER, 'corpus.mm'), corpus)
    corpus = corpora.MmCorpus(os.path.join(TEMP_FOLDER, 'corpus.mm'))
    #Train tf-idf model and lsi model
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=5)
    lsi.save(os.path.join(TEMP_FOLDER,'model.lsi')) # 
    #Create similarities matrix
    corpus_lsi = lsi[corpus_tfidf]
    index = similarities.MatrixSimilarity(lsi[corpus])
    index.save(os.path.join(TEMP_FOLDER, 'index.index'))
def answer_to_user_query(user_query, dictionary, lsi, index, answer_lines):
    #user_query=input("How can I help you? ")
    lemma = nltk.wordnet.WordNetLemmatizer()
    tokenizer = RegexpTokenizer(r'\w+')
    #Process query
    tokens = [lemma.lemmatize(word) for word in tokenizer.tokenize(user_query)]
    vec_bow=dictionary.doc2bow(tokens)
    vec_lsi=lsi[vec_bow]
    sims = index[vec_lsi]
    sims = sorted(enumerate(sims), key=lambda item:-item[1])
    match=sims[0][0]
    answer = nltk.sent_tokenize(answer_lines[match])
    if (len(answer)<3):
        for line in answer:
            print(line)
            time.sleep(len(line)*0.04)
    else:
        print(answer[0]+'...')
        time.sleep(len(answer[0])*0.04)
        user_response=input("Is this answer relevant?(yes/no)")
        if (user_response=='yes'):
            time.sleep(2)
            print('Great! Here is the rest of the answer:')
            time.sleep(2)
            for i in range(1, len(answer)):
                print(answer[i])
                time.sleep(len(answer[i])*0.04)
        else:
            print('Sorry. You will need to ask someone else')

def softwar(question):

    q_text='software.txt'
    a_text='answers.txt'
    

    dictionary = corpora.Dictionary.load(os.path.join(TEMP_FOLDER, 'dictionary.dict'))
    lsi = models.LsiModel.load(os.path.join(TEMP_FOLDER,'model.lsi'))
    index=similarities.MatrixSimilarity.load(os.path.join(TEMP_FOLDER, 'index.index'))
    answer_lines=[line for line in open(a_text)]
    answer_to_user_query(question, dictionary, lsi, index, answer_lines)


