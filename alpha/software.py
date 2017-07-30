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


def software(question):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    TEMP_FOLDER = tempfile.gettempdir()
    print(tempfile)
    #load answers into memory
    answer_lines=[]
    for line in open ('answers.txt'):
        answer_lines.append(line)



    #Create dictionary
    nltk.extract_rels
    lemma = nltk.wordnet.WordNetLemmatizer()
    stoplist = set(stopwords.words('english'))
    #TAG NOT COMPLETE!!!!!
    tags=['WRB', 'WP', 'WP$', 'WPT', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'NNP', 'NNPS', 'NN', 'NNS', 'JJ', 'JJS', 'JJ']
    tokenizer = RegexpTokenizer(r'\w+')
    text_cleaned = []
    for q_line, a_line in zip(open('questions.txt'), open('answers.txt')):
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

    #Load dictionary and corpus
    dictionary = corpora.Dictionary.load(os.path.join(TEMP_FOLDER, 'dictionary.dict'))

    #Create corpus
    class Corpus(object):
        def __iter__(self):
            for q_line, a_line in zip(open('questions.txt'), open('answers.txt')):
                line = q_line.lower().split()+a_line.lower().split()
                yield dictionary.doc2bow(line)
    corpus = Corpus()
    corpora.MmCorpus.serialize(os.path.join(TEMP_FOLDER, 'corpus.mm'), corpus)

    corpus = corpora.MmCorpus(os.path.join(TEMP_FOLDER, 'corpus.mm'))

    #Train tf-idf model and lsi model
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=8)
    corpus_lsi = lsi[corpus_tfidf]
    index = similarities.MatrixSimilarity(lsi[corpus])
    index.save(os.path.join(TEMP_FOLDER, 'index.index'))

    #Continue training with additonal training data
    #lsi.add_documents(another_tfidf_corpus)
    #corpus_lsi= lsi[corpus_tfidf] 

    #Load similarity matrix
    index=similarities.MatrixSimilarity.load(os.path.join(TEMP_FOLDER, 'index.index'))


    #user query and match 
    user_query=question
    tokens = [lemma.lemmatize(word) for word in tokenizer.tokenize(user_query)]
    vec_bow=dictionary.doc2bow(tokens)
    vec_lsi=lsi[vec_bow]
    sims = index[vec_lsi]
    sims = sorted(enumerate(sims), key=lambda item:-item[1])
    match=sims[0][0]
    answer = nltk.sent_tokenize(answer_lines[match])
    if (len(answer)<2):
        for line in answer:
            time.sleep(3)
            print(line)
        return line
    else:
        time.sleep(3)
        print(answer[0])
        time.sleep(3)
        system.stdout.write(answer[1]+'...'+'\n')
        user_response=input("Is this answer relevant?(yes/no)")
        if (user_response=='yes'):
            time.sleep(3)
            print('Great! Here is the rest of the answer:')
            for i in range(2, len(answer)):
                time.sleep(3)
                print(answer[1])
        else:
            print('Sorry. You will need to ask someone else.')
        return answer[1]
    
