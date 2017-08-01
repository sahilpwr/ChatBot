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
    for line in open ('C:\\Users\\Owner\\Downloads\\ChatBot\\ChatBot-master\\alpha\\textfiles\\answers.txt'):
        answer_lines.append(line)



    #Create dictionary
    nltk.extract_rels
    lemma = nltk.wordnet.WordNetLemmatizer()
    stoplist = set(stopwords.words('english'))
    #TAG NOT COMPLETE!!!!!
    tags=['WRB', 'WP', 'WP$', 'WPT', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'NNP', 'NNPS', 'NN', 'NNS', 'JJ', 'JJS', 'JJ']
    tokenizer = RegexpTokenizer(r'\w+')
    text_cleaned = []
    for q_line, a_line in zip(open('C:\\Users\\Owner\\Downloads\\ChatBot\\ChatBot-master\\alpha\\textfiles\\software.txt'), open('C:\\Users\\Owner\\Downloads\\ChatBot\\ChatBot-master\\alpha\\textfiles\\answers.txt')):
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
            for q_line, a_line in zip(open('C:\\Users\\Owner\\Downloads\\ChatBot\\ChatBot-master\\alpha\\textfiles\\software.txt'), open('C:\\Users\\Owner\\Downloads\\ChatBot\\ChatBot-master\\alpha\\textfiles\\answers.txt')):
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

def KAMPS(question):
            
            from math import log
            from math import sqrt
            from collections import Counter
            from operator import itemgetter
            import nltk
            from nltk.corpus import stopwords
            import fileinput
            import nltk
            import re
            print(question)
    #matching question to a category
            def idf(term, allDocuments):
                numDocumentsWithThisTerm = 0
                for cnt in allDocuments:
                    if term in cnt:
                        numDocumentsWithThisTerm = numDocumentsWithThisTerm + 1
                if numDocumentsWithThisTerm > 0:
                    return round(log(float(float(len(allDocuments))/float(numDocumentsWithThisTerm)),2),3)
                else:
                    return 0

            def tf(term, document):
                return document.count(term)


            def caltfidf(term,doc):
                return tf(term,doc)*idf(term,terms)


            def cosineSimilarity(doc,question,d):
                a=0
                for x in d:
                    a=a+caltfidf(x,doc)*caltfidf(x,question)
                b=lengthof(doc,d)*lengthof(question,d)
                if not b:
                    return 0
                else:
                    return round(a/b,3)
            def lengthof(doc,d):
                val=0
                for x in d:
                    val=val+pow(caltfidf(x,doc),2)
                return sqrt(val)


            f=[]
            doc=['C:\\Users\\Owner\\Downloads\\ChatBot\\ChatBot-master\\alpha\\textfiles\\person.txt','C:\\Users\\Owner\\Downloads\\ChatBot\\ChatBot-master\\alpha\\textfiles\\team.txt','C:\\Users\\Owner\\Downloads\\ChatBot\\ChatBot-master\\alpha\\textfiles\\topics.txt','C:\\Users\\Owner\\Downloads\\ChatBot\\ChatBot-master\\alpha\\textfiles\\software.txt']
            dataset=[['C:\\Users\\Owner\\Downloads\\ChatBot\\ChatBot-master\\alpha\\person.txt',"people-team-location"],['C:\\Users\\Owner\\Downloads\\ChatBot\\ChatBot-master\\alpha\\textfiles\\team.txt',"team-location"],['C:\\Users\\Owner\\Downloads\\ChatBot\\ChatBot-master\\alpha\\textfiles\\topics.txt',"topic-team"],['C:\\Users\\Owner\\Downloads\\ChatBot\\ChatBot-master\\alpha\\textfiles\\software.txt',"software-team"]]
            for x in doc:
                f.append(open(x,'r').read())

            


            terms=[]
            for x in f:
                terms.append(x.rstrip('\n'))


            fin_terms=[]
            for x in terms:
                fin_terms=fin_terms+x.split()
            fin_terms=set(fin_terms)
            fin_terms=list(fin_terms)


            print( '\nCosine Similarity Values')

            cnt=0
            for x in terms:
                dataset[cnt]=dataset[cnt]+[cosineSimilarity(question,x,fin_terms)]
                cnt=cnt+1
            print( dataset)


            k=1

            sorted_dataset=sorted(dataset,key=itemgetter(2),reverse=True)
            top_k=sorted_dataset[:k]
            top_k[:] = (x for x in top_k if x[2] != 0)

            if len(top_k)== 0:
                print(classification)
                return 'Didn\'t understand your question. Can you ask questions related to team,people or location?'
            else:
                class_counts=Counter(category for (document,category,value) in top_k)
                print(class_counts)
	            #match class to the class which is max in top k
                classification=max(class_counts,key=lambda cls:class_counts[cls])
                print( 'Class of test file is : ',classification)

            classification_list=classification.split("-")
            print(classification.split("-"))


            if(classification_list[0]=='software'):
                software_answer=software(question)
                return software_answer


            #creating a queue containing traversal route
            queue=[]

            map={"people":0,"team":1,"location":2,"topic":3}
            for temp in classification_list:
                queue.append(map[temp])

            print(queue)
            #identification of names, location, team and topics
            Names=[]
            Value=[]


            def teamExtraction(question):
                words = nltk.word_tokenize(question)
                tagged = nltk.pos_tag(words)


                chunkGram = r"""Chunk: {<NNP.?>*<JJ.?>*}"""
                chunkParser = nltk.RegexpParser(chunkGram)
                chunked = chunkParser.parse(tagged)
                print(chunked)

                count=0
                chunked_sentence=[]
                for subtree in chunked.subtrees(filter=lambda t: t.label() == 'Chunk'):
                 for leave in subtree.leaves():
                        chunked_sentence.append(leave)
                        count=count+1
    
                a=chunked_sentence[0]
                teamName=a[0]
                for var in range(count-1):
                    a=chunked_sentence[var+1]
                    teamName=teamName+' '+a[0]
                print(teamName)
                return teamName
        


            def namesExtraction(qustion):
                for sent in nltk.sent_tokenize(question):
                    for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
                        if hasattr(chunk, 'label'):
                            Value.append(' '.join(c[0] for c in chunk.leaves()))
 

            def nounExtraction(question):
                nltk.extract_rels
                tokens=nltk.word_tokenize(question)

                pos=nltk.pos_tag(tokens)
                stop_words=set(stopwords.words('english'))
                filtered_tokens=[]
                pos_tag=[]

                Adjective=[]
                Common_Nouns=[]
                Nouns=[]

                for i,j in pos:
                        if i not in stop_words:
                            if (j == 'NNP' or j == 'NNPS'):
                                Value.append(i)
                
                            if (j == 'JJ'or j=='JJS' or j=='JJR'):
                                Value.append(i)
                            filtered_tokens.append(i)


            if(classification_list[0]=='people'):
                namesExtraction(question)
            elif(classification_list[0]=='team'):
                Value.append(teamExtraction(question))
   
            else:
                nounExtraction(question)

            

            import re


            from math import log
            from math import sqrt
            from collections import Counter
            from operator import itemgetter

            class Vertex:
                def __init__(self,key,hash):
                    self.id = key
                    self.hashTable=[]
                    self.hashTable=hash
                    self.connectedTo = {} 
                    print(self.hashTable)

                def addNeighbor(self,nbr,weight):
                    self.connectedTo[nbr] = weight

                def getConnections(self):
                    return self.connectedTo.keys()

                def getId(self):
                    return self.id

            class Graph:
                def __init__(self):
                    self.vertList = {}
                    self.hashTable=[]
                    self.numVertices = 0

                def addVertex(self,key,hashTable):
                    self.numVertices = self.numVertices + 1
                    newVertex = Vertex(key,hashTable)
                    self.vertList[key] = newVertex

                def getVertex(self,n):
                    if n in self.vertList:
                        return self.vertList[n]
                    else:
                        return None
                def returnList(self):
                    return self.vertList

                def __contains__(self,n):
                    return n in self.vertList

                def addEdge(self,f,t,cost):
                    self.vertList[f].addNeighbor(self.vertList[t], cost)

                def getVertices(self):
                    return self.vertList.keys()

                def __iter__(self):
                    return iter(self.vertList.values())

                def getWeight(self,nbr):
                    return self.connectedTo[nbr]

            g = Graph()

            file1=open("C:\\Users\\Owner\\Downloads\\ChatBot\\ChatBot-master\\alpha\\textfiles\\people_db.txt",'r+')
            file2=open("C:\\Users\\Owner\\Downloads\\ChatBot\\ChatBot-master\\alpha\\textfiles\\team_db.txt",'r+')
            file3=open("C:\\Users\\Owner\\Downloads\\ChatBot\\ChatBot-master\\alpha\\textfiles\\location_db.txt",'r+')
            file4=open("C:\\Users\\Owner\\Downloads\\ChatBot\\ChatBot-master\\alpha\\textfiles\\topics_db.txt",'r+')

            person=[]
            team=[]
            location=[]
            topics=[]


            person=re.split(',|\n',file1.read())
            team=re.split(',|\n',file2.read())
            location=re.split(';|\n',file3.read())
            topics=re.split(',|\n',file4.read())

            g.addVertex(0,person)
            g.addVertex(1,team)
            g.addVertex(2,location)
            g.addVertex(3,topics)

            g.addEdge(0,1,1)
            g.addEdge(1,0,1)
            g.addEdge(1,2,1)
            g.addEdge(2,1,1)
            g.addEdge(1,3,1)
            g.addEdge(3,1,1)


            dict={}
            dict=g.returnList()

            for v in g:
                for w in v.getConnections():
                    print("( %s , %s )" % (v.getId(), w.getId()))


            answer=[]
            for nodeValue in Value:
                    anotherValue=nodeValue
                    i=0
                    for j in range(len(queue)):
                        #print('j',j)
                        if(i<len(queue)-1):
                            #print('i',i)
                            start=queue[i]
                            end=queue[i+1]
                            #print(start,end)
                            if(dict[start].connectedTo[dict[end]]==1):
                                 if(not(nodeValue.isdigit()) and (nodeValue in dict[start].hashTable or anotherValue in dict[start].hashTable)):
                                    if nodeValue in dict[start].hashTable:
                                        index=dict[start].hashTable.index(nodeValue)
                                    else:
                                        index=dict[start].hashTable.index(anotherValue)
                                        if(anotherValue.isdigit()):
                                            key=dict[start].hashTable[index+1]
                                            answer.append(key)
                                        else:
                                            key=dict[start].hashTable[index-1]
                                            answer.append(key)
                        
                                        break
                                    key=dict[start].hashTable[index-1]
                                    anotherValue=nodeValue
                                    nodeValue=key
                                    answer.append(key)

                                 elif(nodeValue.isdigit() and (nodeValue in dict[start].hashTable or anotherValue in dict[start].hashTable)):
                                    if nodeValue in dict[start].hashTable:
                                     index=dict[start].hashTable.index(nodeValue)
                                    else:
                                        index=dict[start].hashTable.index(anotherValue)
                                        if(anotherValue.isdigit()):
                                             key=dict[start].hashTable[index+1]
                                             answer.append(key)
                                        else:
                                            key=dict[start].hashTable[index-1]
                                            answer.append(key)
                                        break
                                    key=dict[start].hashTable[index+1]
                                    anotherValue=nodeValue
                                    nodeValue=key
                                    answer.append(key)

        
                        else:
                            start=end
                            #print(start,end)
                            if(not(nodeValue.isdigit()) and (nodeValue in dict[start].hashTable or anotherValue in dict[start].hashTable)):
                               if nodeValue in dict[start].hashTable:
                                index=dict[start].hashTable.index(nodeValue)
                               else:
                                index=dict[start].hashTable.index(anotherValue)
                                if(anotherValue.isdigit()):
                                    key=dict[start].hashTable[index+1]
                                    answer.append(key)
                                else:
                                    key=dict[start].hashTable[index-1]
                                    answer.append(key)
                                break
                               #print(index)
                               key=dict[start].hashTable[index-1]
                               anotherValue=nodeValue
                               nodeValue=key
                               answer.append(key)
                            elif(nodeValue.isdigit() and (nodeValue in dict[start].hashTable or anotherValue in dict[start].hashTable)):
                                if nodeValue in dict[start].hashTable:
                                 index=dict[start].hashTable.index(nodeValue)
                                else:
                                 index=dict[start].hashTable.index(anotherValue)
                                 if(anotherValue.isdigit()):
                                    key=dict[start].hashTable[index+1]
                                    answer.append(key)
                                 else:
                                    key=dict[start].hashTable[index-1]
                                    answer.append(key)
                                 break
                                key=dict[start].hashTable[index+1]
                                anotherValue=nodeValue
                                nodeValue=key
                                answer.append(key)

                        i=i+1


            template={'people-team-location':'Name who works with team sits on location','team-location':'team sits on location. Can I help anything else with you?','topic-team':'Well, team works mainly on topic'}
            print(answer)
            print(Value)
            finalAnswer=''
            print('Count',len(answer))
            if(len(answer)!=0):

                

                if(classification=='people-team-location'):
                        finalAnswer=template[classification]
                        finalAnswer=finalAnswer.replace('Name',Value[0])
                        finalAnswer=finalAnswer.replace('team',answer[1])
                        finalAnswer=finalAnswer.replace('location',answer[2])
                elif(classification=='team-location'):
                        finalAnswer=template[classification]
                        finalAnswer=finalAnswer.replace('team',Value[0])
                        finalAnswer=finalAnswer.replace('location',answer[1])
                elif(classification=='topic-team'):
                         finalAnswer=template[classification]
                         finalAnswer=finalAnswer.replace('team',answer[1])
                         finalAnswer=finalAnswer.replace('topic',Value[0])
                print(finalAnswer)
            else:
                finalAnswer= 'Didn\'t understand your question. Can you ask questions related to team,people or location?'
            
            return finalAnswer

