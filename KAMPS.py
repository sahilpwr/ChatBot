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
    doc=['person.txt','team.txt','software-support-team.txt','topics.txt']
    dataset=[['person.txt',"people-team-location"],['team.txt',"team-location"],['software-support-team.txt',"support-team-people"],['topics.txt',"topic-team"]]
    for x in doc:
	    f.append(open(x,'r').read())

    terms=[]
    for x in f:
	    terms.append(x.lower().rstrip('\n'))


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
	    print('Does not match')
    else:
	    class_counts=Counter(category for (document,category,value) in top_k)
	    print(class_counts)
	    #match class to the class which is max in top k
	    classification=max(class_counts,key=lambda cls:class_counts[cls])
	    print( 'Class of test file is : ',classification)

    classification_list=classification.split("-")
    print(classification.split("-"))


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

    file1=open("C:\\Users\\Owner\\Desktop\\Chatbot\\people.txt",'r+')
    file2=open("C:\\Users\\Owner\\Desktop\\Chatbot\\team.txt",'r+')
    file3=open("C:\\Users\\Owner\\Desktop\\Chatbot\\location.txt",'r+')
    file4=open("C:\\Users\\Owner\\Desktop\\Chatbot\\topics.txt",'r+')

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


    fifinalAnswer=''
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
    return fifinalAnswer

