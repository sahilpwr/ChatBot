import nltk
from nltk.corpus import stopwords
import fileinput
import re

f=open("C:\\Users\\Owner\\Desktop\\question.txt",'r+')

sentences=f.readline()

for line in f:
    #tokenize
    nltk.extract_rels
    tokens=nltk.word_tokenize(line)

   # IN=re.compile(r'.*\bin\b(?!\b.+ing)')
    #print(nltk.sem.extract_rels('ORG', 'LOC', tokens,corpus='ieer', pattern = IN))

    #parts of speech tagging
    pos=nltk.pos_tag(tokens)
    print(pos)

    #removing stopwords
    stop_words=set(stopwords.words('english'))
    filtered_tokens=[]
    pos_tag=[]

    WH_Words=[]
    Action_Verb=[]
    Adjective=[]
    Common_Nouns=[]
    Nouns=[]

    for i,j in pos:
        if i not in stop_words:
            if (j=='WRB'or j=='WP'or j=='WP$'or j=='WDT'):
                WH_Words.append(i)
            if (j == 'VB' or j == 'VBD' or j=='VBG' or j=='VBN' or j=='VBP' or j=='VBz'):
                Action_Verb.append(i)
            if (j == 'NNP' or j == 'NNPS'):
                Nouns.append(i)
            if (j == 'NN' or j=='NNS') :
                Common_Nouns.append(i)
            if (j == 'JJ'or j=='JJS' or j=='JJR'):
                Adjective.append(i)
            filtered_tokens.append(i)
            pos_tag.append(j)

    print(WH_Words)
    print(Action_Verb)
    print(Adjective)
    print(Common_Nouns)
    print(Nouns)
   

    #Stemming 
    from nltk.stem import PorterStemmer
    ps=PorterStemmer()
    stemmed_words=[]
    #for w in filtered_tokens:
       # print(ps.stem(w))
