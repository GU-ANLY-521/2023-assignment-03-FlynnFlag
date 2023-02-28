import xml.etree.ElementTree as ET
import abc
from nltk.tokenize import word_tokenize
import string
from collections import Counter


# Problem2-A
tree=ET.parse("20newsgroups-initial.xml")
docid=[]
xml=[]
root=tree.getroot()
for child in root:
    # document the docid
    docid.append(child[0].text)

    # document the content
    xml.append(child[1].text)

# Problem2-B
# create the abc
class tokenizer(abc.ABC):
    @abc.abstractmethod
    def tokenize(self,text):
        pass

# whitespace tokenizer
class tokenizer1(tokenizer):    
    def tokenize(self,text):
        return text.split()

# off-the-shelf tokenizer
class tokenizer2(tokenizer):
    def tokenize(self,text):
        return word_tokenize(text)


# n-grams tokenizer
class tokenizer3(tokenizer):
    def __init__(self, n):
        self.n=n
           
    def tokenize(self,text):
        tokens=text.split()
        grams=[]
        for i in range(len(tokens)-self.n+1):
            grams.append(" ".join(tokens[i:i+self.n]))
        return grams

# use the function and normalize the words
def tokenize(sentence):
    tokenizer_string=tokenizer1().tokenize(sentence)
    res=[]
    for i in tokenizer_string:
        a=i.lower().strip(string.punctuation)
        res.append(a)
    return res


# Problem2-C
# apply porterstemmer
def stem(list):
    from nltk.stem.porter import PorterStemmer
    stemmer=PorterStemmer()
    res=[]
    for i in list:
        res.append(stemmer.stem(i))
    return res



# Problem2-D
index = dict()
# get the index and the xml
for n,text_body in enumerate(xml): # for the raw text you’re pulling out of the xml
    #tokenizee the tokens
    list_of_tokens = tokenize(text_body)
    
    # stem the tokens
    list_of_stemmed_tokens = stem(list_of_tokens)

    # count the term_frequency
    term_frequency = Counter(list_of_stemmed_tokens)

    # document the result
    for token in term_frequency:
        if token in index:
            index[token] += [docid[n], term_frequency[token]]
        else:
            index[token] = [docid[n], term_frequency[token]]

# Problem2-E
print(index["system"])
print("docid ['3','5653','1127','3319','8935','2552','16515','14398','16638','8772'] are returned for the query “system”")

# Problem2-F
# since "compatibility" has been stemmed to "compat", we query the "compat"
print(index["compat"])
print("docid ['14398','8772'] are returned for the query “compatibility”")

#Probme2-E
# find the intersection of two results
print((set(index["system"])).intersection(set(index["compat"])))
print("docid ['14398','8772'] are returned for the query terms “system” and “compatibility”")
