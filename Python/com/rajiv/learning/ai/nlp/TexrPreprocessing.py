import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

paragraph = """
Albert Einstein (/ˈaɪnstaɪn/ EYEN-styne;[4] German: [ˈalbɛɐt ˈʔaɪnʃtaɪn] ⓘ; 14 March 1879 – 18 April 1955) was a German-born theoretical physicist who is widely held to be one of the greatest and most influential scientists of all time. Best known for developing the theory of relativity, Einstein also made important contributions to quantum mechanics, and was thus a central figure in the revolutionary reshaping of the scientific understanding of nature that modern physics accomplished in the first decades of the twentieth century.[1][5] His mass–energy equivalence formula E = mc2, which arises from relativity theory, has been called "the world's most famous equation".[6] He received the 1921 Nobel Prize in Physics "for his services to theoretical physics, and especially for his discovery of the law of the photoelectric effect",[7] a pivotal step in the development of quantum theory. His work is also known for its influence on the philosophy of science.[8][9]

Born in the German Empire, Einstein moved to Switzerland in 1895, forsaking his German citizenship (as a subject of the Kingdom of Württemberg)[note 1] the following year. In 1897, at the age of seventeen, he enrolled in the mathematics and physics teaching diploma program at the Swiss federal polytechnic school in Zürich, graduating in 1900. In 1901, he acquired Swiss citizenship, which he kept for the rest of his life. In 1903, he secured a permanent position at the Swiss Patent Office in Bern. In 1905, he submitted a successful PhD dissertation to the University of Zurich. In 1914, he moved to Berlin in order to join the Prussian Academy of Sciences and the Humboldt University of Berlin. In 1917, he became director of the Kaiser Wilhelm Institute for Physics; he also became a German citizen again, this time as a subject of the Kingdom of Prussia.[note 1]
"""

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

sentences = nltk.sent_tokenize(paragraph)

lemmatizer = WordNetLemmatizer()

corpus = []
##Clean the words by removing special characters
for i in range(len(sentences)):
    corpus.append(re.sub('[^a-zA-Z\\s]', '', sentences[i]).lower())

##Remove stop words from the corpus
stopWordSet = set(stopwords.words('english'))

lemmetaizedCorpus = []

for sen in corpus:
    words = nltk.word_tokenize(sen)
    spaceString = " "
    strArray = []
    for word in words:
        if word not in stopWordSet:
            strArray.append(lemmatizer.lemmatize(word))
    lemmetaizedCorpus.append(spaceString.join(strArray))

print(lemmetaizedCorpus)

##Bag of words
cv = CountVectorizer(ngram_range=(3,3))
trans = cv.fit_transform(lemmetaizedCorpus)
print(cv.vocabulary_)
print(trans[0].toarray())

#TF-IDF
tfidfCV = TfidfVectorizer(ngram_range=(1,5), max_features=5)
tfidfTrans = tfidfCV.fit_transform(lemmetaizedCorpus)
print(tfidfTrans[0].toarray())




