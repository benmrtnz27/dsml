import nltk
# Sentences
# nltk.download('punkt')  # Downloads required for some features
sentence_data = "The First sentence is about Python. The Second: about Django. You can learn Python,Django and Data Ananlysis here. "
nltk_tokens = nltk.sent_tokenize(sentence_data)
print(nltk_tokens, "\n")

# Words
word_data = "It originated from the idea that there are readers who prefer learning new skills from the comforts of their drawing rooms"
nltk_tokens = nltk.word_tokenize(word_data)
print(nltk_tokens, "\n")

# Stemming
from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()
def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]

print(tokenizer_porter('Runners like running and so they run'), "\n")

# Stop Words
nltk.download('stopwords')
from nltk.corpus import stopwords

stop = stopwords.words('english')
print([w for w in tokenizer_porter('a runner likes running and runs a lot')[-10:] if w not in stop])