import pandas as pd

# pandas is a python library built for
# data analysis and manipulation

df = pd.read_csv("movie_data.csv")
df.head()  # This lets us see top 5 rows in the df

print(df.iloc[1], "\n")  # iloc returns number of rows

exampleReview = df.iloc[1]["review"]
print(exampleReview, "\n")

print(df.iloc[1]["sentiment"], "\n")

import re

"""
Expression  Description
.           A dot is a wildcard that finds any character except a newline.
[FGz]       This looks for any character in the square bracket. Here F, G, z.
[a-z]       This looks for a range of characters. Here lowercase a to z. 
\w          This looks for any character or underscore for example [A-Za-z0-9_]
\d          This looks for any digit
\s          a space
\t          Tab character
\n          Newline character
\r          Carriage return character. (like hitting enter to form a newline)
^           Match at the start of a string.
$           Match at the end of the string.
?           Get the preceding match zero or one time.
*           Get the preceding match zero or more times.
+           Get the preceding match one or more times.
{m}         Get the preceding match exactly m times.
{m, n}      Get the preceding match between m and n times.
a|b         The pipe denotes alternation. Find either a or b.
()          Create and report a capture group or set precedence.
(?:)        Negate the reporting of a capture group
"""


def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)  # remove all the HTML tags
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)  # remove all the useless punctuation
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')  # this line does actual removing
    return text


exampleReview = preprocessor(
    df.iloc[1]["review"])  # This applies or preprocessor clean up method to the first row of the dataset
print(exampleReview, "\n")

import nltk

"""tokenization basically refers to 
splitting up a larger body of text into 
smaller lines, words or even creating words 
for a non-English language."""


def tokenizer(text):
    return text.split()  # literally finds every whitespace (" ") and adds it into an array


exampleReview = tokenizer(exampleReview)
print(exampleReview, "\n")

# These two things do the same thing

# exampleReview = nltk.word_tokenize(exampleReview)
# print(exampleReview, "\n")


"""Word stemming is the process of transforming a word into its root form. It allows us to map related words to the same stem. The original stemming algorithm was developed by
Martin F. Porter in 1979 and is known as the Porter stemmer algorithm."""
from nltk.stem.porter import PorterStemmer

porter = PorterStemmer()


def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]


testPorter = preprocessor(df.iloc[1]["review"])
print(tokenizer_porter(testPorter), "\n")
# Notice we are getting kind of weird results because the porter doesnt know what is a verb and what
# isn't, so we'll skip this step for now
"""Stemming can create non-real words, such as 'thu' from 'thus'. A technique called lemmaitization aims to obtain the grammatically correct forms of individual words. These are
called the lemmas. Lemmatization is computationally more difficult and expensive compared to stemming and in the end, the two have been observed to have similar impact on text
classification."""

"""Stop-words are words that are extremely common in all sorts of texts and probably hold no useful information that can be used to distinguish between different classes of
documents."""
from nltk.corpus import stopwords

nltk.download('stopwords')
stop = stopwords.words('english')  # download and store the common stopwords
print(stop, "\n")


def removeStopWords(text):
    new_sentence_list = []
    for w in text:
        if w not in stop:
            new_sentence_list.append(w)
    return exampleReview

exampleReview = removeStopWords(exampleReview)
print(exampleReview, "\n")

# Lets apply the preprocessing function to all the reviews
df["review"] = df["review"].apply(preprocessor)
print(df.iloc[4]["review"], "\n")

# The stopwords, tokenizer, and tokenizer_porter function will be applied later in the next section





# TODO:
# 1. CountVectorizer
# 2. TFIDTransformer
# 3. HashingVectorizer

# Attempt 1: Count Vectorizer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

count = CountVectorizer()
docs = np.array([
        'The sun is shining',
        'The weather is sweet',
        'The sun is shining, the weather is sweet, and one and one is two'])
bag = count.fit_transform(docs)
print(count.vocabulary_)    # The vocab is stored in a Python dictionary that maps unique words to integer indices alphabetically. 'and' is 0 and 'weather' is 8
print(bag.toarray())
# Notice: The array is a 3 by 9 (3x9) 2D array.
# The rows (ex. [0 1 0 1 1 0 1 0 0]) represent the count vector of the first sentence "The sun is shining"
# The columns represent the associated count of each word in the sentence

# Example: First row is [0 1 0 1 1 0 1 0 0] =  "The sun is shining"
# Notice the first element (at position 0) corresponds to the word 'and' in the dictionary above
# The second element (at position 1) which is a 1 corresponds to 'is'
# and notice the word 'is' occurs once in the first sentence.

# Note: Downsides to this method is the entire vocabulary along with every single sentence
# must be stored and calculated which is very computationaly costly


# Attempt 2: TfidfTransformer

# Notice in the CountVectorizer we had the word "and" occurring twice in the third sentence.
# The word 'and' doesnt help us determine if the movie is positive or negative so
# we can try TfidTransformer to downweight these frequently occurring words
# When text is raw term frequencies like from the CountVectorizer class we can use the TfidfTransformer
# to add less weight into the frequently occurring words
print()
from sklearn.feature_extraction.text import TfidfTransformer

tfidf = TfidfTransformer(use_idf=True,
                         norm='l2',
                         smooth_idf=True)

np.set_printoptions(precision=2)
print(tfidf.fit_transform(count.fit_transform(docs))
      .toarray())

# Notice kind of similiar to the countvectorizer but weighed values
# Notice: the word "and" in sentence 3 used to be of value 2, however because it occurs a lot
# in multiple sentences, it is unlikely to be useful so the new value is 0.5 (lower than 2)

# However, same downside to the CountVectorizer is all the vocabulary must be stored


print()


# Attempt 3: HashingVectorizer

from sklearn.feature_extraction.text import HashingVectorizer

vectorizer = HashingVectorizer(n_features=2**4)
X = vectorizer.fit_transform(docs)

# Is an efficient way of mapping terms to features. Similiar to CountVectorizer in that it returns
# a normalized count vector; however, the way it stores the vector is much more efficient.

# Cons: a little less interpretable (harder to tell which words responds to what element). However,
# luckely we dont really need to know

print(X)