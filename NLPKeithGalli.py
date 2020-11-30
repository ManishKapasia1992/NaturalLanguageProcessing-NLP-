import spacy
#
# class Category:
#     BOOKS = 'BOOKS'
#     CLOTHING = 'CLOTHING'
#
# # Word Vector
# nlp = spacy.load('en_core_web_md')
# train_x = ['this is a book book', 'i love reading book', 'i love sports shoes', 'they are fit to wear']
# train_y = [Category.BOOKS, Category.BOOKS, Category.CLOTHING, Category.CLOTHING]
# docs = [nlp(text) for text in train_x]
# train_x_word_vectors = [x.vector for x in docs]
# # print(docs[0].vector)
# test_x = ['i love t-shirts']
# test_docs = [nlp(text) for text in test_x]
# test_x_word_vectors = [x.vector for x in test_docs]
#
#
# from sklearn import svm
# svm_clf_wv = svm.SVC(kernel='linear')
# svm_clf_wv.fit(train_x_word_vectors, train_y)
# print(svm_clf_wv.predict(test_x_word_vectors))


# docs = nlp('this is a book', 'this is a car')
# print(docs[0].vector)


# from sklearn.feature_extraction.text import CountVectorizer
# vectorizer = CountVectorizer(binary=True)


# class Category:
#     BOOKS = 'BOOKS'
#     CLOTHING = 'CLOTHING'
# train_x = ['this is a book book', 'i love reading book', 'i love sports shoes', 'they are fit to wear']
# train_y = [Category.BOOKS, Category.BOOKS, Category.CLOTHING, Category.CLOTHING ]
#
#
# X = vectorizer.fit_transform(train_x)
# print(vectorizer.get_feature_names())
# print(X.toarray())
#
# test_x = ['this is a best story']
# Y = vectorizer.transform(test_x)
# #
# from sklearn import svm
# svm_clf = svm.SVC(kernel='linear')
# svm_clf.fit(X, train_y)
# print(svm_clf.predict(Y))

# Regular expressions
# import re

# regexp = re.compile(r'ab[^\s]*cd')
#
# phrases = ["abdfhhdcd", "sd absde", "aaa abdccd e", "abjdjsdcd"]
# matches = []
# #
# for phrase in phrases:
#     if re.match(regexp, phrase): # it will match with exact compile function of regexp
#     # if re.search(regexp, phrase): # it will search the whole strng and then match the condition
#         matches.append(phrase)
# print(matches)

# regexp = re.compile(r'read|story|book')
# regexp = re.compile(r'\bread|\bstory|\bbook') # so here we will make changes, here \b means the whole word not in btw any word
# phrases = [' i love reading books', 'i like this story', 'this hat is really nice']
# phrases = ['the car treaded up the hill', 'i like the history', 'this hat is really nice'] # if here we changes the phrases
           # it will still catch the same result due to search option
# matches = []

# for phrase in phrases:
#     if re.match(regexp, phrase): # result will be an empty matches list
#     if re.search(regexp, phrase): # it will append phrases[0], phrases[1]
#         matches.append(phrase)
# print(matches)

# Stemming / Lemmatization
import nltk

# nltk.download('wordnet')
# nltk.download('stopwords')
# nltk.download('punkt')

from nltk.tokenize import word_tokenize
# from nltk.stem import PorterStemmer
#
# stemmer = PorterStemmer()
#
# phrase = 'stories' # its result will be stori
# phrase = 'reading the books.'
# words = word_tokenize(phrase)
# print(words)
#
# stemmed_words = []
# # print(stemmer.stem(phrase))
# for word in words:
#     stemmed_words.append(stemmer.stem(word))
# print(' '.join(stemmed_words))
# # print(stemmed_words)

# from nltk.stem import WordNetLemmatizer
#
# lemmatizer = WordNetLemmatizer()
#
# phrase = 'reading the books.'
# words = word_tokenize(phrase)
#
# lemmatized_words = []
# for word in words:
#     lemmatized_words.append(lemmatizer.lemmatize(word, pos='v')) # pos= 'v' means it will read the verbs also
# print(' '.join(lemmatized_words))

# Stop Words Removal
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords

# stop_words = stopwords.words('english')
# print(len(stop_words))
# phrase = 'there is an example demonstrating the removal of stop words'

# words = word_tokenize(phrase)
# # print(words)
# stripped_phrase = []
# for word in words:
#     if word not in stop_words:
#         stripped_phrase.append(word)

# print(' '.join(stripped_phrase))

# Various other techniques(spell correction, sentiment & pos tagging)

# from textblob import TextBlob
#
# phrase = ['this is a bad exampleee', 'everything is all about money.', 'the book was horriable']
#
# tb_phrase = TextBlob(phrase[2])
# # print(tb_phrase)
#
# print(tb_phrase.correct()) # This will correct the phrase
# print(tb_phrase.tags)
# print(tb_phrase.sentiment)
