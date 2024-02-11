# -*- coding: utf-8 -*-

contractions = {
    "ain't": "am not / are not / is not / has not / have not",
    "aren't": "are not / am not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he had / he would",
    "he'd've": "he would have",
    "he'll": "he shall / he will",
    "he'll've": "he shall have / he will have",
    "he's": "he has / he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how has / how is / how does",
    "I'd": "I had / I would",
    "I'd've": "I would have",
    "I'll": "I shall / I will",
    "I'll've": "I shall have / I will have",
    "I'm": "I am",
    "I've": "I have",
    "isn't": "is not",
    "it'd": "it had / it would",
    "it'd've": "it would have",
    "it'll": "it shall / it will",
    "it'll've": "it shall have / it will have",
    "it's": "it has / it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she had / she would",
    "she'd've": "she would have",
    "she'll": "she shall / she will",
    "she'll've": "she shall have / she will have",
    "she's": "she has / she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as / so is",
    "that'd": "that would / that had",
    "that'd've": "that would have",
    "that's": "that has / that is",
    "there'd": "there had / there would",
    "there'd've": "there would have",
    "there's": "there has / there is",
    "they'd": "they had / they would",
    "they'd've": "they would have",
    "they'll": "they shall / they will",
    "they'll've": "they shall have / they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we had / we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what shall / what will",
    "what'll've": "what shall have / what will have",
    "what're": "what are",
    "what's": "what has / what is",
    "what've": "what have",
    "when's": "when has / when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where has / where is",
    "where've": "where have",
    "who'll": "who shall / who will",
    "who'll've": "who shall have / who will have",
    "who's": "who has / who is",
    "who've": "who have",
    "why's": "why has / why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you had / you would",
    "you'd've": "you would have",
    "you'll": "you shall / you will",
    "you'll've": "you shall have / you will have",
    "you're": "you are",
    "you've": "you have"
}

import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re

# Data frame processing
df = pd.read_csv("dataff.csv")
df.pop("id")

df.drop_duplicates(subset="title", inplace=True, ignore_index=True)
df['isPhone'] = df['isPhone'].fillna("0").astype(int)
df['price'] = df.price.apply(lambda x: x.replace(',', '') if ',' in x.lower() else x)
df['min_price'] = df.price.apply(lambda x: float(x.split('-')[0]) if '-' in x.lower() else float(x))
df['max_price'] = df.price.apply(lambda x: float(x.split('-')[1]) if '-' in x.lower() else float(x))
df['price'] = (df['min_price'] + df['max_price']) / 2
df = df.drop(['min_price', 'max_price'], axis=1)

df = df.convert_dtypes(infer_objects=False, convert_string=True, convert_integer=False, convert_floating=False)
df.dtypes


# **************************** Title cleaning **************************************
def title_processing(title, lower_case=True, stop_words=True, stem=True, lemma=True, contrac=True):
    if lower_case:
        title = title.lower()

    if contrac:
        title = title.split()
        text = []
        for word in title:
            if word in contractions:
                text.append(contractions[word])
            else:
                text.append(word)
        title = " ".join(text)

    title = re.sub(r'w/', ' ', title)
    title = re.sub('<[^<]+?>', '', title)
    title = re.sub(r'https?:\/\/.*[\r\n]*', '', title, flags=re.MULTILINE)
    title = re.sub(r'\<a href', ' ', title)
    title = re.sub(r'&amp;', '', title)
    title = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', title)
    title = re.sub(r'<br />', ' ', title)
    title = re.sub(r'\'', ' ', title)

    tokens = word_tokenize(title)
    tokens = [word for word in tokens if word.isalpha()]

    if stop_words:
        sw = set(stopwords.words('english'))
        tokens = [word for word in tokens if not word in sw]
    if stem:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(token) for token in tokens]
    if lemma:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
    title = " ".join(tokens)
    return title

# *********************************** Data preparation *******************************

# Apply the processing technics implemented in the "title_processing" method on the title feature
df.title = df.title.apply(title_processing)

from sklearn.model_selection import train_test_split

X = df.title
y = df.isPhone

# Create a matrix of TF-IDF features from the titles
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
X_vectors = vectorizer.fit_transform(X)

from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression

# *********************************** Cleaning Noise *******************************
from cleanlab.latent_estimation import estimate_cv_predicted_probabilities
from cleanlab.pruning import get_noise_indices

probabilities2 = estimate_cv_predicted_probabilities(X_vectors, np.array(y), clf=MultinomialNB())
label_error_indices2 = get_noise_indices(s=np.array(y), psx=probabilities2)

probabilities_xgb2 = estimate_cv_predicted_probabilities(X_vectors, np.array(y), clf=XGBClassifier())
label_error_indices_xgb2 = get_noise_indices(s=np.array(y), psx=probabilities_xgb2)

probabilities_svm2 = estimate_cv_predicted_probabilities(X_vectors, np.array(y), clf=svm.SVC(probability=True))
label_error_indices_svm2 = get_noise_indices(s=np.array(y), psx=probabilities_svm2)

probabilities_lg2 = estimate_cv_predicted_probabilities(X_vectors, np.array(y), clf=LogisticRegression())
label_error_indices_lg2 = get_noise_indices(s=np.array(y), psx=probabilities_lg2)

X = df.title[
    (label_error_indices2 == False) & (label_error_indices_xgb2 == False) & (label_error_indices_svm2 == False) & (
                label_error_indices_lg2 == False)]
y = df.isPhone[
    (label_error_indices2 == False) & (label_error_indices_xgb2 == False) & (label_error_indices_svm2 == False) & (
                label_error_indices_lg2 == False)]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

train_vectors = vectorizer.fit_transform(X_train)
test_vectors = vectorizer.transform(X_test)

# *********************************** Model Creation *******************************
SVM_clf = svm.SVC().fit(train_vectors, y_train)

# ****************************** Evaluating the predictions *************************
from sklearn.metrics import accuracy_score
SVM_predictions = SVM_clf.predict(test_vectors)
print(accuracy_score(y_test, SVM_predictions))

# ********************************** Prediction Function ************************
def predcit_phone(title):
    title = title_processing(title)
    title_input = [title]
    title_input = vectorizer.transform(title_input)
    result = SVM_clf.predict(title_input)
    if result == [1]:
        result = True
    elif result == [0]:
        result = False
    return result


predcit_phone("UNEN iphone Charger（3/3/6/6/10FT）5 Pack-Black and Blue")
