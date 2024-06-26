from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import re
from cleantext import clean
import nltk
from nltk.stem.snowball import SnowballStemmer
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix


#Functions used for models
def label_news(field, fakenews_labels):
    '''function for labeling news articles as either fake, reliable or unknown'''
    if field in fakenews_labels:
        return 'fake'
    elif field == 'reliable':
        return 'reliable'
    else:
        return 'unknown'
    
def bin_target(x):
    '''Function that transforms labels "reliable" and "fake"
    into 0 for reliable content and 1 for fake content'''
    if x=="reliable":
        return 0
    if x=="fake":
        return 1
    
def word_count(field,word):
    '''Function that counts a how many times a word appears in a document'''
    count = 0
    for words in field:
        if words == word:
            count+=1
    return count   
    
def gridSearch(model, solver, penalties, data, target):
    '''Function for gridsearching (hyperparameter tuning)'''
    model = model()
    parameters = {'solver' : solver,'penalty': penalties, 'C' : [0.00001,0.001, 0.01, 0.1, 1.0, 10]}
    clf = GridSearchCV(model, parameters, scoring='accuracy', cv=5)
    clf.fit(data,target)
    return clf

#functions used for preprocessing
def remove_dates_from_content(content):
    '''Function that attempts to substitute dates in a document for the token "_DATE_".
    If it fails to do so - for example if the content is not convertable to string, it 
    handles the typeerror exception and doesnt do anything with the content.'''
    date_pattern = re.compile(r"(([0-9]{4}-(0[0-9]|1[0-2])-([0-2][0-9]|[3[01])|[a-z]{,9} [0-9]{1,2}, [0-9]{2,4})|\b(\w+\s)(\d{2})(th)?,?(\s\d{4})\b)")
    try:
        content_without_dates = re.sub(date_pattern, "_DATE_", str(content))
    except TypeError:
        content_without_dates = content
    return content_without_dates 

def remove_bar_from_content(content):
    '''Function for removing every occurence of "|"'''
    content_without_bar = str(content).replace("|", "")
    return content_without_bar

def remove_a_from_content(content):
    '''Function for removing every occurence of "a"'''
    return [word for word in content if word != "a"]


def remove_stopwords(list, stopwords):
     '''Function that returns a list containing a document with the stopwords removed'''
     return [word for word in list if word not in stopwords]

#Initializing stemmer

def list_stemmer (wordlist): #stemmer hvert ord i en liste
    '''Function that stems each word in the given input list and returns this'''
    stemmer = SnowballStemmer("english")
    stemmed_list = []
    for word in wordlist:
        stemmed_list.append(stemmer.stem(word))
    return stemmed_list


def reduction_rate(after,before):
     '''Computes the reduction rate of the size of the vocabulary
     and returns this rounded to 3 decimal points'''
     return round((before - after)/before, 3)

def word_frequency_plot(counter_dict, title, y_lim=None):
    '''Plots the frequency of the 10000 most common words given a counter object
    of words and their frequencies and a title'''

    # Select the top 10,000 most common words and frequencies
    most_common_words = counter_dict.most_common(10000)
    words, frequencies = zip(*most_common_words)
    
    # Creating the plot
    plt.figure(figsize=(20, 10))
    plt.bar(range(len(frequencies)), frequencies, width=1.0)
    
    plt.title(title)
    plt.xlabel('Words Ranked by Frequency')
    plt.ylabel('Frequency')  
    
    if y_lim:
        plt.ylim(y_lim)
    # Removing x-ticks
    plt.xticks([])
    
    plt.tight_layout()
    plt.show()

def update_frequency_counter(frequency_counter, content):
    '''function that updates a counter object with the words from each document
    in a corpus. If theyre already in the counter the values are simply updated'''
    for list in content:
        frequency_counter.update(list)
    return frequency_counter

def evaluate_and_plot(model, name, test_vec, y_test):
    '''Function for evaluating a given model on the metrics:
    accuracy, f1, precision and recall. Futhermore it plots the confusion
    matrix of the models predictions'''

    print(f'evaluating {name}')
    y_pred = model.predict(test_vec)

    #Evaluating performance on different metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    #Printing performance values
    print(f'accuracy: {accuracy}\nf1: {f1}\nPrecision: {precision}\nRecall: {recall}')

    #Plotting confusion matrix
    cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.matshow(cm, cmap=plt.cm.Blues, alpha = 0.3)
    for i in range(2):
        for j in range(2):
            ax.text(x=j, y=i, s=cm[i, j], va='center', ha='center', size='medium')
    plt.xlabel('Predicted', fontsize=8)
    plt.ylabel('Actual', fontsize=8)
    plt.title(f'Confusion Matrix, {name}', fontsize=10)
    plt.show


def vocabulary_size(series):
    '''Computes the vocabulary size of a corpus by iterating through every word
    in every document in the corpus and adding it to a dictionary if it hasnt been seen before
    as well as incrementing a variable counting every unique word. Returns the unique_word_count
    and sorted list of words and their frequencies'''
    # Initialize a counter for unique words count and dictionary to store word frequencies
    unique_words_count = 0
    word_frequency = {}
    # Iterate through each element in the series
    for field in series:
        # Iterate through each word in the current element
        for word in field:
            if word in word_frequency: 
                # If word already in dict, increment its count
                word_frequency[word] += 1
            else:
                # If not add it to the dictionary with count 1 and increment unique_words_count
                word_frequency[word] = 1
                unique_words_count += 1
    # Sort the word_frequency dictionary by frequency in descending order
    sorted_word_frequency = sorted(word_frequency.items(), key=lambda x: x[1], reverse=True)

    return unique_words_count, sorted_word_frequency

def most_frequent(series, from_value, to_value):
    '''returns a dictionary containing the
    "n" most frequent words in an interval - [from_value:to_value] - in the input series.'''
    return dict(vocabulary_size(series)[1][from_value:to_value]) 

def mean_length_of_words(wordlist):
    totallength = 0
    for word in wordlist:
        totallength = totallength + len(str(word))
    return totallength/len(wordlist)

def unique_words_ratio(wordlist):
    count = 0
    words = {}
    for word in wordlist:
        if word not in words:
            words[word]=1
            count +=1
    return count/len(wordlist)

def mention_ratio(data, topic,type):
    #gives mention ratio for whole column
    count = 0
    for field in data[data.type==type]['content_no_stopwords']:
        if topic in field:
            count+=1
    return count/len(data[data.type==type])