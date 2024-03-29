from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.metrics import accuracy_score

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
    
def word_count_reg(field,word):
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

def reg_from_one_word(X_train, X_val, y_train, y_val, word):
    X_word_train = pd.DataFrame(X_train.apply(lambda x: word_count_reg(x,word)))
    X_word_val = pd.DataFrame(X_val.apply(lambda x: word_count_reg(x,word)))
    model = LogisticRegression()
    reg = model.fit(X_word_train,y_train)
    y_pred = model.predict(X_word_val)
    return accuracy_score(y_pred,y_val)