#%%
import numpy as np
import pandas as pd
import re
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Read stop words from file and save them in stopwords list
stopwords_file = open("stopwords.txt", "r")
try:
    content = stopwords_file.read()
    stopwords = content.split(",")
finally:
    stopwords_file.close()

# Read the dataset
news_dataset = pd.read_csv('fake_or_real_news.csv')

# Drop Nulls
news_dataset = news_dataset.dropna()

# Replace Fake with 0 and Real with 1 in label column
news_dataset.replace({"label": {'FAKE': 0, 'REAL': 1}}, inplace=True)

# Apply stratified sampling by grouping by label and take 70% of the data
daframe = pd.DataFrame(news_dataset)
daframe_groups = daframe.groupby('label', group_keys=False)
daframe_samples = daframe_groups.apply(lambda x: x.sample(frac=0.7))

# Remove Stop words function
port_stem = PorterStemmer()
def Processing(content):
    Processedcontent = re.sub('[^a-zA-Z]',' ',content)
    Processedcontent = Processedcontent.lower()
    Processedcontent = Processedcontent.split()
    Processedcontent = [port_stem.stem(word) for word in Processedcontent if not word in stopwords]
    Processedcontent = ' '.join(Processedcontent)
    return Processedcontent

# Remove stop words from news text
daframe_samples['text'] = daframe_samples['text'].apply(Processing)

# save news text data in X and its label in Y
X = daframe_samples.drop(columns=['ID','label'], axis=1)
Y = daframe_samples['label']

# Calculate term frequency for each news text
tf = TfidfVectorizer()
X = tf.fit_transform(daframe_samples['text'])
feature_name =tf.get_feature_names_out()
X = pd.DataFrame(X.todense().tolist(), columns=feature_name)


# split the data to train(70%) and test(30%) sets 
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, stratify=Y, random_state=0)

# Create each model and collect the data from it
print("--------------------------------------------------------")
print(">>>>>>>>>>>>>>>>>>>> K Nearest Neighbors Model <<<<<<<<<<<<<<<<<<<<<<")
KNeighborsmodel= KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2 )
# metric='minkowski', p=2 equivalent to the standard Euclidean metric
KNeighborsmodel.fit(x_train, y_train)  
y_pred = KNeighborsmodel.predict(x_test)
KNNACC = accuracy_score(y_test, y_pred) * 100
print("Accuracy:", KNNACC) # Calculate the Accuracy
report = classification_report(y_test, y_pred)
print('report:', report, sep='\n') # Show classification Report
print("--------------------------------------------------------")
print(">>>>>>>>>>>>>>>>>>>> Decision Tree Model <<<<<<<<<<<<<<<<<<<<<<")
DecisionTreeModel = DecisionTreeClassifier() # Create the Model
DecisionTreeModel.fit(x_train, y_train) # train the Model
y_pred = DecisionTreeModel.predict(x_test) # predic test data
DTACC = accuracy_score(y_test, y_pred) * 100
print("Accuracy:", DTACC) # Calculate the Accuracy
report = classification_report(y_test, y_pred)
print('report:', report, sep='\n') # Show classification Report
print("--------------------------------------------------------")
print(">>>>>>>>>>>>>>>>>>>> Logistic Regression Model <<<<<<<<<<<<<<<<<<<<<<")
LogisticRegressionmodel = LogisticRegression()
LogisticRegressionmodel.fit(x_train, y_train) # train the Model
y_pred = LogisticRegressionmodel.predict(x_test) # predic test data
LRACC = accuracy_score(y_test, y_pred) * 100
print("Accuracy:", LRACC) # Calculate the Accuracy
report = classification_report(y_test, y_pred)
print('report:', report, sep='\n') # Show classification Report
print("--------------------------------------------------------")
print(">>>>>>>>>>>>>>>>>>>> SVM Model <<<<<<<<<<<<<<<<<<<<<<")
SVMModel = svm.SVC(kernel='linear') # Create the Model
SVMModel.fit(x_train, y_train) # train the Model
y_pred = SVMModel.predict(x_test) # predic test data
SVMACC = accuracy_score(y_test, y_pred) * 100
print("Accuracy:", SVMACC) # Calculate the Accuracy
report = classification_report(y_test, y_pred)
print('report:', report, sep='\n') # Show classification Report
print("--------------------------------------------------------")
print(">>>>>>>>>>>>>>>>>>>> Gaussian NB Model <<<<<<<<<<<<<<<<<<<<<<")
GaussianNBmodel = GaussianNB() # Create the Model
GaussianNBmodel.fit(x_train, y_train) # train the Model
y_pred = GaussianNBmodel.predict(x_test) # predic test data
GBACC = accuracy_score(y_test, y_pred) * 100
print("Accuracy:", GBACC) # Calculate the Accuracy
report = classification_report(y_test, y_pred)
print('report:', report, sep='\n') # Show classification Report
print("--------------------------------------------------------")

xaxis = ["KNN","Decision Tree","Logistic Regression","SVM","GaussianNB"]
yaxis = [KNNACC,DTACC,LRACC,SVMACC,GBACC]

# Check the max accuracy and print what is the best model
max = max(yaxis) 
if max == KNNACC:
    print("According to the graph best model is: K Nearest Neighbors Model")
elif max == DTACC:
    print("According to the graph best model is: Decision Tree Model")
elif max == LRACC:
    print("According to the graph best model is: Logistic Regression Model")
elif max == SVMACC:
    print("According to the graph best model is: SVM Model")
elif max == GBACC:
    print("According to the graph best model is: Gaussian NB Model")

# creating the graph between different models in the system with their accuracy
fig = plt.figure(figsize=(10,10))
plt.plot(xaxis, yaxis)
plt.xlabel('Classifiers')
plt.ylabel('Accuracy')
plt.title("Classifiers' Accuracy")
plt.show()





# %%
