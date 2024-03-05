
from scipy.io.arff import loadarff  # To load .arff files
import pandas
import copy #For deep copy
from rich.console import Console    # Rich used for pretty console printing
from rich.table import Table        # Rich used for printing table

from sklearn.model_selection import train_test_split  # Naive bayes
from sklearn import preprocessing  # Naive bayes & C4.5 decision tree
from sklearn.naive_bayes import GaussianNB  # Naive bayes
from sklearn.tree import DecisionTreeClassifier, export_graphviz #C4.5 decision tree
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score #C4.5 decision tree

import matplotlib.pyplot as plt #Plotting tool
import numpy as np  #General tool

from six import StringIO #Used for image creation
import pydotplus #Used for image creation
from IPython.display import Image #Used for image creation

RANDOM_STATE = 43 #Sets seed for splitting the data
TEST_SIZE = 0.25 #Sets size for test set of total data set
CCP_ALPHA = 0.0065 #Sets cost complexity for decision tree classifier

#Load, encode, and clean dataframe from .arff file
def arff_to_dataframe(arff):

    # Load in the dataset
    data = loadarff(arff)
    df = pandas.DataFrame(data[0])

    # Encode from byte literals to utf-8
    for column in df:
        df[column] = df[column].apply(lambda s: s.decode("utf-8"))

    # Clean dataframe from rows with missing values
    df = df.replace('?', np.nan)  # Replace ? with np.nan
    df.dropna(inplace=True)  # Remove rows
    df.reset_index(drop=True, inplace=True)  # Reset index

    return df

#Prints table from dataset variable
def print_table(df):

    # Initialize table
    table = Table(show_header=True)

    # Add titles to table
    titles = []
    # titles.append("number")
    table.add_column("Index")
    for col in df.columns:
        table.add_column(col)
        titles.append(col)

    # Add rows to table
    for i, _ in enumerate(df.iterrows()):
        rowList = []
        rowList.append(str(i+1))
        for title in titles:
            # print(df[title][i])
            rowList.append(df[title][i])
        table.add_row(*rowList)

    # Print table
    console.print(table)

#Run the sklearn Naive bayes on dataframe
def sklearn_naive_bayes(df):

    #Encode string into incremental value
    le = preprocessing.LabelEncoder()
    for column_name in df.columns:
        if df[column_name].dtype == object:
            df[column_name] = le.fit_transform(df[column_name])

    #Split feature part X, and label part Y, from dataset, into train and test parts
    X = df[['age', 'menopause', 'tumor-size', 'inv-nodes','node-caps', 'deg-malig', 'breast', 'breast-quad', 'irradiat']]
    Y = df[['Class']]
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    #Create and use gassian Naive bayes model
    model = GaussianNB()
    model.fit(x_train, y_train.values.ravel())

    #Run prediction
    y_pred = model.predict(x_test)

    #Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    #Create confusion matrix table
    table = Table(show_header=False, title="Confusion matrix NB")
    table.add_row(str(cm[0][0]), str(cm[0][1]))
    table.add_row(str(cm[1][0]), str(cm[1][1]))
    console.print(table)

    #Calculate and print accuracy
    accuracy = str(((cm[0][0] + cm[1][1]) / sum(map(sum, cm)))*100)[:4]
    print('NB Accuracy: ' + accuracy + "% \n")

    #Create ROC-curve
    y_score = model.predict_proba(x_test)[:,1]
    false_positive_rate, true_positive_rate, threshold = roc_curve(y_test, y_score)
    print('roc_auc_score for Naive Bayes:', roc_auc_score(y_test, y_score))
    plot_roc_curve(false_positive_rate, true_positive_rate, 'Receiver Operating Characteristic - Naive Bayes')

#Run sklearn CART tree decision algorithm on dataframe
def sklearn_cart_tree_decision(df):

    #Create image for export
    feature_names = list(df.columns[:-1])
    class_names = list(['recurrence-events','no-recurrence-events'])    #Temp hardcoded solution

    #Encode string into incremental value
    le = preprocessing.LabelEncoder()
    for column_name in df.columns:
        if df[column_name].dtype == object:
            df[column_name] = le.fit_transform(df[column_name])

    #Split feature part X, and label part Y, from dataset, into train and test parts
    X = df[['age', 'menopause', 'tumor-size', 'inv-nodes','node-caps', 'deg-malig', 'breast', 'breast-quad', 'irradiat']]
    Y = df[['Class']]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    #Fit method is used to train data
    classifier = DecisionTreeClassifier(criterion="gini" ,random_state=RANDOM_STATE, ccp_alpha=CCP_ALPHA)
    classifier = classifier.fit(X_train, Y_train)

    #Run prediction
    Y_pred = classifier.predict(X_test)

    #Create confusion matrix
    cm = confusion_matrix(Y_test, Y_pred)

    #Create confusion matrix table
    table = Table(show_header=False, title="Confusion matrix C4.5")
    table.add_row(str(cm[0][0]), str(cm[0][1]))
    table.add_row(str(cm[1][0]), str(cm[1][1]))
    console.print(table)

    #Calculate and print accuracy
    accuracy = str(((cm[0][0] + cm[1][1]) / sum(map(sum, cm)))*100)[:4]
    print('C4.5 Accuracy: ' + accuracy + "% \n")

    #Create tree image
    create_decision_tree_image(classifier, feature_names, class_names)

    #Create ROC-curve
    Y_score = classifier.predict_proba(X_test)[:,1]
    false_positive_rate, true_positive_rate, threshold = roc_curve(Y_test, Y_score)
    print('roc_auc_score for Decision Tree:', roc_auc_score(Y_test, Y_score))
    plot_roc_curve(false_positive_rate, true_positive_rate, 'Receiver Operating Characteristic - Decision Tree')

#Create image of decision tree
def create_decision_tree_image(decision_tree, feature_names, class_names):

    dot_data = StringIO()

    export_graphviz(decision_tree, 
            out_file=dot_data,
            filled=True,
            rounded=True,
            special_characters=True,
            feature_names=feature_names,
            class_names=class_names)

    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png("big_tree.png")
    Image(graph.create_png())

#Creats and prints a roc curve
def plot_roc_curve(false_positive_rate, true_positive_rate, title):
    plt.subplots(1, figsize=(10,10))
    plt.title(title)
    plt.plot(false_positive_rate, true_positive_rate)
    plt.plot([0, 1], ls="--")   #Straight line
    plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7") #Straight line
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.draw()  #Create plot

# Initialize console and table object for pretty printing
console = Console()

with console.status("[bold green]Processing data...") as status:
    df = arff_to_dataframe("breast-cancer.arff")
    print_table(df)
    sklearn_naive_bayes(copy.deepcopy(df))
    sklearn_cart_tree_decision(copy.deepcopy(df))
    print('Test size: ' + str(TEST_SIZE * 100) + '%')
    print('Random state: ' + str(RANDOM_STATE))
    plt.show()  #Show plots
