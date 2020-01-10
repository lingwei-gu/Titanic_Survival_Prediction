import DataCleaning
import sklearn.linear_model as sk
import pandas as pd


def sk_logistic(train_x, train_y, test_x):
    logistic_reg = sk.LogisticRegression()
    logistic_reg.fit(train_x, train_y)  # fit in the data
    predictions = logistic_reg.predict(test_x)  # predict the survival rate for test_data
    return predictions


def manual_logistic(train_x, train_y, test_x):
    return 0


data_sets = DataCleaning.clean_data()

# set variables for x, y of the model function
train_data_x = data_sets[0].drop(['Survived'], axis=1)
train_data_y = data_sets[0]['Survived']
test_data_x = data_sets[1].drop(['PassengerId'], axis=1)

# read the answers/results from gender_submission.csv
results = pd.read_csv('gender_submission.csv')
results = results['Survived'].values.tolist()

prediction = sk_logistic(train_data_x, train_data_y, test_data_x)

# compare the answers and the predictions to get the accuracy
accuracy = 0
for i in range(len(results)):
    if prediction[i] == results[i]:
        accuracy += 1
accuracy = accuracy / len(results)
print(accuracy)  # which is 94.7% accuracy

