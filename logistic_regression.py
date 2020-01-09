import sklearn
import pandas as pd
import numpy as np

# read csv file by pandas
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

"""
# Organizing and observing data

print(train_data.describe())
print(train_data.describe(include=['O']))


# Since Pclass Sex SibSp and Parch are not numeric measurable variables

print(train_data[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().
      sort_values(by='Survived', ascending=False))

print(train_data[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().
      sort_values(by='Survived', ascending=False))

print(train_data[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().
      sort_values(by='Survived', ascending=False))

print(train_data[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().
      sort_values(by='Survived', ascending=False))
      
"""

# By inspection, the data for 'Ticket' and 'Cabin" are highly incomplete, Age is kept since it should have a
#   strong correlation with survival rate intuitively
# Drop columns 'Ticket' and 'Cabin' for my model training

train_data = train_data.drop(['Ticket', 'Cabin'], axis=1)
test_data = test_data.drop(['Ticket', 'Cabin'], axis=1)

# also drop names and passenger IDs for training data
train_data = train_data.drop(['Name', 'PassengerId'], axis=1)
test_data = test_data.drop(['Name'], axis=1)

data_sets = [train_data, test_data]

for data in data_sets:
    data['Sex'] = data['Sex'].map({'female': 1, 'male': 0})

# now fill up the missing Age data by taking medians according to Pclass and Sex
# Since there are 3 Pclass and 2 Sex, I need to find 6 possible medians for ages
possible_ages = np.zeros((2, 3))

# I include the data from test_data since the age information in test_data also needs to be filled up
for data in data_sets:
    for sex in range(2):
        for p_class in range(1, 4):
            temp_age = data[(data['Sex'] == sex) & (data['Pclass'] == p_class)]['Age'].dropna()

            possible_ages[sex, p_class - 1] = temp_age.median()

    for sex in range(2):
        for p_class in range(1, 4):
            data.loc[(data.Sex == sex) & (data.Pclass == p_class) & (data.Age.isnull()), 'Age'] = \
                possible_ages[sex, p_class - 1]

"""
for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']
train_df.head()
"""

data_sets[0]['AgeBand'] = pd.cut(train_data['Age'], 5)

# print(train_data[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().\
#       sort_values(by='AgeBand', ascending=True))

for data in data_sets:
    data.loc[data['Age'] <= 16.336, 'Age'] = 0
    data.loc[(data['Age'] > 16.336) & (data['Age'] <= 32.252), 'Age'] = 1
    data.loc[(data['Age'] > 32.252) & (data['Age'] <= 48.168), 'Age'] = 2
    data.loc[(data['Age'] > 48.168) & (data['Age'] <= 64.084), 'Age'] = 3
    data.loc[data['Age'] > 64.084, 'Age'] = 4

data_sets[0] = train_data.drop(['AgeBand'], axis=1)

# See if the # of family members can affect the survival rate
for data in data_sets:
    data['FamilyMember'] = data['SibSp'] + data['Parch'] + 1
    data['Alone'] = 0
    data.loc[data['FamilyMember'] == 1, 'Alone'] = 1
    data = data.drop(['SibSp', 'Parch', 'FamilyMember'], axis=1)

# print(train_data[['FamilyMember', 'Survived']].groupby(['FamilyMember'], as_index=False).mean().
#       sort_values(by='FamilyMember', ascending=True))







