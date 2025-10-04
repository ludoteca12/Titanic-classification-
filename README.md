# Titanic - Passenger Survival Prediction

This notebook aims to predict which passengers survived the sinking of the Titanic using Machine Learning techniques.
Two classification models were developed and compared:

Random Forest Classifier

K-Nearest Neighbors (KNN)

The dataset comes from the classic [Kaggle Titanic competition](https://www.kaggle.com/competitions/titanic/overview)

## 1. Importing the Libraries

Start by importing the essential Python libraries for data manipulation, visualization, and modeling.

```
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
```

## 2 . Loading the Data

Loading both the training and test datasets, adding a temp "Survived" column on `train.df` to keep the same structure, merging them into one for EDA and data treatment and saving the lenght of the training dataframe for dividing later.

```
train_df = pd.read_csv("train.csv")

test_df = pd.read_csv("test.csv")

test_df["Survived"] = None

# merged df
full_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
```

## 3 . EDA

First we take a look on what the data looks like.

```
full_df.head()
```

|    |   PassengerId |   Survived |   Pclass | Name                                                | Sex    |   Age |   SibSp |   Parch | Ticket           |    Fare | Cabin   | Embarked   |
|---:|--------------:|-----------:|---------:|:----------------------------------------------------|:-------|------:|--------:|--------:|:-----------------|--------:|:--------|:-----------|
|  0 |             1 |          0 |        3 | Braund, Mr. Owen Harris                             | male   |    22 |       1 |       0 | A/5 21171        |  7.25   | nan     | S          |
|  1 |             2 |          1 |        1 | Cumings, Mrs. John Bradley (Florence Briggs Thayer) | female |    38 |       1 |       0 | PC 17599         | 71.2833 | C85     | C          |
|  2 |             3 |          1 |        3 | Heikkinen, Miss. Laina                              | female |    26 |       0 |       0 | STON/O2. 3101282 |  7.925  | nan     | S          |
|  3 |             4 |          1 |        1 | Futrelle, Mrs. Jacques Heath (Lily May Peel)        | female |    35 |       1 |       0 | 113803           | 53.1    | C123    | S          |
|  4 |             5 |          0 |        3 | Allen, Mr. William Henry                            | male   |    35 |       0 |       0 | 373450           |  8.05   | nan     | S          |

Then lets understand some of the dataframe features.

```
# number of rows and columns
print(Fore.CYAN + "df shape: " + Style.RESET_ALL)
print(f"{full_df.shape}\n")

# column names, data types, non-null values
print(Fore.GREEN + "df info: " + Style.RESET_ALL)
print(f"{full_df.info()}\n") 

# column names, data types, non-null values
print(Fore.GREEN + "df unique: " + Style.RESET_ALL)
print(f"{full_df.nunique()}\n") 

# NaN values
print(Fore.YELLOW + "df isnull sum: " + Style.RESET_ALL)
print(f"{full_df.isnull().sum()}\n")

# count, mean, std, min, max, etc.
print(Fore.MAGENTA + "df describe: " + Style.RESET_ALL)
print(f"{full_df.describe()}\n")
```

Output:
```
df shape: 
(1309, 12)

df info: 
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1309 entries, 0 to 1308
Data columns (total 12 columns):
 #   Column       Non-Null Count  Dtype  
---  ------       --------------  -----  
 0   PassengerId  1309 non-null   int64  
 1   Survived     891 non-null    object 
 2   Pclass       1309 non-null   int64  
 3   Name         1309 non-null   object 
 4   Sex          1309 non-null   object 
 5   Age          1046 non-null   float64
 6   SibSp        1309 non-null   int64  
 7   Parch        1309 non-null   int64  
 8   Ticket       1309 non-null   object 
 9   Fare         1308 non-null   float64
 10  Cabin        295 non-null    object 
 11  Embarked     1307 non-null   object 
dtypes: float64(2), int64(4), object(6)
memory usage: 122.8+ KB
None

df unique: 
PassengerId    1309
Survived          2
Pclass            3
Name           1307
Sex               2
Age              98
SibSp             7
Parch             8
Ticket          929
Fare            281
Cabin           186
Embarked          3
dtype: int64

df isnull sum: 
PassengerId       0
Survived        418
Pclass            0
Name              0
Sex               0
Age             263
SibSp             0
Parch             0
Ticket            0
Fare              1
Cabin          1014
Embarked          2
dtype: int64

df describe: 
               PassengerId               Pclass                  Age  \
count              1309.00              1309.00              1046.00   
mean                655.00                 2.29                29.88   
std                 378.02                 0.84                14.41   
min                   1.00                 1.00                 0.17   
25%                 328.00                 2.00                21.00   
50%                 655.00                 3.00                28.00   
75%                 982.00                 3.00                39.00   
max                1309.00                 3.00                80.00   

                     SibSp                Parch                 Fare  
count              1309.00              1309.00              1308.00  
mean                  0.50                 0.39                33.30  
std                   1.04                 0.87                51.76  
min                   0.00                 0.00                 0.00  
25%                   0.00                 0.00                 7.90  
50%                   0.00                 0.00                14.45  
75%                   1.00                 0.00                31.27  
max                   8.00                 9.00               512.33  
```

Since we gonna use KNN and RandomForrest our data must be suitable for those models, so the next steps are:

Fix missing values from Age and Cabin

Separete Pclass into 3 new columns = 1 = Upper_class, 2 = Middle_class and 3 = Lower_class

Separete Embarked into 3 columns = C = Cherbourg, Q = Queenstown and S = Southampton

Transform Sex into a binary Male and Female column

Drop useless columns - Name, Ticket, Suvived, PassengerId

## 4. Data Preprocessing

For the missing values on Age column we gonna use a median based on the Pclass and Sex column and fill in all NaN values:

```
# Age

# creating a median by group: Pclass + Sex and filling all the nans
full_df["Age"] = full_df.groupby(["Pclass", "Sex"])["Age"].transform(
    lambda x: x.fillna(x.median())
)

```

For Cabin instead of filling NaN, a new binary column Has_Cabin was created:

```
# Cabin

# instead of filling the nans, create a new Has_Cabin column
full_df["Has_Cabin"] = full_df["Cabin"].notna().astype(int)
```

The transformation of the Pclass, Embarked and Sex column will be done using dummy variables to create new columns:

```
# Pclass

# creating dummies
full_df = pd.get_dummies(full_df, columns=["Pclass"])

# renaming
full_df = full_df.rename(columns={
    "Pclass_1": "Upper_class",
    "Pclass_2": "Middle_class",
    "Pclass_3": "Lower_class"
})

full_df[["Upper_class", "Middle_class", "Lower_class"]] = full_df[["Upper_class", "Middle_class", "Lower_class"]].astype(int)
```

```
# Embarked

# creating dummies
full_df = pd.get_dummies(full_df, columns=["Embarked"])

# renaming
full_df = full_df.rename(columns={
    "Embarked_C": "Cherbourg",
    "Embarked_Q": "Queenstown",
    "Embarked_S": "Southampton"
})

full_df[["Cherbourg", "Queenstown", "Southampton"]] = full_df[["Cherbourg", "Queenstown", "Southampton"]].astype(int)
```

```
# Sex

# creating dummies
full_df = pd.get_dummies(full_df, columns=["Sex"])

# rename
full_df = full_df.rename(columns={
    "Sex_male": "Male",
    "Sex_female": "Female"
})

full_df[["Male", "Female"]] = full_df[["Male", "Female"]].astype(int)
```

We run a last NaN check since the data had a lot of missing values

```
# checking if the only nan values are the temp Survived
full_df.isna().sum()
```
Output:

```
Survived        418
Age               0
SibSp             0
Parch             0
Fare              1
Has_Cabin         0
Upper_class       0
Middle_class      0
Lower_class       0
Cherbourg         0
Queenstown        0
Southampton       0
Female            0
Male              0
dtype: int64
```

We can see that theres 1 NaN value in the Fare column, to fix that we just gonna replace it with a 0

```
# replacing nan for 0 since both models doesnt accept NaN values
full_df = full_df.fillna(0)
```

Lastly we drop unwanted columns

```
# dropping
full_df = full_df.drop(columns=["Name", "Ticket", "Cabin", "PassengerId"])
```

Let's take a look on what the data looks like now


|    |   Survived |   Age |   SibSp |   Parch |    Fare |   Has_Cabin |   Upper_class |   Middle_class |   Lower_class |   Cherbourg |   Queenstown |   Southampton |   Female |   Male |
|---:|-----------:|------:|--------:|--------:|--------:|------------:|--------------:|---------------:|--------------:|------------:|-------------:|--------------:|---------:|-------:|
|  0 |          0 |    22 |       1 |       0 |  7.25   |           0 |             0 |              0 |             1 |           0 |            0 |             1 |        0 |      1 |
|  1 |          1 |    38 |       1 |       0 | 71.2833 |           1 |             1 |              0 |             0 |           1 |            0 |             0 |        1 |      0 |
|  2 |          1 |    26 |       0 |       0 |  7.925  |           0 |             0 |              0 |             1 |           0 |            0 |             1 |        1 |      0 |
|  3 |          1 |    35 |       1 |       0 | 53.1    |           1 |             1 |              0 |             0 |           0 |            0 |             1 |        1 |      0 |
|  4 |          0 |    35 |       0 |       0 |  8.05   |           0 |             0 |              0 |             1 |           0 |            0 |             1 |        0 |      1 |


## 5. Splitting Training and Test Sets

After cleaning, we separate the dataframes and the target variable (Survived) from the features.
The test dataset remains without labels, to be used for final prediction.

```
# separeting df
train_cleaned = full_df.iloc[:train_len].copy()
test_cleaned  = full_df.iloc[train_len:].copy()

# Target
y_train = train_cleaned["Survived"].astype(int)
X_train = train_cleaned.drop(columns=["Survived"])

# Test (dropping temp Survived)
X_test = test_cleaned.drop(columns=["Survived"])
```

## 6. Model Training

We gonna use two different models, RandomForestClassifier and KNeighborsClassifier to see which one performs best on this data.

For both models we will use the default parameters to see which one performs best and the try to boost its accuracy with some parameters tunning.

```
# model RandomForestClassifier

# instantiating the model
rf_model = RandomForestClassifier(random_state=42)

# fitting the train and test data
rf_model.fit(X_train, y_train)

# train df prediction
y_pred_rf = rf_model.predict(X_train)
```

```
# model KNeighborsClassifier

# instantiating the model
knn_model = KNeighborsClassifier(n_neighbors=5)

# fitting the train and test data
knn_model.fit(X_train, y_train)

# train df prediction
y_pred_knn = knn_model.predict(X_train)
```

Comparing both models accuracy

```
Random Forest Classifier (training data)
Accuracy: 0.9865319865319865

K-Nearest Neighbors (training data)
Accuracy: 0.8159371492704826
```

The RandomForestClassifier got an extremely high acurracy, probably indicates some kind of overfitting based on the train data so in the next session we will use it as our model but using GridSearchCV to find a more realistic accuracy score.

## 7. Hyperparameter Tuning

We will perform hyperparameter tuning using GridSearchCV, which exhaustively tests combinations of parameters through cross-validation, to get a more real accuracy

```
# params grid
param_grid_rf = {
    "n_estimators": [100, 200, 500],
    "max_depth": [None, 5, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"]
}

# passing params through GridSearchCV
grid_rf = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid_rf,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid_rf.fit(X_train, y_train)

rf_scores = cross_val_score(grid_rf.best_estimator_, X_train, y_train, cv=5)
print("Cross-validation mean accuracy (Random Forest):", rf_scores.mean())

print("Best parameters for RandomForestClassifier:", grid_rf.best_params_)
print("Best score RandomForestClassifier:", grid_rf.best_score_)
```

Output:

```
Best parameters for RandomForestClassifier: {'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 200}
Best score RandomForestClassifier: 0.8271859895800642
```

Now we got a more reliable accuracy.


## 8. Generating Predictions

Since this is a kaggle competition dataset, all we got do now is create a submission dataframe with our predictions.

```
# creating submission df
submission_rf = pd.DataFrame({"PassengerId": test_df["PassengerId"], "Survived": rf_predictions})
submission_rf.to_csv("Kaggle_submission.csv", index=False)
```

## 9. Conclusion

This project demonstrates a complete supervised learning workflow, including:

Data preprocessing and feature engineering

Model training and evaluation

Hyperparameter optimization

Generating predictions for submission.
