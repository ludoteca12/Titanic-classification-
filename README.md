Titanic - Passenger Survival Prediction

This notebook aims to predict which passengers survived the sinking of the Titanic using Machine Learning techniques.
Two classification models were developed and compared:

Random Forest Classifier

K-Nearest Neighbors (KNN)

The dataset comes from the classic Kaggle Titanic competition https://www.kaggle.com/competitions/titanic/overview.

1. Importing the Libraries

Start by importing the essential Python libraries for data manipulation, visualization, and modeling.

```
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
```

2 . Loading the Data

Loading both the training and test datasets, adding a temp "Survived" column on `train.df` to keep the same structure, merging them into one for EDA and data treatment, saving the lenght of the training dataframe for dividing later.

```
train_df = pd.read_csv("train.csv")

test_df = pd.read_csv("test.csv")

test_df["Survived"] = None

# merged df
full_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
```

3 . EDA


|    |   PassengerId |   Survived |   Pclass | Name                                                | Sex    |   Age |   SibSp |   Parch | Ticket           |    Fare | Cabin   | Embarked   |
|---:|--------------:|-----------:|---------:|:----------------------------------------------------|:-------|------:|--------:|--------:|:-----------------|--------:|:--------|:-----------|
|  0 |             1 |          0 |        3 | Braund, Mr. Owen Harris                             | male   |    22 |       1 |       0 | A/5 21171        |  7.25   | nan     | S          |
|  1 |             2 |          1 |        1 | Cumings, Mrs. John Bradley (Florence Briggs Thayer) | female |    38 |       1 |       0 | PC 17599         | 71.2833 | C85     | C          |
|  2 |             3 |          1 |        3 | Heikkinen, Miss. Laina                              | female |    26 |       0 |       0 | STON/O2. 3101282 |  7.925  | nan     | S          |
|  3 |             4 |          1 |        1 | Futrelle, Mrs. Jacques Heath (Lily May Peel)        | female |    35 |       1 |       0 | 113803           | 53.1    | C123    | S          |
|  4 |             5 |          0 |        3 | Allen, Mr. William Henry                            | male   |    35 |       0 |       0 | 373450           |  8.05   | nan     | S          |
