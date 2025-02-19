```python
!!pip install lightgbm
```

```python
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import ndcg_score
import numpy as np

# Load the dataset for one fold
def load_one_fole(data_path):
    X_train, y_train, qid_train = load_svmlight_file(str(data_path + 'train.txt'), query_id=True)
    X_test, y_test, qid_test = load_svmlight_file(str(data_path + 'test.txt'), query_id=True)
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    _, group_train = np.unique(qid_train, return_counts=True)
    _, group_test = np.unique(qid_test, return_counts=True)
    return X_train, y_train, qid_train, group_train, X_test, y_test, qid_test, group_test

def ndcg_single_query(y_score, y_true, k):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])

    gain = 2 ** y_true - 1

    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gain / discounts)

# calculate NDCG score given a trained model 
def compute_ndcg_all(model, X_test, y_test, qids_test, k=10):
    unique_qids = np.unique(qids_test)
    ndcg_ = list()
    for i, qid in enumerate(unique_qids):
        y = y_test[qids_test == qid]

        if np.sum(y) == 0:
            continue

        p = model.predict(X_test[qids_test == qid])

        idcg = ndcg_single_query(y, y, k=k)
        ndcg_.append(ndcg_single_query(p, y, k=k) / idcg)
    return np.mean(ndcg_)

# get importance of features
def get_feature_importance(model, importance_type='gain'):
    return model.booster_.feature_importance(importance_type=importance_type)
```

# Question 13


```python
import os
import zipfile
from sklearn.datasets import load_svmlight_file
import numpy as np
import matplotlib.pyplot as plt

file_path = "MSLR-WEB10K.zip"
destination_path = "MSLR-WEB10K"

# checks if the folder already exists, if not extract
if not os.path.exists(destination_path):
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(destination_path)
        
data_path = "./MSLR-WEB10K/Fold1/"

# loading dataset
X_train, y_train, qid_train = load_svmlight_file(str(data_path + "train.txt"), query_id=True)

# converting relevance labels to integers
y_train = y_train.astype(int)

num_unique_queries = len(np.unique(qid_train))
print(f"Number of unique queries: {num_unique_queries}")

# computing the distribution of relevance labels
relevance_counts = np.bincount(y_train, minlength=5) 
print("Relevance Label Distribution:")
for label, count in enumerate(relevance_counts):
    print(f"Label {label}: {count} occurrences")

# ploting distribution of relevence labels
plt.figure(figsize=(8, 5))
plt.bar(range(5), relevance_counts, tick_label=[0, 1, 2, 3, 4])
plt.xlabel("Relevance Label")
plt.ylabel("Frequency")
plt.title("Distribution of Relevance Labels in Training Data")
plt.show()
```

#### Output:
Number of unique queries: 6000  
Relevance Label Distribution:  
Label 0: 377957 occurrences  
Label 1: 232569 occurrences  
Label 2: 95082 occurrences  
Label 3: 12658 occurrences  
Label 4: 5146 occurrences  

![Distribution of Relevance Data](lorenzo_images/Q13.png)

# Question 14

```python
import lightgbm as lgb
import pandas as pd
from IPython.display import display

# definitions
dataset_path = "./MSLR-WEB10K/"
folds = [f"Fold{i}" for i in range(1, 6)]
ndcg_k_values = [3, 5, 10]

# Dictionary to store results
results = {}

# Loop through each fold
for fold in folds:
    print(f"\n{fold} Training:\n")
    
    # load training and testing data
    data_path = os.path.join(dataset_path, fold)
    X_train, y_train, qid_train = load_svmlight_file(os.path.join(data_path, "train.txt"), query_id=True)
    X_test, y_test, qid_test = load_svmlight_file(os.path.join(data_path, "test.txt"), query_id=True)

    # LightGBM dataset format
    train_data = lgb.Dataset(X_train, label=y_train, group=np.bincount(qid_train.astype(int)))
    test_data = lgb.Dataset(X_test, label=y_test, group=np.bincount(qid_test.astype(int)), reference=train_data)

    # LightGBM parameters
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": ndcg_k_values,  
        "learning_rate": 0.05, 
        "boosting_type": "gbdt",
        "lambda_l1": 0.1, 
        "lambda_l2": 0.1,  
        "verbosity": -1 
    }


    # training LightGBM model
    model = lgb.train(params, train_data, num_boost_round=100, valid_sets=[test_data])  


    # test set score predictions
    y_pred = model.predict(X_test)

    ndcg_scores = {f"nDCG@{k}": ndcg_score([y_test], [y_pred], k=k) for k in ndcg_k_values}

    results[fold] = ndcg_scores

    # Print nDCG results
    print(f"\n{fold} Performance:")
    for metric, score in ndcg_scores.items():
        print(f"{metric}: {score:.4f}")


results_df = pd.DataFrame(results).T
# Simply print the DataFrame
print("Final nDCG Scores:")
display(results_df)
```

#### Output:

Fold1 Performance:  
nDCG@3: 1.0000  
nDCG@5: 1.0000  
nDCG@10: 0.9266  

Fold2 Performance:  
nDCG@3: 1.0000  
nDCG@5: 1.0000  
nDCG@10: 0.9841  

Fold3 Performance:  
nDCG@3: 0.8827  
nDCG@5: 0.9152  
nDCG@10: 0.8936  

Fold4 Performance:  
nDCG@3: 0.9260  
nDCG@5: 0.9465  
nDCG@10: 0.9487  

Fold5 Performance:  
nDCG@3: 0.8520  
nDCG@5: 0.8930  
nDCG@10: 0.9306  
Final nDCG Scores:  

| Fold  | nDCG@3  | nDCG@5  | nDCG@10  |
|-------|--------|--------|---------|
| Fold1 | 1.000000 | 1.000000 | 0.926636 |
| Fold2 | 1.000000 | 1.000000 | 0.984095 |
| Fold3 | 0.882680 | 0.915210 | 0.893567 |
| Fold4 | 0.925980 | 0.946503 | 0.948721 |
| Fold5 | 0.851959 | 0.893007 | 0.930569 |
