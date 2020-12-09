# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 11:54:11 2020

@author: ASUS
"""
## Libraries 
import warnings
import numpy as np 
import pandas as pd
import seaborn as sns
import lightgbm as lgb
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, log_loss
import seaborn as sns

warnings.simplefilter(action='ignore')
sns.set_style('whitegrid')

## Data types
dtypes = {"crew": "int8",
          "experiment": "category",
          "time": "float32",
          "seat": "int8",
          "eeg_fp1": "float32",
          "eeg_f7": "float32",
          "eeg_f8": "float32",
          "eeg_t4": "float32",
          "eeg_t6": "float32",
          "eeg_t5": "float32",
          "eeg_t3": "float32",
          "eeg_fp2": "float32",
          "eeg_o1": "float32",
          "eeg_p3": "float32",
          "eeg_pz": "float32",
          "eeg_f3": "float32",
          "eeg_fz": "float32",
          "eeg_f4": "float32",
          "eeg_c4": "float32",
          "eeg_p4": "float32",
          "eeg_poz": "float32",
          "eeg_c3": "float32",
          "eeg_cz": "float32",
          "eeg_o2": "float32",
          "ecg": "float32",
          "r": "float32",
          "gsr": "float32",
          "event": "category",
         }


## Loading data


train_df = pd.read_csv("E:/E/new beginings/Projects/Predicting Awarness of Pilots from ECG , EEG data/reducing-commercial-aviation-fatalities/train.csv", dtype=dtypes)
test_df = pd.read_csv("E:/E/new beginings/Projects/Predicting Awarness of Pilots from ECG , EEG data/reducing-commercial-aviation-fatalities/test.csv", dtype=dtypes)

## Cheacking heads

train_df.head(5)
test_df.head(5)

## Ploting the overall state of the pilots throughout the experiment

plt.figure(figsize=(15,10))
sns.countplot(train_df['event'])
plt.xlabel("State of mind", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.title("Effect of disraction", fontsize=15)
plt.show()

''' 
As expected the pilot is aware most of the time. We can see that the concentration affects the most when doing CA(Channelized Attention) task.
DA(Diverted attention) and B(Startle/Surprise) have fairly minimum effect
'''
## Let us now see how the indivvidual experiment have effect on the concentration

plt.figure(figsize=(15,10))
sns.countplot('experiment', hue='event', data=train_df)
plt.xlabel("State of mind for each experiment", fontsize=12)
plt.ylabel("Count (log)", fontsize=12)
plt.yscale('log')
plt.title("Effect of distractions for each experiment ", fontsize=15)
plt.show()

'''
As expecteD , when doing CA(Channelized Attention) task the pilots concentration is fully diverted
In the other two tasks Pilots failry maintain the concentration on flying the plane.
Further on the pilots seat position has no signifiant impact on the result.
'''


## Let us see if there are any deviations from the test and the training data


features_m = ["eeg_fp1", "eeg_f7", "eeg_f8", "eeg_t4", "eeg_t6", "eeg_t5", "eeg_t3", "eeg_fp2", "eeg_o1", "eeg_p3", "eeg_pz", "eeg_f3", "eeg_fz", "eeg_f4", "eeg_c4", "eeg_p4", "eeg_poz", "eeg_c3", "eeg_cz", "eeg_o2", "ecg", "r", "gsr"]
plt.figure(figsize=(20,25))
plt.title('Eeg features distributions')
i = 0
## Plotting all the data and seeing for deviation

for feature in features_m:
    i += 1
    plt.subplot(11, 4, i)
    sns.distplot(test_df.sample(10000)[feature], label='Test set', hist=False)
    sns.distplot(train_df.sample(10000)[feature], label='Train set', hist=False)
    plt.xlim((-500, 500))
    plt.legend()
    plt.xlabel(feature, fontsize=12)
plt.show()


## ECG

plt.figure(figsize=(15,10))
sns.distplot(test_df['ecg'], label='Test set')
sns.distplot(train_df['ecg'], label='Train set')
plt.legend()
plt.xlabel("Electrocardiogram Signal (µV)", fontsize=12)
plt.title("Electrocardiogram Signal Distribution", fontsize=15)
plt.show()

## Challenge Duration and intensity through Violin plot


plt.figure(figsize=(15,10))
sns.violinplot(x='event', y='time', data=train_df.sample(50000))
plt.ylabel("Time (s)", fontsize=12)
plt.xlabel("Challenges", fontsize=12)
plt.title("Intensity of challenge", fontsize=15)
plt.show()

''' 
The violin plot shows that all three experiments are equally spread through time except the B(Surprise) where we can make out that there are roughly two distinct surprises per experiment
'''

## Gradiant Boosting


train_df['pilot'] = 100 * train_df['seat'] + train_df['crew']
print("Number of pilots : ", len(train_df['pilot'].unique()))

# Normalizing 
def normalize_by_pilots(df):
    pilots = df["pilot"].unique()
    for pilot in tqdm(pilots):
        ids = df[df["pilot"] == pilot].index
        scaler = MinMaxScaler()
        df.loc[ids, features_m] = scaler.fit_transform(df.loc[ids, features_m])
    return df

train_df = normalize_by_pilots(train_df)
train_df.head()

train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=420)
print(f"Training on {train_df.shape[0]} samples.")

features = ["crew", "seat"] + features_m


## Light Gradient Boost    
def run_lgb(df_train, df_test):
    # Classes as integers
    dic = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    try:
        df_train["event"] = df_train["event"].apply(lambda x: dic[x])
        df_test["event"] = df_test["event"].apply(lambda x: dic[x])
    except: 
        pass
    
    params = {"objective" : "multiclass",
              "num_class": 4,
              "metric" : "multi_error",
              "num_leaves" : 30,
              "min_child_weight" : 50,
              "learning_rate" : 0.1,
              "bagging_fraction" : 0.7,
              "feature_fraction" : 0.7,
              "bagging_seed" : 420,
              "verbosity" : -1
             }
    
    lg_train = lgb.Dataset(df_train[features], label=(df_train["event"]))
    lg_test = lgb.Dataset(df_test[features], label=(df_test["event"]))
    model = lgb.train(params, lg_train, 1000, valid_sets=[lg_test], early_stopping_rounds=50, verbose_eval=100)
    
    return model

model = run_lgb(train_df, val_df)

## Features

fig, ax = plt.subplots(figsize=(12,10))
lgb.plot_importance(model, height=0.8, ax=ax)
ax.grid(False)
plt.ylabel('Feature', size=12)
plt.xlabel('Importance', size=12)
plt.title("Importance of the Features of our LightGBM Model", fontsize=15)
plt.show()
pred_val = model.predict(val_df[features], num_iteration=model.best_iteration)


## Log Loss

print("Log loss on validation data :", round(log_loss(np.array(val_df["event"].values), pred_val), 3))
conf_mat_val = confusion_matrix(np.argmax(pred_val, axis=1), val_df["event"].values)

cmap = sns.cm.rocket_r

# Normalise and Confusion Matrix

labels = ['A','B','C','D']
conf_mat_valn = conf_mat_val.astype('float') / conf_mat_val.sum(axis=1)[:, np.newaxis]
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(conf_mat_valn, annot=True, fmt='.2f',cmap=cmap,xticklabels=labels,yticklabels=labels)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show(block=False)
