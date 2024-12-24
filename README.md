# Email-Spam-Detector

### A machine learning-based project to classify emails as spam or not spam using supervised learning.

## Project Overview

The Email Spam Detector is designed to identify spam emails effectively using a machine learning approach. The project applies text preprocessing and classification algorithms to differentiate between legitimate and spam emails.

## Features
1. **Data Preprocessing**
    - Remove duplicates and handle missing values.
    - Clean the text by removing unwanted characters.

```python
# Remove duplicate entries
data.drop_duplicates(inplace=True)
print(data.duplicated().sum())
```
```python
# Handle missing values
data.dropna(inplace=True)
print(data.isnull().sum())
```
``` python
def clean_text(text):
    # Remove mentions
    text = re.sub(r'@[A-Za-z0-9]+', '', text)
    # Remove hashtags
    text = re.sub(r'#', '', text)
    # Remove retweets
    text = re.sub(r'RT[\s]+', '', text)
    # Remove hyperlinks
    text = re.sub(r'https?:\/\/\S+', '', text)
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    return text

```
  2. Feature extraction using Count Vectorizer.
  3. Classification using supervised learning algorithms such as Naïve Bayes, Logistic Regression, or Support Random Forest.
   - best performance model is **Naive Bayes(Multinomial)** : 98%
  4. Accuracy metrics and performance evaluation.
     ```python
        from sklearn.metrics import  r2_score
        r2 = r2_score(y_test, predict)
        r2
       
     ```
  5. streamlit App

     ![image](https://github.com/user-attachments/assets/887b4cff-66e9-4ba6-82aa-a7f72dcc01ce)





### This data set is imbalance but, 

```python

data['label'].value_counts()

```

| label | count  |
|-------|--------|
| 1     | 43910  |
| 0     | 39538  |


1. **Total Samples** = 43910 + 39538 = 83448
2. **Class proportions:** <br> <br>
      - proportion of class 1 (spam): <br>
        Proportion = 43910 / 83448 ≈ 0.526 (or 52.6%) <br>

      - Proportion of Class 0 (not spam): <br>
        Proportion = 39538 / 83448 ≈ 0.474 (or 47.4%)

### Is the dataset Imbalanced? 
 


 - if one class of dataset value count is significantly higher or lower than the second class of dataset value count it would be an imbalance dataset<br> ex: one class of dataset value count is 90% and the second class of dataset value count is 10%  


## So, this dataset is not imbalance, we can get as a balanced dataset
