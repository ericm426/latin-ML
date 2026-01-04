import sys
import pickle
import os
import time 

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

from collections import Counter

from preprocess import create_labeled_dataset

texts, labels = create_labeled_dataset('../data/', balance=True, chunk_size=3)

print(f"Total Dataset: {len(texts)} samples, {len(set(labels))} authors")

X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42, stratify=labels) # split data into test/train sets, 20% test
### answer to everything is always forty-two!!!!

#converts text to number based on importance. How many occurences 

vectorizer = TfidfVectorizer(
    max_features=2000, #increase from 500-2000 to attempt to improve results
    analyzer='char',
    ngram_range=(3, 5),
    min_df=2
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# train classifier"

clf = LogisticRegression(
    max_iter = 1000,
    C=0.5,
    random_state=42
)

clf.fit(X_train_vec, y_train)

# evaluation
train_acc = clf.score(X_train_vec, y_train)
test_acc = clf.score(X_test_vec, y_test)

print(f"Train Accuracy: {train_acc:.3f}")
print(f"Test Acccuracy: {test_acc:.3f}")

#predict
y_pred = clf.predict(X_test_vec)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("-" * 100)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

#feature importance ( top 20)
print("-" * 100)

feature_names = vectorizer.get_feature_names_out()
if hasattr(clf, 'feature_importances_'):
    # Random Forest
    importances = clf.feature_importances_
    top_features_idx = np.argsort(importances)[-20:]
    
    print("\nTop 20 most important features:")
    for idx in reversed(top_features_idx):
        print(f"{feature_names[idx]}: {importances[idx]:.4f}")
        
elif hasattr(clf, 'coef_'):
    # Logistic Regression
    # For multiclass, coef_ has shape (n_classes, n_features)
    # Take absolute values and sum across classes to get overall importance
    importances = np.abs(clf.coef_).sum(axis=0)
    top_features_idx = np.argsort(importances)[-20:]
    
    print("\nTop 20 most important features:")
    for idx in reversed(top_features_idx):
        print(f"{feature_names[idx]}: {importances[idx]:.4f}")
else:
    print("\nsomething went wrong")



print("-" * 100)

print(f"Balanced to min lines")
print(f"Label distribution: {Counter(labels)}")
print(f"Total Dataset: {len(texts)} samples, {len(set(labels))} authors")
print("-" * 100)

# sample texts
print("\nSample chunks from each author:")
for author in set(labels):
    idx = labels.index(author)
    print(f"{author}: {texts[idx][:100]}...")

# chunk lengths
print("-"*50)
import numpy as np
line_lengths = [len(text.split()) for text in texts]
print(f"\nAverage words per chunk: {np.mean(line_lengths):.1f}")
print(f"Min: {np.min(line_lengths)}, Max: {np.max(line_lengths)}")

#save model
os.makedirs('../models', exist_ok=True)

pickle.dump(clf, open('../models/classifier.pkl','wb')) #clkassifer
pickle.dump(vectorizer, open('../models/vectorizer.pkl', 'wb')) #vectorizer

metadata = {
    'model_type': 'LogisticRegression',
    'train_accuracy': train_acc,
    'test_accuracy': test_acc,
    'chunk_size': 3,
    'authors': list(clf.classes_),
    'n_features': vectorizer.max_features, 
    'analyzer': vectorizer.analyzer,
    'ngram_range': vectorizer.ngram_range
}

pickle.dump(metadata, open('../models/metadata.pkl', 'wb'))

print("-"*50)
print("model save success")


