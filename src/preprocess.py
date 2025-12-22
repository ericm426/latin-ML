import re 
import os
from random import randint

filepath = '../data'

def load_latin_text(filepath):
    """
    load text and do basic cleaning 
    """

    with open(filepath, "r", encoding="utf-8") as file: ### open file
        text = file.read()

    text = text.lower()
    text = re.sub(r'āēīōūăĕĭŏŭ', 'aeiouaeiou', text) ### replace macron characters with non-macron counterparts
    text = re.sub(r'[^a-zāēīōūăĕĭŏŭ\s]', '', text) ### removes all non latin characters and whitespace w/regex

    return text

def create_labeled_dataset(data_dir):
    """
    Docstring for create_labeled_dataset
    
    :param data_dir: Description
    """
    texts = [] ### holds lines of poetry
    labels = [] ### holds author names

    for filename in os.listdir(data_dir):
        if filename.endswith('.txt'):
            author = re.sub(r"-.*", "", filename) #parse author name from filename
            text = load_latin_text(os.path.join(data_dir, filename))
            
            lines = [line.strip() for line in text.split('\n') if line.strip()]

            texts.extend(lines) 
            labels.extend([author] * len(lines))
    
    return texts, labels


if __name__ == "__main__":
    texts, labels = create_labeled_dataset(filepath)
    print(f"Loaded {len(texts)} lines")
    print(f"Authors: {set(labels)}")
    print(f"Sample: {labels[randint(0, len(labels)-1)]} - {texts[randint(0, len(texts)-1)]}")
