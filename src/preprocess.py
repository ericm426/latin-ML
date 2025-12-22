import re 
import os
from random import randint
import pandas as pd 

filepath = '../data'

def load_latin_text(filepath):
    """
    load text and do basic cleaning 
    """

    with open(filepath, "r", encoding="utf-8") as file: ### open file
        text = file.read()

    text = text.lower()

    ### replace macron characters with non-macron counterparts
    macron_map = str.maketrans('āēīōūăĕĭŏŭ', 'aeiouaeiou')
    text = text.translate(macron_map)
    text = re.sub(r'[^a-z\s]', '', text) ### removes all non latin characters and whitespace w/regex

    return text

def create_labeled_dataset(data_dir, balance=False, chunk_size=1):
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

            # combine lines into chunks
            if chunk_size > 1:
                chunks = []
                for i in range(0, len(lines), chunk_size):
                    chunk = ' '.join(lines[i:i+chunk_size])
                    if chunk:
                        chunks.append(chunk)
                texts.extend(chunks)
                labels.extend([author] * len(chunks))

            else:
                texts.extend(lines) 
                labels.extend([author] * len(lines))

    if balance: ### balance the dataset to smallest number of lines 
        df = pd.DataFrame({'text': texts, 'label': labels}) ### create dataframe using pandas

        min_size = df['label'].value_counts().min()
        balanced_dfs = []
        for author in df['label'].unique():
            author_df = df[df['label'] == author]
            if len(author_df) > min_size:
                author_df = author_df.sample(n=min_size, random_state=42)
            balanced_dfs.append(author_df)
        
        df = pd.concat(balanced_dfs)
        texts = df['text'].tolist()
        labels = df['label'].tolist()

        print(f"Dataset balanced to {min_size} lines per author")


    return texts, labels


if __name__ == "__main__":
    texts, labels = create_labeled_dataset(filepath)
    print(f"Loaded {len(texts)} lines")
    print(f"Authors: {set(labels)}")
    print(f"Sample: {labels[randint(0, len(labels)-1)]} - {texts[randint(0, len(texts)-1)]}")
    print(f"Lines of Each Author\nCatullus: {labels.count('Catullus')}\nHorace: {labels.count('Horace')}\nVergil: {labels.count('Vergil')}")
