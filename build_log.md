# Date: 12/3-11
## 12/3
- Initialized repo and set up infrastructure (folders) 
- Added sample lines from three poets: Vergil, Horace, Catullus
- built preprocess for cleaning data.
- **kept macrons in text:**
    - **Thought process:**
        Macrons show vowel length, which is fundamental when analyzing latin poetry
        different poets use different metrical patters, which often depend on macrons for us as English speakers to discern more easily

## 12/4
- basic training.py implemented
- decided to go with no macrons since not all texts include them
- Trained model, 
    - **Terrible results:** Testing on unfamiliar data results in a 39.0% accuracy
Debugging by adding print statements

- realized my line of code removing macrons was not working properly
- Changed max_features to 2000 rather than 500 and added latin stopwords into the vectorizer

- realized i was using max_df=2 as one of the vectorizer parameters when it is actually min_df=2, improving my test accuracy from .390 to .541

## 12/6-7
### ngram breakthrough
- ngrams from (1,2) to (3,5) improved my model from .541 to .624, however, train accuracy went to 1.000 which indicates something wrong
- vergil's accuracy improved the most, showing that there is something about Vergil's writing style that is most easily identified with increased ngrams
- predictions have all increased and become more balanced 
- character n-grams are able to capture morphological patterns unique to latin
    - Cese endings: "us ", "is ", "um ", "em "
    - enclitics: "que", "ue" (Latin "and")
    - Author-specific inflection preferences
    - Metrical patterns at character level

**Issues remaining:**
- still severe overfitting 100% to 62.4%, around 38% gap
- looks like model is memorizing rather than learning

### random forest complexity decrease
- decreased overfitting issue, but the tradeoff is a decrease in test accuracy
- debugging possibilities: each line is only around 5 words, this is kinda short, so i may combine some lines. Other method would probably be try8ing to use logistic regression model
- Logistic regression decreased overfitting, I also decreased the C value to 0.5, (increase regularization), decreasing overfitting to 27%. I will now implement chunking

# Date: 12/15-16
### Line Chunking

**Problem:** Even with character n-grams, single lines lacked context due to small amount of words (5.4) 

**My solution** Combine 3 consecutive lines into one sample
- Lines per author: 482
- Chunks per poet: 162 (482 / 3)
- Words per chunk: around 15-16 per chunk

**results**
- my test accuracy upped by 14.5 percent. 60 -> 74.5
- Overfitting gap from 31 points -> 22.5 points 

**Performance by author:**
- Catullus: 84% recall - distinct personal/colloquial style commonly about his girlfriend, Lesbia, or other funny topics
- Vergil: 79% recall 
- Horace: 61% recall (weakest) - overlaps with Vergil's literary style a bit

**config**
- Character n-grams (3-5 chars)
- Logistic Regression (C=0.5)
- 3-line chunks
- 74.5% test accuracy

**Note** 
- 3 seems to be the sweet spot for chunk_size as of now, probably due to me limiting to 482, lines, creating 162 chunks, I'm guessing as I increase the amouunt of text, I might get better results (maybe????)

## 12/27/2025

### pickle model saving

wrote a script in training.py to save my model in folder models
created demo.py to create a interactive user demo, taking in user inputs for text

### model Limitations: Input Length Matters

Testing revealed that single lines of poetry (~7-10 words) produce unreliable 
predictions with confidence scores around 35% (barely better than random guessing 
for 3 classes).

**Example** Individual lines from Vergil's Aeneid Book 5 were frequently 
misclassified as Horace or Catullus.

**Root cause:** Model was trained on 3-line chunks (~16 words). Single lines 
lack sufficient context and character n-gram frequency for reliable classification.

**Solution:** Combine 3 consecutive lines for testing, or warn users when input 
is <15 words. This demonstrates the importance of matching test data format to 
training data format.

# Portfolio Prep
- I used claude code to create helper functions for plotting visualizations. They are then stored in /visualizations folder
- I updated the readme file and polished it up a little
- I took another photo of my final results. I'll put all the visualizations and photos into slideroom and caption them