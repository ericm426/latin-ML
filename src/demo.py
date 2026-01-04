import pickle
import re


def load_model():
    clf = pickle.load(open('../models/classifier.pkl', 'rb'))
    vectorizer = pickle.load(open('../models/vectorizer.pkl', 'rb'))
    metadata = pickle.load(open('../models/metadata.pkl', 'rb'))
    return clf, vectorizer, metadata

def preprocess(text):
    macron_map = {'ā':'a', 'ē':'e', 'ī':'i', 'ō':'o', 'ū':'u',
                'Ā':'A', 'Ē':'E', 'Ī':'I', 'Ō':'O', 'Ū':'U'}
    for macron, plain in macron_map.items():
        text = text.replace(macron, plain)
    
    # Remove numbers and brackets
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    
    # lowercase and remove non-Latin characters
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    
    return text

def predict_author(text, clf, vectorizer):
    cleaned_text = preprocess(text)

    #vectorize
    text_vec = vectorizer.transform([cleaned_text])

    # Predict
    prediction = clf.predict(text_vec)[0]
    probabilities = clf.predict_proba(text_vec)[0]
    
    # display results
    print(f"\nPredicted Author: {prediction.upper()}")
    print("\nConfidence:")
    for author, prob in zip(clf.classes_, probabilities):
        print(f"  {author}: {prob:.1%}")
    
    return prediction


if __name__ == "__main__":
    print("LATIN POETRY CLASSIFIER DEMO")
    print("Loading Model...")
    clf, vectorizer, metadata = load_model()
    print(f"Model Loaded. Test accuracy:{metadata['test_accuracy']:.1%}\n")

    print(f"Authors: {', '.join(metadata['authors'])}")

    while True:
        print("-"*100)
        try: 
            text = input("Enter text to analyze (q to quit): ")

            if text.lower() == "q":
                print("bye")
                break
            if not text:
                print("Enter text to analyze")
            if len(text.split()) < 5:
                print("warning: short input may be less reliable")

            predict_author(text,clf,vectorizer)

        except Exception as e:
            print(f"try again error occured: {e}")


