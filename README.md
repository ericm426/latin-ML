# Latin Poetry Classifier

This project trains a machine learning model to classify Latin poetry by author (Vergil, Horace, Catullus) with 75.5% accuracy by discovering morphological patterns that distinguish each poet's unique style.


Classical scholars can identify poets by their distinct stylistic choices. Could a machine learn to recognize them?

Build a classifier that distinguishes between three major Latin poets using only their text, without modern linguistic annotations or translations.

---

## Accuracy from 39% -> 75.5%

### Initial Attempt (39% accuracy)
**Approach:** Word-based TF-IDF with unigrams/bigrams
**Problem:** Latin's inflectional morphology breaks word-based approaches: *amor* (love), *amoris* (of love), *amorem* (love-accusative) are treated as completely different features.
**Result:** Severe overfitting (100% train vs 39% test) and near-random performance

### Stage 2: Character N-grams Discovery (62.4% accuracy)
**Key Insight:** Latin's meaning lives in morphemes, not words. Character n-grams (3-5 characters) can capture:
- **Case endings:** `"um "`, `"is "`, `"ae "` (declension patterns)
- **Enclitics:** `"que"`, `"ue"` (Latin's suffix for "and")
- **Author-specific morphology:** Vergil's preference for epic forms

**Implementation:**
```python
vectorizer = TfidfVectorizer(
    analyzer='char',      # Character-level instead of word-level
    ngram_range=(3, 5),   # Captures morphological patterns
    max_features=2000
)
```

**Result:** +23 percentage point improvement, discovered that Vergil uses `"que"` 43% more than other poets in epic contexts

### Stage 3: Context Through Chunking (75.5% accuracy)
**Problem:** Single lines averaged only 5.4 words, insufficient context for reliable classification
**Solution:** Combine 3 consecutive lines into chunks (~16 words each)
**Impact:**
- Test accuracy: 62.4% to 75.5% (+13.1 points)
- Overfitting gap: 31% to 22.5%
- Longer character n-gram patterns could emerge (syntactic preferences, not just morphology)

**Insight:** Chunking by combining three consecutive lines showed the best results, chunks of 4, 5 and 2 were impacted either by lack of context (due to less data, if you chunk 5 lines together, you get less total data to analyze)

---

## Final Results

| Metric | Value |
|--------|-------|
| **Overall Test Accuracy** | **75.5%** |
| **Vergil (Epic)** | 79% recall |
| **Catullus (Personal/Lyric)** | 84% recall |
| **Horace (Lyric/Satire)** | 61% recall* |

*Horace's lower performance reflects genuine stylistic overlap with Vergil's literary register. Both use formal, polished Latin, while Catullus often employs colloquial expressions and personal subject matter.

### Most Discriminative Features
Top character n-grams that distinguish the poets:
- `"que"` / `"ue "` - Enclitic frequency (Vergil's epic style uses these heavily)
- `" et "` - Conjunction preferences
- `"um "` / `"is "` - Case ending patterns (dactylic hexameter vs. lyric meters)
- `"nt "` - Verb ending frequency (3rd person plural patterns)

---

## How to use

### Installation
```bash
git clone https://github.com/yourusername/latin-ML.git
cd latin-ML
pip install -r requirements.txt
```

Run the demo

```bash
cd src
python demo.py
```

**Example:**
```bash
Enter text to analyze: Arma virumque cano, Troiae qui primus ab oris

Predicted Author: VERGIL

Confidence:
  Vergil: 79.3%
  Catullus: 12.1%
  Horace: 8.6%
```

## Technical Specifications

**Model:** Logistic Regression (C=0.5 for regularization)

**Feature Engineering:** TF-IDF character n-grams (3-5 chars, 2000 max features)

**Data:** ~482 lines per author from The Latin Library, balanced and chunked into 3-line samples

**Training Split:** 80/20 train/test, stratified by author

**Why Logistic Regression?**
- Outperformed Random Forest (lower overfitting)
- Interpretable coefficients reveal which linguistic patterns matter


---

## Data Source

All texts collected from [The Latin Library](http://www.thelatinlibrary.com/)
- Vergil: *Aeneid* (epic poetry, dactylic hexameter)
- Horace: *Odes*, *Satires* (lyric & satirical verse)
- Catullus: Various (elegiac, hendecasyllabic, personal poems)

---

## Documentation

- **[build_log.md](build_log.md)** - Detailed development journal with decision rationale
- **Visualizations** - Auto-generated charts showing model performance and linguistic insights


