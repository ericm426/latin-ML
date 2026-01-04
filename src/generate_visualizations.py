"""
Generate publication-quality visualizations for portfolio documentation.
Creates: confusion matrix, feature importance, accuracy progression charts.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from preprocess import create_labeled_dataset
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['font.size'] = 11

# Create output directory
os.makedirs('../visualizations', exist_ok=True)

print("Loading model and data...")
clf = pickle.load(open('../models/classifier.pkl', 'rb'))
vectorizer = pickle.load(open('../models/vectorizer.pkl', 'rb'))
metadata = pickle.load(open('../models/metadata.pkl', 'rb'))

# Recreate test set
texts, labels = create_labeled_dataset('../data/', balance=True, chunk_size=3)
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)
X_test_vec = vectorizer.transform(X_test)
y_pred = clf.predict(X_test_vec)

# 1. CONFUSION MATRIX

print("\n1. Generating confusion matrix...")

plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=clf.classes_, yticklabels=clf.classes_,
            cbar_kws={'label': 'Number of Samples'})
plt.title('Confusion Matrix: Latin Poetry Classifier\n75.5% Test Accuracy',
          fontsize=14, fontweight='bold')
plt.ylabel('True Author', fontsize=12)
plt.xlabel('Predicted Author', fontsize=12)
plt.tight_layout()
plt.savefig('../visualizations/confusion_matrix.png', dpi=300, bbox_inches='tight')
print("   Saved: visualizations/confusion_matrix.png")
plt.close()

# ============================================================================
# 2. FEATURE IMPORTANCE (Top 15 Character N-grams)
# ============================================================================
print("2. Generating feature importance chart...")

feature_names = vectorizer.get_feature_names_out()
importances = np.abs(clf.coef_).sum(axis=0)
top_n = 15
top_features_idx = np.argsort(importances)[-top_n:]

plt.figure(figsize=(10, 6))
y_pos = np.arange(top_n)
plt.barh(y_pos, importances[top_features_idx], color='steelblue')
plt.yticks(y_pos, [f'"{feature_names[idx]}"' for idx in top_features_idx])
plt.xlabel('Importance Score (Sum of Absolute Coefficients)', fontsize=12)
plt.title('Top 15 Most Discriminative Character N-grams\n'
          'Key Morphological Patterns in Latin Poetry',
          fontsize=14, fontweight='bold')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('../visualizations/feature_importance.png', dpi=300, bbox_inches='tight')
print("   Saved: visualizations/feature_importance.png")
plt.close()

# Print linguistic insights
print("\n   Linguistic Insights:")
print("   Top features capture:")
for idx in reversed(top_features_idx[-5:]):
    feature = feature_names[idx]
    if 'que' in feature or 'ue' in feature:
        print(f"     - '{feature}': Latin enclitic patterns")
    elif any(end in feature for end in ['um ', 'is ', 'us ', 'ae ']):
        print(f"     - '{feature}': Case/declension endings")
    else:
        print(f"     - '{feature}': Author-specific pattern")

# ============================================================================
# 3. ACCURACY PROGRESSION OVER DEVELOPMENT
# ============================================================================
print("3. Generating accuracy progression chart...")

# Based on build_log.md
stages = ['Initial\nWord-based', 'Character\nN-grams', 'Line\nChunking', 'Final\nTuning']
test_accuracies = [0.390, 0.624, 0.745, 0.755]
train_accuracies = [1.000, 1.000, 0.970, 0.975]
dates = ['Dec 3-7', 'Dec 7-15', 'Dec 15-19', 'Dec 27']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left plot: Test accuracy over time
ax1.plot(stages, test_accuracies, marker='o', linewidth=2.5,
         markersize=10, color='#2E86AB', label='Test Accuracy')
ax1.fill_between(range(len(stages)), test_accuracies, alpha=0.3, color='#2E86AB')
ax1.set_ylim([0.3, 1.0])
ax1.set_ylabel('Test Accuracy', fontsize=12, fontweight='bold')
ax1.set_xlabel('Development Stage', fontsize=12)
ax1.set_title('Model Improvement Over Time', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='lower right')

# Add percentage labels
for i, (stage, acc) in enumerate(zip(stages, test_accuracies)):
    ax1.annotate(f'{acc:.1%}', (i, acc), textcoords="offset points",
                xytext=(0,10), ha='center', fontweight='bold')

# Right plot: Overfitting gap reduction
gaps = [train - test for train, test in zip(train_accuracies, test_accuracies)]
colors = ['#D62828' if gap > 0.3 else '#F77F00' if gap > 0.2 else '#06A77D'
          for gap in gaps]
bars = ax2.bar(stages, gaps, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax2.set_ylabel('Overfitting Gap (Train - Test)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Development Stage', fontsize=12)
ax2.set_title('Overfitting Reduction', fontsize=13, fontweight='bold')
ax2.axhline(y=0.2, color='green', linestyle='--', alpha=0.5, label='Target (<20%)')
ax2.grid(True, alpha=0.3, axis='y')
ax2.legend()

# Add percentage labels
for bar, gap in zip(bars, gaps):
    height = bar.get_height()
    ax2.annotate(f'{gap:.1%}', xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', fontweight='bold')

plt.suptitle('Latin Poetry Classifier: Development Journey',
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('../visualizations/accuracy_progression.png', dpi=300, bbox_inches='tight')
print("   Saved: visualizations/accuracy_progression.png")
plt.close()

# ============================================================================
# 4. PER-AUTHOR PERFORMANCE BREAKDOWN
# ============================================================================
print("4. Generating per-author performance chart...")

from sklearn.metrics import precision_recall_fscore_support

precision, recall, f1, support = precision_recall_fscore_support(
    y_test, y_pred, labels=clf.classes_, zero_division=0
)

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(clf.classes_))
width = 0.25

bars1 = ax.bar(x - width, precision, width, label='Precision', color='#2E86AB')
bars2 = ax.bar(x, recall, width, label='Recall', color='#06A77D')
bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='#F77F00')

ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_xlabel('Author', fontsize=12)
ax.set_title('Per-Author Classification Performance\nTest Set Results',
             fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(clf.classes_, fontsize=12)
ax.legend(loc='lower right')
ax.set_ylim([0, 1])
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.0%}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('../visualizations/per_author_performance.png', dpi=300, bbox_inches='tight')
print("   Saved: visualizations/per_author_performance.png")
plt.close()

# ============================================================================
# 5. SUMMARY STATISTICS FIGURE
# ============================================================================
print("5. Generating summary statistics...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Latin Poetry Classifier: Model Summary', fontsize=16, fontweight='bold')

# Top-left: Key metrics
ax = axes[0, 0]
ax.axis('off')
metrics_text = f"""
MODEL CONFIGURATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Algorithm:        Logistic Regression
Vectorizer:       TF-IDF (Character N-grams)
N-gram Range:     {metadata['ngram_range']}
Max Features:     {metadata['n_features']}
Regularization:   C=0.5
Chunk Size:       {metadata['chunk_size']} lines

PERFORMANCE METRICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Train Accuracy:   {metadata['train_accuracy']:.1%}
Test Accuracy:    {metadata['test_accuracy']:.1%}
Overfitting Gap:  {metadata['train_accuracy'] - metadata['test_accuracy']:.1%}

AUTHORS CLASSIFIED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{', '.join(metadata['authors'])}
"""
ax.text(0.1, 0.5, metrics_text, fontsize=11, family='monospace',
        verticalalignment='center')

# Top-right: Sample size info
ax = axes[0, 1]
from collections import Counter
label_counts = Counter(labels)
ax.barh(list(label_counts.keys()), list(label_counts.values()), color='steelblue')
ax.set_xlabel('Number of Chunks', fontsize=11)
ax.set_title('Dataset Distribution\n(Balanced)', fontsize=12, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

# Bottom-left: Classification report heatmap
ax = axes[1, 0]
report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
metrics = ['precision', 'recall', 'f1-score']
data = [[report_dict[author][metric] for metric in metrics]
        for author in clf.classes_]
sns.heatmap(data, annot=True, fmt='.2f', cmap='RdYlGn', vmin=0, vmax=1,
            xticklabels=metrics, yticklabels=clf.classes_, ax=ax, cbar=False)
ax.set_title('Classification Metrics by Author', fontsize=12, fontweight='bold')

# Bottom-right: Word count distribution
ax = axes[1, 1]
chunk_lengths = [len(text.split()) for text in texts]
ax.hist(chunk_lengths, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
ax.axvline(np.mean(chunk_lengths), color='red', linestyle='--', linewidth=2,
           label=f'Mean: {np.mean(chunk_lengths):.1f} words')
ax.set_xlabel('Words per Chunk', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.set_title('Input Length Distribution', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('../visualizations/model_summary.png', dpi=300, bbox_inches='tight')
print("   Saved: visualizations/model_summary.png")
plt.close()

print("\n" + "="*80)
print("ALL VISUALIZATIONS GENERATED SUCCESSFULLY")
print("="*80)
print("\nFiles created in visualizations/ directory:")
print("  1. confusion_matrix.png")
print("  2. feature_importance.png")
print("  3. accuracy_progression.png")
print("  4. per_author_performance.png")
print("  5. model_summary.png")
print("\nThese visualizations are ready for your MIT Maker Portfolio!")
print("="*80)
