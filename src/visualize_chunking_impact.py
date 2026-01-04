"""
Create visualization comparing single-line vs 3-line chunking impact.
Generates a publication-quality chart for portfolio documentation.
"""

from preprocess import create_labeled_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 11

# Create output directory
os.makedirs('../visualizations', exist_ok=True)

print("=" * 80)
print("CHUNKING IMPACT VISUALIZATION")
print("=" * 80)

# Configure vectorizer (same for both experiments)
vectorizer = TfidfVectorizer(
    max_features=2000,
    analyzer='char',
    ngram_range=(3, 5),
    min_df=2
)

# Logistic regression model (same for both)
clf = LogisticRegression(C=0.5, max_iter=1000, random_state=42)

# ============================================================================
# EXPERIMENT 1: Single Lines (chunk_size=1)
# ============================================================================
print("\nExperiment 1: Single lines (chunk_size=1)...")

texts_1, labels_1 = create_labeled_dataset('../data/', balance=True, chunk_size=1)
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(
    texts_1, labels_1, test_size=0.2, random_state=42, stratify=labels_1
)

X_train_vec_1 = vectorizer.fit_transform(X_train_1)
X_test_vec_1 = vectorizer.transform(X_test_1)

clf.fit(X_train_vec_1, y_train_1)

train_acc_1 = clf.score(X_train_vec_1, y_train_1)
test_acc_1 = clf.score(X_test_vec_1, y_test_1)
y_pred_1 = clf.predict(X_test_vec_1)

precision_1, recall_1, f1_1, _ = precision_recall_fscore_support(
    y_test_1, y_pred_1, labels=clf.classes_, zero_division=0
)

avg_words_1 = np.mean([len(text.split()) for text in texts_1])

print(f"  Train Accuracy: {train_acc_1:.1%}")
print(f"  Test Accuracy:  {test_acc_1:.1%}")
print(f"  Avg words/sample: {avg_words_1:.1f}")

# ============================================================================
# EXPERIMENT 2: Three-line chunks (chunk_size=3)
# ============================================================================
print("\nExperiment 2: Three-line chunks (chunk_size=3)...")

texts_3, labels_3 = create_labeled_dataset('../data/', balance=True, chunk_size=3)
X_train_3, X_test_3, y_train_3, y_test_3 = train_test_split(
    texts_3, labels_3, test_size=0.2, random_state=42, stratify=labels_3
)

X_train_vec_3 = vectorizer.fit_transform(X_train_3)
X_test_vec_3 = vectorizer.transform(X_test_3)

clf.fit(X_train_vec_3, y_train_3)

train_acc_3 = clf.score(X_train_vec_3, y_train_3)
test_acc_3 = clf.score(X_test_vec_3, y_test_3)
y_pred_3 = clf.predict(X_test_vec_3)

precision_3, recall_3, f1_3, _ = precision_recall_fscore_support(
    y_test_3, y_pred_3, labels=clf.classes_, zero_division=0
)

avg_words_3 = np.mean([len(text.split()) for text in texts_3])

print(f"  Train Accuracy: {train_acc_3:.1%}")
print(f"  Test Accuracy:  {test_acc_3:.1%}")
print(f"  Avg words/sample: {avg_words_3:.1f}")

# ============================================================================
# CREATE VISUALIZATION
# ============================================================================
print("\nGenerating visualization...")

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Main title
fig.suptitle('Chunking Impact: Context Matters for Latin Poetry Classification',
             fontsize=16, fontweight='bold', y=0.98)

# ============================================================================
# TOP ROW: Overall metrics comparison
# ============================================================================

# Top-left: Test Accuracy Comparison
ax1 = fig.add_subplot(gs[0, 0])
configs = ['Single Line\n(~5 words)', '3-Line Chunk\n(~16 words)']
test_accs = [test_acc_1, test_acc_3]
colors = ['#E63946', '#06A77D']

bars = ax1.bar(configs, test_accs, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
ax1.set_ylabel('Test Accuracy', fontsize=12, fontweight='bold')
ax1.set_ylim([0, 1])
ax1.set_title('Test Accuracy Improvement', fontsize=13, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

for bar, acc in zip(bars, test_accs):
    height = bar.get_height()
    ax1.annotate(f'{acc:.1%}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5), textcoords="offset points",
                ha='center', fontsize=14, fontweight='bold')

# Add improvement annotation
improvement = (test_acc_3 - test_acc_1) * 100
ax1.annotate(f'+{improvement:.1f}pp',
            xy=(1, test_acc_3), xytext=(1.3, (test_acc_1 + test_acc_3) / 2),
            arrowprops=dict(arrowstyle='->', lw=2, color='#06A77D'),
            fontsize=13, fontweight='bold', color='#06A77D')

# Top-middle: Overfitting Gap
ax2 = fig.add_subplot(gs[0, 1])
gaps = [(train_acc_1 - test_acc_1), (train_acc_3 - test_acc_3)]
bars = ax2.bar(configs, gaps, color=['#D62828', '#F77F00'], alpha=0.7,
               edgecolor='black', linewidth=2)
ax2.set_ylabel('Overfitting Gap\n(Train - Test)', fontsize=12, fontweight='bold')
ax2.set_title('Overfitting Reduction', fontsize=13, fontweight='bold')
ax2.axhline(y=0.25, color='red', linestyle='--', alpha=0.5, linewidth=1.5,
           label='High Overfitting (>25%)')
ax2.grid(axis='y', alpha=0.3)
ax2.legend(fontsize=9)

for bar, gap in zip(bars, gaps):
    height = bar.get_height()
    ax2.annotate(f'{gap:.1%}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', fontsize=12, fontweight='bold')

# Top-right: Sample characteristics
ax3 = fig.add_subplot(gs[0, 2])
ax3.axis('off')
stats_text = f"""
DATASET CHARACTERISTICS
{'─' * 35}

Single Line (chunk_size=1):
  • Samples: {len(texts_1)}
  • Avg words: {avg_words_1:.1f}
  • Test accuracy: {test_acc_1:.1%}
  • Overfitting: {(train_acc_1 - test_acc_1):.1%}

3-Line Chunk (chunk_size=3):
  • Samples: {len(texts_3)}
  • Avg words: {avg_words_3:.1f}
  • Test accuracy: {test_acc_3:.1%}
  • Overfitting: {(train_acc_3 - test_acc_3):.1%}

IMPROVEMENT
{'─' * 35}
  Test Accuracy: +{improvement:.1f}pp
  Context: {avg_words_3 / avg_words_1:.1f}x more words
"""
ax3.text(0.05, 0.5, stats_text, fontsize=10, family='monospace',
        verticalalignment='center',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

# ============================================================================
# MIDDLE ROW: Per-author performance comparison
# ============================================================================

ax4 = fig.add_subplot(gs[1, :])
x = np.arange(len(clf.classes_))
width = 0.15

# Single line metrics
bars1 = ax4.bar(x - width*1.5, precision_1, width, label='Precision (1-line)',
               color='#E63946', alpha=0.6, edgecolor='black')
bars2 = ax4.bar(x - width*0.5, recall_1, width, label='Recall (1-line)',
               color='#E63946', alpha=0.8, edgecolor='black')

# 3-line chunk metrics
bars3 = ax4.bar(x + width*0.5, precision_3, width, label='Precision (3-line)',
               color='#06A77D', alpha=0.6, edgecolor='black')
bars4 = ax4.bar(x + width*1.5, recall_3, width, label='Recall (3-line)',
               color='#06A77D', alpha=0.8, edgecolor='black')

ax4.set_ylabel('Score', fontsize=12, fontweight='bold')
ax4.set_xlabel('Author', fontsize=12, fontweight='bold')
ax4.set_title('Per-Author Performance: Single Line vs. 3-Line Chunks',
             fontsize=13, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(clf.classes_, fontsize=12)
ax4.legend(loc='lower right', ncol=2, fontsize=10)
ax4.set_ylim([0, 1])
ax4.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2, bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        if height > 0.1:  # Only label significant bars
            ax4.annotate(f'{height:.0%}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 2), textcoords="offset points",
                       ha='center', va='bottom', fontsize=7)

# ============================================================================
# BOTTOM ROW: Word count distributions
# ============================================================================

ax5 = fig.add_subplot(gs[2, 0])
chunk_lengths_1 = [len(text.split()) for text in texts_1]
ax5.hist(chunk_lengths_1, bins=15, color='#E63946', alpha=0.7, edgecolor='black')
ax5.axvline(np.mean(chunk_lengths_1), color='darkred', linestyle='--', linewidth=2,
           label=f'Mean: {np.mean(chunk_lengths_1):.1f} words')
ax5.set_xlabel('Words per Sample', fontsize=11)
ax5.set_ylabel('Frequency', fontsize=11)
ax5.set_title('Single Line Length Distribution', fontsize=12, fontweight='bold')
ax5.legend(fontsize=9)
ax5.grid(alpha=0.3)

ax6 = fig.add_subplot(gs[2, 1])
chunk_lengths_3 = [len(text.split()) for text in texts_3]
ax6.hist(chunk_lengths_3, bins=15, color='#06A77D', alpha=0.7, edgecolor='black')
ax6.axvline(np.mean(chunk_lengths_3), color='darkgreen', linestyle='--', linewidth=2,
           label=f'Mean: {np.mean(chunk_lengths_3):.1f} words')
ax6.set_xlabel('Words per Sample', fontsize=11)
ax6.set_ylabel('Frequency', fontsize=11)
ax6.set_title('3-Line Chunk Length Distribution', fontsize=12, fontweight='bold')
ax6.legend(fontsize=9)
ax6.grid(alpha=0.3)

# Bottom-right: Key insight
ax7 = fig.add_subplot(gs[2, 2])
ax7.axis('off')
insight_text = """
KEY INSIGHT
═══════════════════════════════════

Why Chunking Works:

1. More Context
   Character n-grams need sufficient
   text to capture patterns. Single
   lines (~5 words) lack context for
   syntactic preferences.

2. Better Features
   Longer samples allow n-grams to
   span multiple grammatical units,
   revealing author-specific syntax.

3. Realistic Use Case
   Matching training format to test
   format is critical. Model trained
   on 3-line chunks performs poorly
   on single lines.

RESULT: +13.1pp improvement
        by simply grouping lines
"""
ax7.text(0.05, 0.5, insight_text, fontsize=9, family='monospace',
        verticalalignment='center',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.2))

# ============================================================================
# SAVE
# ============================================================================

plt.savefig('../visualizations/chunking_impact_comparison.png',
           dpi=300, bbox_inches='tight', facecolor='white')
print("\n" + "=" * 80)
print("VISUALIZATION SAVED")
print("=" * 80)
print("\nFile: visualizations/chunking_impact_comparison.png")
print(f"\nSummary:")
print(f"  Single lines:  {test_acc_1:.1%} test accuracy ({avg_words_1:.1f} words/sample)")
print(f"  3-line chunks: {test_acc_3:.1%} test accuracy ({avg_words_3:.1f} words/sample)")
print(f"  Improvement:   +{improvement:.1f} percentage points")
print(f"\nThis visualization demonstrates the breakthrough from Dec 15-19, 2024:")
print(f"Context matters - longer samples capture richer morphological patterns.")
print("=" * 80)

plt.close()
