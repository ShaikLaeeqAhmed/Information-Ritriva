# ---------------------------------------------------------------------------------
# IT Support Ticket Classifier: Multi-Class Text Classification Benchmark
# Compares Decision Tree, K-Nearest Neighbors, Linear SVM, and Naive Bayes
# ---------------------------------------------------------------------------------

import re
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# ---------------------------------------------------------------------------------
# Step 1: Text Preprocessing Functions
# ---------------------------------------------------------------------------------
STOP_WORDS = set([
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", 
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", 
    "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", 
    "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him", 
    "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "me", "more", 
    "most", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", 
    "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "should", "so", 
    "some", "such", "than", "that", "the", "their", "theirs", "them", "themselves", "then", 
    "there", "these", "they", "this", "those", "through", "to", "too", "under", "until", "up", 
    "very", "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom", 
    "why", "with", "would", "you", "your", "yours", "yourself", "yourselves"
])

def simple_stem(word):
    """Basic suffix stripping for text normalization"""
    if word.endswith('ing') and len(word) > 5:
        return word[:-3]
    if word.endswith('ed') and len(word) > 4:
        return word[:-2]
    if word.endswith('es') and len(word) > 4:
        return word[:-2]
    if word.endswith('s') and len(word) > 3:
        return word[:-1]
    return word

def preprocess_text(text):
    # 1. Lowercase
    text = text.lower()
    # 2. Remove punctuation and keep alphabetic words
    words = re.findall(r'\b[a-z]{2,}\b', text)
    # 3. Remove stop words
    filtered = [w for w in words if w not in STOP_WORDS]
    # 4. Stem words
    stemmed = [simple_stem(w) for w in filtered]
    return " ".join(stemmed)

# ---------------------------------------------------------------------------------
# Step 2: Load Dataset & Explore
# ---------------------------------------------------------------------------------
print("=" * 65)
print("  IT Support Ticket Classification: Multi-Model Benchmark")
print("=" * 65)

df = pd.read_csv("dataset.csv")
print(f"Total dataset samples: {len(df)}")
print("\nClass distribution:")
print(df['category'].value_counts())

# Show a before-and-after preprocessing sample
sample_raw = df.iloc[0]['text']
sample_clean = preprocess_text(sample_raw)
print("\n--- Preprocessing Demonstration ---")
print(f"Raw Text    : {sample_raw}")
print(f"Cleaned Text: {sample_clean}")

df['cleaned_text'] = df['text'].apply(preprocess_text)

# ---------------------------------------------------------------------------------
# Step 3: Train-Test Split & TF-IDF Vectorization
# ---------------------------------------------------------------------------------
X = df['cleaned_text']
y = df['category']

# Consistent 80/20 train-test split with fixed random seed
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"\nTraining set size: {len(X_train)} samples")
print(f"Testing set size : {len(X_test)} samples")

# Feature representation: TF-IDF
tfidf = TfidfVectorizer()
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)
print(f"Total extracted TF-IDF features: {X_train_tfidf.shape[1]}")

# ---------------------------------------------------------------------------------
# Step 4: Implement & Evaluate 4 Classifiers
# ---------------------------------------------------------------------------------
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN (k=3)": KNeighborsClassifier(n_neighbors=3),
    "SVM (Linear)": SVC(kernel='linear', random_state=42),
    "Naïve Bayes": MultinomialNB()
}

results = []
cms = {}
labels = sorted(y.unique())

for name, model in models.items():
    start_time = time.time()
    model.fit(X_train_tfidf, y_train)
    train_time = (time.time() - start_time) * 1000 # in milliseconds

    y_pred = model.predict(X_test_tfidf)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    
    cms[name] = confusion_matrix(y_test, y_pred, labels=labels)
    
    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "Train Time (ms)": round(train_time, 2)
    })

# Convert to DataFrame and print comparison table
results_df = pd.DataFrame(results)
print("\n" + "=" * 65)
print("  MODEL PERFORMANCE COMPARISON")
print("=" * 65)
print(results_df.to_string(index=False))

# Detailed report for best model (SVM)
best_model = models["SVM (Linear)"]
best_pred = best_model.predict(X_test_tfidf)
print("\n--- Detailed Report for Best Model: SVM (Linear) ---")
print(classification_report(y_test, best_pred, target_names=labels))

# ---------------------------------------------------------------------------------
# Step 5: Plotting Confusion Matrices and Comparison Chart
# ---------------------------------------------------------------------------------
# 1. Confusion Matrix Plot (all 4 models)
fig, axes = plt.subplots(2, 2, figsize=(11, 9))
axes = axes.flatten()

for i, (name, cm) in enumerate(cms.items()):
    ax = axes[i]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax, cbar=False)
    ax.set_title(f"{name}", fontsize=12, fontweight='bold')
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

plt.suptitle("Confusion Matrix Comparison Across 4 Classifiers", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=200)
plt.close()
print("\nSaved confusion matrix plot as 'confusion_matrix.png'")

# 2. Performance Comparison Bar Chart
plt.figure(figsize=(8, 5))
bar_width = 0.35
index = np.arange(len(results_df))

plt.bar(index, results_df['Accuracy'] * 100, bar_width, label='Accuracy (%)', color='#4C72B0')
plt.bar(index + bar_width, results_df['F1-Score'] * 100, bar_width, label='F1-Score (%)', color='#55A868')

plt.xlabel('Classifier', fontweight='bold')
plt.ylabel('Score (%)', fontweight='bold')
plt.title('Comparison of 4 Classifiers on IT Support Dataset', fontsize=12, fontweight='bold')
plt.xticks(index + bar_width / 2, results_df['Model'])
plt.ylim(0, 110)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.6)

for i in range(len(results_df)):
    plt.text(i - 0.05, results_df['Accuracy'][i] * 100 + 2, f"{results_df['Accuracy'][i]*100:.1f}%", fontsize=9, fontweight='bold')
    plt.text(i + bar_width - 0.05, results_df['F1-Score'][i] * 100 + 2, f"{results_df['F1-Score'][i]*100:.1f}%", fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig("model_comparison.png", dpi=200)
plt.close()
print("Saved model comparison chart as 'model_comparison.png'")

# ---------------------------------------------------------------------------------
# Step 6: Hyperparameter & Feature Representation Analysis
# ---------------------------------------------------------------------------------
print("\n" + "=" * 65)
print("  HYPERPARAMETER & FEATURE REPRESENTATION ANALYSIS")
print("=" * 65)

# What-If 1: Changing k value in KNN (k = 1, 3, 5, 7, 9)
print("What-If Experiment 1: Testing different k values in KNN...")
k_values = [1, 3, 5, 7, 9]
knn_accs = []

for k in k_values:
    test_knn = KNeighborsClassifier(n_neighbors=k)
    test_knn.fit(X_train_tfidf, y_train)
    k_pred = test_knn.predict(X_test_tfidf)
    k_acc = accuracy_score(y_test, k_pred)
    knn_accs.append(k_acc)
    print(f"  k = {k} -> Accuracy: {k_acc * 100:.1f}%")

# Save KNN what-if plot
plt.figure(figsize=(7, 4))
plt.plot(k_values, [a * 100 for a in knn_accs], marker='o', color='#C44E52', linewidth=2)
plt.title("What-If Analysis: Effect of k on KNN Accuracy", fontsize=11, fontweight='bold')
plt.xlabel("k value (Number of Neighbors)", fontweight='bold')
plt.ylabel("Accuracy (%)", fontweight='bold')
plt.xticks(k_values)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("what_if_knn.png", dpi=200)
plt.close()
print("Saved what-if KNN plot as 'what_if_knn.png'")

# What-If 2: Changing feature representation (Unigram vs Bigram)
print("\nWhat-If Experiment 2: Unigram vs Unigram + Bigram feature representation...")
tfidf_bigram = TfidfVectorizer(ngram_range=(1, 2))
X_train_bigram = tfidf_bigram.fit_transform(X_train)
X_test_bigram = tfidf_bigram.transform(X_test)

svm_uni = SVC(kernel='linear', random_state=42)
svm_uni.fit(X_train_tfidf, y_train)
acc_uni = accuracy_score(y_test, svm_uni.predict(X_test_tfidf))

svm_bi = SVC(kernel='linear', random_state=42)
svm_bi.fit(X_train_bigram, y_train)
acc_bi = accuracy_score(y_test, svm_bi.predict(X_test_bigram))

print(f"  Unigram (1,1) TF-IDF features: {X_train_tfidf.shape[1]} -> SVM Accuracy: {acc_uni * 100:.1f}%")
print(f"  Bigram (1,2)  TF-IDF features: {X_train_bigram.shape[1]} -> SVM Accuracy: {acc_bi * 100:.1f}%")
print("\nExperiment completed successfully!")

