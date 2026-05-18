import os
import cv2
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import roc_curve, auc
import time

# ── Label extraction from filename ──────────────────────────────────────────
def get_label(filename):
    name = filename.lower()
    if 'no_' in name:
        return 0, 'No Tumor'
    elif 'gl_' in name:
        return 1, 'Glioma'
    elif 'me_' in name:
        return 1, 'Meningioma'
    elif 'pi_' in name:
        return 1, 'Pituitary'
    else:
        return None, None

# ── Config ───────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
data_dir  = os.path.join(BASE_DIR, 'dataset')
model_dir = os.path.join(BASE_DIR, 'models')
os.makedirs(model_dir, exist_ok=True)

IMG_SIZE  = 64
TEST_SIZE = 0.30

# ── Load images ──────────────────────────────────────────────────────────────
X, y, class_names_list = [], [], []

all_files = [f for f in os.listdir(data_dir)
             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

print(f"📂 Found {len(all_files)} image files in dataset/")

for i, fname in enumerate(all_files, 1):
    label, class_name = get_label(fname)
    if label is None:
        print(f"  ⚠ Skipped (unrecognized name): {fname}")
        continue

    img_path  = os.path.join(data_dir, fname)
    img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img_array is None:
        continue

    resized = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    X.append(resized.flatten())
    y.append(label)
    class_names_list.append(class_name)

    if i % 200 == 0 or i == len(all_files):
        print(f"   Loaded {i}/{len(all_files)} ({i/len(all_files)*100:.1f}%)")

X = np.array(X) / 255.0
y = np.array(y)

print(f"\n✅ Dataset loaded — {len(X)} usable images")
print(f"   Tumor: {y.sum()}  |  No Tumor: {(y==0).sum()}")

# ── Train / Test split ───────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=42, stratify=y
)

print(f"\n📊 Split → Train: {len(X_train)}  |  Test: {len(X_test)}")

# ── Train SVM — search for best C ────────────────────────────────────────────
print("\n🧠 Training SVM")
C_values = [0.1, 1, 10]
accuracies = []

for c in C_values:
    svm = SVC(kernel='rbf', C=c, probability=True)
    svm.fit(X_train, y_train)
    y_pred_c = svm.predict(X_test)
    acc = accuracy_score(y_test, y_pred_c)
    print(f"C={c}, Accuracy={acc:.4f}")
    accuracies.append(acc)

plt.figure(figsize=(6, 4))
plt.plot(C_values, accuracies, marker='o', color='steelblue')
plt.title("SVM Accuracy vs C Value", fontweight='bold')
plt.xlabel("C")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'result_svm_c_search.png'), dpi=150)
plt.show()

best_C = C_values[accuracies.index(max(accuracies))]
print(f"\n🏆 Best C: {best_C}")

# ── Train final model ─────────────────────────────────────────────────────────
model = SVC(C=best_C, kernel='rbf', probability=True, class_weight='balanced')

start = time.time()
model.fit(X_train, y_train)
end = time.time()
print(f"Training Time: {end - start:.2f} seconds")

# ── Evaluate ──────────────────────────────────────────────────────────────────
start = time.time()
y_pred = model.predict(X_test)
end = time.time()
print(f"Prediction Time: {end - start:.2f} seconds")

accuracy = accuracy_score(y_test, y_pred)
print(f"\n🎯 SVM Test Accuracy: {accuracy * 100:.2f}%")
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['No Tumor', 'Tumor']))

# ── Print sensitivity & specificity ──────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
print(f"🔬 Sensitivity (Recall for Tumor):   {sensitivity * 100:.2f}%")
print(f"🔬 Specificity (Recall for No Tumor): {specificity * 100:.2f}%")
print(f"🔬 False Negatives (missed tumors):   {fn}")
print(f"🔬 False Positives (false alarms):    {fp}")

# ── Save model ────────────────────────────────────────────────────────────────
model_path = os.path.join(model_dir, 'svm_brain_tumor.pkl')
joblib.dump(model, model_path)
print(f"\n💾 Model saved → {model_path}")

# ── Confusion Matrix ──────────────────────────────────────────────────────────
plt.figure(figsize=(8, 6))

cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
labels = np.array([[f"{v}\n({p:.1f}%)" for v, p in zip(row_v, row_p)]
                   for row_v, row_p in zip(cm, cm_percent)])

sns.heatmap(cm, annot=labels, fmt='', cmap='Blues',
            xticklabels=['No Tumor', 'Tumor'],
            yticklabels=['No Tumor', 'Tumor'])

plt.title('Confusion Matrix: Brain Tumor Detection (SVM)', fontsize=14, pad=20, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('Actual Label', fontsize=12)
plt.tight_layout()

graph_path = os.path.join(BASE_DIR, 'result_svm.png')
plt.savefig(graph_path, dpi=150)
plt.show()
print(f"📈 Confusion matrix saved → {graph_path}")

# ── ROC Curve ─────────────────────────────────────────────────────────────────
y_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--', label="Random Classifier")
plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)
plt.title("ROC Curve — Brain Tumor Detection (SVM)", fontsize=14, fontweight='bold')
plt.legend(loc="lower right")
plt.tight_layout()

roc_path = os.path.join(BASE_DIR, 'result_roc_svm.png')
plt.savefig(roc_path, dpi=150)
plt.show()
print(f"📈 ROC Curve saved → {roc_path}")