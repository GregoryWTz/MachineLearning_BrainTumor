import os
import cv2
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# ── Label extraction from filename ──────────────────────────────────────────
def get_label(filename):
    """
    Returns (binary_label, class_name) based on filename prefix.
    Te-gl  → glioma     → tumor (1)
    Te-no  → no tumor   → no tumor (0)
    Tr-no  → no tumor   → no tumor (0)
    Tr-aug-me → meningioma → tumor (1)
    Tr-me  → meningioma → tumor (1)
    Tr-pi  → pituitary  → tumor (1)
    Te-pi  → pituitary  → tumor (1)
    Te-me  → meningioma → tumor (1)
    """
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
        return None, None  # unrecognized, will be skipped

# ── Config ───────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
data_dir  = os.path.join(BASE_DIR, 'dataset')   # ← single flat folder
model_dir = os.path.join(BASE_DIR, 'models')
os.makedirs(model_dir, exist_ok=True)

IMG_SIZE  = 64
TEST_SIZE = 0.30   # 70 / 30 split

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

# ── Train SVM ────────────────────────────────────────────────────────────────
print("\n🧠 Training SVM (this may take a while)...")
model = SVC(kernel='rbf', probability=True, C=1.0)
model.fit(X_train, y_train)
print("✅ Training complete!")

# ── Evaluate ─────────────────────────────────────────────────────────────────
y_pred    = model.predict(X_test)
accuracy  = accuracy_score(y_test, y_pred)

print(f"\n🎯 SVM Test Accuracy: {accuracy * 100:.2f}%")
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['No Tumor', 'Tumor']))

# ── Save model ────────────────────────────────────────────────────────────────
model_path = os.path.join(model_dir, 'svm_brain_tumor.pkl')
joblib.dump(model, model_path)
print(f"\n💾 Model saved → {model_path}")

# ── Graphs ────────────────────────────────────────────────────────────────────
# fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# fig.suptitle('Brain Tumor Detection — SVM Results', fontsize=15, fontweight='bold')

# ── Graphs (Confusion Matrix Only) ──────────────────────────────────────────
plt.figure(figsize=(8, 6))

# 1) Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=['No Tumor', 'Tumor'],
    yticklabels=['No Tumor', 'Tumor'],
    # ax=axes[0]
)
# axes[0].set_title('Confusion Matrix')
# axes[0].set_xlabel('Predicted')
# axes[0].set_ylabel('Actual')

plt.title('Confusion Matrix: Brain Tumor Detection (SVM)', fontsize=14, pad=20, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('Actual Label', fontsize=12)

# 2) Class distribution bar chart
# labels_display = ['No Tumor', 'Tumor']
# train_counts   = [(y_train == 0).sum(), (y_train == 1).sum()]
# test_counts    = [(y_test  == 0).sum(), (y_test  == 1).sum()]

# x = np.arange(len(labels_display))
# width = 0.35
# axes[1].bar(x - width/2, train_counts, width, label='Train', color='steelblue')
# axes[1].bar(x + width/2, test_counts,  width, label='Test',  color='coral')
# axes[1].set_xticks(x)
# axes[1].set_xticklabels(labels_display)
# axes[1].set_title('Class Distribution (Train vs Test)')
# axes[1].set_ylabel('Number of Images')
# axes[1].legend()
# axes[1].bar_label(axes[1].containers[0], padding=3)
# axes[1].bar_label(axes[1].containers[1], padding=3)

plt.tight_layout()
graph_path = os.path.join(BASE_DIR, 'result_svm.png')
plt.savefig(graph_path, dpi=150)
plt.show()
print(f"📈 Graph saved → {graph_path}")