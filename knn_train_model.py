import os
import cv2
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# ── Label extraction from filename ──────────────────────────────────────────
def get_label(filename):
    """
    Returns (binary_label, class_name) based on filename prefix.
    Matches your previous SVM dataset structure.
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
        return None, None

# ── Config ───────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
data_dir  = os.path.join(BASE_DIR, 'dataset') 
model_dir = os.path.join(BASE_DIR, 'models')
os.makedirs(model_dir, exist_ok=True)

IMG_SIZE  = 64    # Kept at 64 for speed; kNN is slow with large images
TEST_SIZE = 0.30 

# ── Load images ──────────────────────────────────────────────────────────────
X, y = [], []
all_files = [f for f in os.listdir(data_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

print(f"📂 Found {len(all_files)} image files. Processing...")

for fname in all_files:
    label, _ = get_label(fname)
    if label is None: continue

    img_path = os.path.join(data_dir, fname)
    img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img_array is None: continue

    resized = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    X.append(resized.flatten())
    y.append(label)

X = np.array(X) / 255.0
y = np.array(y)

# ── Train / Test split ───────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=42, stratify=y
)

# ── Train kNN ────────────────────────────────────────────────────────────────
print(f"\n🧠 Training kNN (k=3)...")
# Note: kNN doesn't 'train' in the traditional sense, it just stores the data.
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# ── Evaluate ─────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"🎯 kNN Test Accuracy: {accuracy * 100:.2f}%")
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['No Tumor', 'Tumor']))

# ── Save model ────────────────────────────────────────────────────────────────
model_path = os.path.join(model_dir, 'knn_brain_tumor.pkl')
joblib.dump(model, model_path)

# ── Graph (Confusion Matrix Only) ───────────────────────────────────────────
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Greens', # Changed color to Green to distinguish from SVM
    xticklabels=['No Tumor', 'Tumor'],
    yticklabels=['No Tumor', 'Tumor']
)
plt.title('Confusion Matrix: Brain Tumor Detection (kNN)', fontsize=14, pad=20, fontweight='bold')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.tight_layout()
graph_path = os.path.join(BASE_DIR, 'result_knn.png')
plt.savefig(graph_path, dpi=150)
plt.show()

print(f"📈 Graph saved → {graph_path}")