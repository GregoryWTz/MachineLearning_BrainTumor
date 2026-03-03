import os
import sys
import cv2
import numpy as np
import joblib

IMG_SIZE = 64
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def predict(image_path: str):
    model_path = os.path.join(BASE_DIR, 'models', 'svm_brain_tumor.pkl')
    if not os.path.exists(model_path):
        print("❌ Model not found. Run train_model.py first.")
        return

    model = joblib.load(model_path)

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"❌ Could not read image: {image_path}")
        return

    resized   = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    flattened = resized.flatten().reshape(1, -1) / 255.0

    prediction  = model.predict(flattened)[0]
    probability = model.predict_proba(flattened)[0]

    label = "🔴 TUMOR DETECTED" if prediction == 1 else "🟢 NO TUMOR"
    conf  = probability[prediction] * 100

    print(f"\nResult     : {label}")
    print(f"Confidence : {conf:.2f}%")
    print(f"  No Tumor : {probability[0]*100:.2f}%")
    print(f"  Tumor    : {probability[1]*100:.2f}%")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_image>")
    else:
        input_arg = sys.argv[1]

        # If it's already a valid full path, use it directly
        if os.path.exists(input_arg):
            predict(input_arg)
        else:
            # Try searching in the dataset folder
            dataset_dir = os.path.join(BASE_DIR, 'dataset')
            
            # Try with and without extension
            candidates = [
                input_arg,                          # exact as typed
                input_arg + '.jpg',                 # add .jpg
                input_arg + '.jpeg',                # add .jpeg
                input_arg + '.png',                 # add .png
                os.path.join(dataset_dir, input_arg),
                os.path.join(dataset_dir, input_arg + '.jpg'),
                os.path.join(dataset_dir, input_arg + '.jpeg'),
                os.path.join(dataset_dir, input_arg + '.png'),
            ]

            found = None
            for c in candidates:
                if os.path.exists(c):
                    found = c
                    break

            if found:
                predict(found)
            else:
                print(f"❌ Could not find image '{input_arg}' in dataset/ or as a direct path.")