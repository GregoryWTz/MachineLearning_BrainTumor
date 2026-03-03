import os
import re
from collections import defaultdict

FOLDER_PATH = "dataset"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}

# Remove Tr / Te
TR_TE_REGEX = re.compile(r"^(Tr|Te)[-_]?", re.IGNORECASE)

# Remove aug-
AUG_REGEX = re.compile(r"^aug[-_]?", re.IGNORECASE)

# Extract class name (before last _number)
CLASS_REGEX = re.compile(r"(.+?)_\d+$")

groups = defaultdict(list)

# Pass 1: normalize names and group by class
for filename in os.listdir(FOLDER_PATH):
    name, ext = os.path.splitext(filename)

    if ext.lower() not in IMAGE_EXTENSIONS:
        continue

    # Step 1: remove Tr / Te
    clean = TR_TE_REGEX.sub("", name)

    # Step 2: remove aug-
    clean = AUG_REGEX.sub("", clean)

    # Step 3: extract class
    match = CLASS_REGEX.match(clean)
    if not match:
        continue

    class_name = match.group(1)
    groups[class_name].append(filename)

# Pass 2: rename continuously
for class_name, files in groups.items():
    files.sort()  # reproducible ordering

    for idx, old_name in enumerate(files, start=1):
        ext = os.path.splitext(old_name)[1]
        new_name = f"{class_name}_{idx}{ext}"

        os.rename(
            os.path.join(FOLDER_PATH, old_name),
            os.path.join(FOLDER_PATH, new_name)
        )

print("✔ Tr / Te / aug removed, datasets merged, numbering continued cleanly.")