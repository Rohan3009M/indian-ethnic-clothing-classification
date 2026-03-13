import json
import random
import shutil
from pathlib import Path
from collections import defaultdict
import pandas as pd

# ============================
# CONFIG
# ============================

SEED = 42

SAMPLES_PER_CLASS = 500
TRAIN_SPLIT = 350
VAL_SPLIT = 75
TEST_SPLIT = 75

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATASET = PROJECT_ROOT / "data/raw/indofashion_dataset"
PROCESSED_DATASET = PROJECT_ROOT / "data/processed/dataset_subset"
INTERIM_DIR = PROJECT_ROOT / "data/interim"

TRAIN_JSON = RAW_DATASET / "train_data.json"
VAL_JSON = RAW_DATASET / "val_data.json"
TEST_JSON = RAW_DATASET / "test_data.json"

random.seed(SEED)

# ============================
# FUNCTIONS
# ============================


def load_jsonl(file_path):
    """Load JSONL metadata"""
    records = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    return records


def clean_class_name(name):
    """Make class folder safe"""
    name = name.lower().strip()
    name = name.replace(" ", "_")
    name = name.replace("&", "and")
    name = name.replace("-", "_")
    return name


def resolve_image_path(img_path):
    """Resolve actual image location"""
    return RAW_DATASET / img_path


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)


# ============================
# LOAD METADATA
# ============================

print("\nLoading metadata...")

records = []

for json_file in [TRAIN_JSON, VAL_JSON, TEST_JSON]:

    data = load_jsonl(json_file)

    print(f"{json_file.name}: {len(data)} records")

    for item in data:

        img_path = item["image_path"]
        class_label = item["class_label"]

        resolved = resolve_image_path(img_path)

        if not resolved.exists():
            continue

        records.append(
            {
                "class_name": clean_class_name(class_label),
                "image_path": resolved,
            }
        )

df = pd.DataFrame(records)

print("\nTotal valid images:", len(df))

# ============================
# GROUP BY CLASS
# ============================

grouped = defaultdict(list)

for row in df.to_dict("records"):
    grouped[row["class_name"]].append(row)

print("\nImages per class:")

for cls in grouped:
    print(cls, len(grouped[cls]))

# ============================
# SAMPLE DATASET
# ============================

sampled_rows = []

print("\nSampling dataset...")

for cls in grouped:

    images = grouped[cls]

    if len(images) < SAMPLES_PER_CLASS:
        raise ValueError(f"{cls} has less than {SAMPLES_PER_CLASS} images")

    random.shuffle(images)

    selected = images[:SAMPLES_PER_CLASS]

    train = selected[:TRAIN_SPLIT]
    val = selected[TRAIN_SPLIT:TRAIN_SPLIT + VAL_SPLIT]
    test = selected[TRAIN_SPLIT + VAL_SPLIT:]

    for r in train:
        r["split"] = "train"
        sampled_rows.append(r)

    for r in val:
        r["split"] = "val"
        sampled_rows.append(r)

    for r in test:
        r["split"] = "test"
        sampled_rows.append(r)

subset_df = pd.DataFrame(sampled_rows)

# ============================
# CREATE OUTPUT FOLDERS
# ============================

print("\nCreating folder structure...")

classes = subset_df["class_name"].unique()

for split in ["train", "val", "test"]:
    for cls in classes:
        ensure_dir(PROCESSED_DATASET / split / cls)

# ============================
# COPY FILES
# ============================

print("\nCopying images...")

count = 0

for row in subset_df.to_dict("records"):

    src = Path(row["image_path"])

    dst_dir = PROCESSED_DATASET / row["split"] / row["class_name"]

    dst = dst_dir / src.name

    shutil.copy(src, dst)

    count += 1

    if count % 500 == 0:
        print(f"{count} images copied")

print("\nTotal images copied:", count)

# ============================
# SAVE SUMMARY
# ============================

ensure_dir(INTERIM_DIR)

summary_file = INTERIM_DIR / "subset_summary.csv"

subset_df.to_csv(summary_file, index=False)

print("\nSummary saved:", summary_file)

# ============================
# FINAL STATS
# ============================

print("\nFinal distribution:\n")

print(subset_df.groupby(["split", "class_name"]).size())

print("\nDataset preparation completed successfully.")