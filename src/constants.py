DB_TEST_PATH = "database/birds-525-species-image-classification/data/test-00000-of-00001.parquet"
DB_VALIDATION_PATH = "database/birds-525-species-image-classification/data/validation-00000-of-00001.parquet"
DB_TRAIN_PATHS = (
    "database/birds-525-species-image-classification/data/train-00000-of-00004.parquet",
    "database/birds-525-species-image-classification/data/train-00001-of-00004.parquet",
    "database/birds-525-species-image-classification/data/train-00002-of-00004.parquet",
    "database/birds-525-species-image-classification/data/train-00003-of-00004.parquet"
)
README_PATH = "database/birds-525-species-image-classification/README.md"

# Path of the class label names in the README.md nested structure
#   +------------------------+
#   | dataset_info:          |
#   |   features:            |
#   |     name: label        |
#   |       dtype:           |
#   |         class_label:   |
#   |           names:       |
#   |             '0': $NAME |
#   |             '1': $NAME |
#   |             ...        |
#   +------------------------+
LABEL_NAME_PATH = (
    "dataset_info:",
    "features:",
    "name: label",
    "dtype:",
    "class_label:",
    "names:"
)

NUM_OF_CLASSES = 526
