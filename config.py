import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SEED = 42

## MODEL ##
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 4
# CHECKPOINT_DIR = "./data/model"
# CHECKPOINT_NAME = "model.pth.tar"
# LOG_DIR = "./data/logs"
#
# ## DATASET ##
TRAIN_DATASET_SIZE = 1_000

# DATA_PATH = "./data/sets/{name}"
# IMAGES_PATH = "images/{id_}.jpg"
# ANNOTATIONS_PATH = "annotations/{id_}.txt"



