# """
# Defend against attribute and augmentation attacks in face recognition models
# deep learning project
# authors:
# Gil Kepel
# Shelly Francis.

# """

FILE=$1

if [ $FILE == "pretrained-network-celeba-hq" ]; then
    URL=https://www.dropbox.com/s/96fmei6c93o8b8t/100000_nets_ema.ckpt?dl=0
    mkdir -p ./expr/checkpoints/celeba_hq
    OUT_FILE=./expr/checkpoints/celeba_hq/100000_nets_ema.ckpt
    wget -N $URL -O $OUT_FILE

elif  [ $FILE == "pretrained-network-afhq" ]; then
    URL=https://www.dropbox.com/s/etwm810v25h42sn/100000_nets_ema.ckpt?dl=0
    mkdir -p ./expr/checkpoints/afhq
    OUT_FILE=./expr/checkpoints/afhq/100000_nets_ema.ckpt
    wget -N $URL -O $OUT_FILE

elif  [ $FILE == "wing" ]; then
    URL=https://www.dropbox.com/s/tjxpypwpt38926e/wing.ckpt?dl=0
    mkdir -p ./expr/checkpoints/
    OUT_FILE=./expr/checkpoints/wing.ckpt
    wget -N $URL -O $OUT_FILE
    URL=https://www.dropbox.com/s/91fth49gyb7xksk/celeba_lm_mean.npz?dl=0
    OUT_FILE=./expr/checkpoints/celeba_lm_mean.npz
    wget -N $URL -O $OUT_FILE

elif  [ $FILE == "celeba-hq-dataset" ]; then
    URL=https://postechackr-my.sharepoint.com/:u:/g/personal/dongbinna_postech_ac_kr/ES-jbCNC6mNHhCyR4Nl1QpYBlxVOJ5YiVerhDpzmoS9ezA?download=1
    ZIP_FILE=CelebA_HQ_facial_identity_dataset.zip
    mkdir -p ./CelebA_HQ_facial_identity_dataset
    wget $URL -O $ZIP_FILE
    unzip $ZIP_FILE -d ./CelebA_HQ_facial_identity_dataset
    rm $ZIP_FILE

elif  [ $FILE == "afhq-dataset" ]; then
    URL=https://www.dropbox.com/s/t9l9o3vsx2jai3z/afhq.zip?dl=0
    ZIP_FILE=./data/afhq.zip
    mkdir -p ./data
    wget -N $URL -O $ZIP_FILE
    unzip $ZIP_FILE -d ./data
    rm $ZIP_FILE

elif  [ $FILE == "afhq-v2-dataset" ]; then
    #URL=https://www.dropbox.com/s/scckftx13grwmiv/afhq_v2.zip?dl=0
    URL=https://www.dropbox.com/s/vkzjokiwof5h8w6/afhq_v2.zip?dl=0
    ZIP_FILE=./data/afhq_v2.zip
    mkdir -p ./data
    wget -N $URL -O $ZIP_FILE
    unzip $ZIP_FILE -d ./data
    rm $ZIP_FILE

elif  [ $FILE == "stargan-pre-trained" ]; then
    URL=https://www.dropbox.com/s/96fmei6c93o8b8t/100000_nets_ema.ckpt?dl=0
    ZIP_FILE=./expr/checkpoints/celeba_hq/100000_nets_ema.ckpt
    mkdir -p ./expr/checkpoints/celeba_hq
    wget -N $URL -O $ZIP_FILE

elif  [ $FILE == "models-weights" ]; then
    ZIP_FILE=./models/models.zip
    mkdir -p ./models
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1G5QNRswQjVNJ3gDTf0W8-vsiuFuVqDY_' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1G5QNRswQjVNJ3gDTf0W8-vsiuFuVqDY_" -O $ZIP_FILE
    unzip $ZIP_FILE -d ./models/
    rm $ZIP_FILE

else
    echo "Available arguments are pretrained-network-celeba-hq, pretrained-network-afhq, celeba-hq-dataset, and afhq-dataset."
    exit 1

fi
