from facenet_pytorch import InceptionResnetV1
from utils import set_parameter_requires_grad, imshow_no_normalization
import os
import argparse
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
import glob
from datetime import datetime
import argparse
from PIL import Image


def model_pre_proccess(state_path, num_classes, device):
    state = torch.load(state_path, map_location=device)
    facenet = InceptionResnetV1(pretrained='vggface2').to(device)
    facenet = set_parameter_requires_grad(facenet, num_classes)

    facenet.load_state_dict(state['net'])
    facenet = facenet.to(device)
    facenet.classify = True
    facenet.eval()
    return facenet


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # model arguments
    parser.add_argument('--img_path', type=str, default='./StarGAN_with_our_changes/attack_objects/80/13726.jpg',
                        help='Image resolution')
    parser.add_argument('--state_path', type=str,
                        default='./StarGAN_with_our_changes/models/CelebA_HQ_Facenet_with_aug_attribute.pth',
                        help='Model state under expr')
    parser.add_argument('--dataset_dir', type=str,
                        default='./StarGAN_with_our_changes/CelebA_HQ_facial_identity_dataset',
                        help='data_dir')
    parser.add_argument('--batch_size', type=int,
                        default=1,
                        help='batch_size')

    args = parser.parse_args()
    transforms_orig = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    data_dir = args.dataset_dir
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transforms_orig)
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transforms_orig)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=args.batch_size)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True,
                                                  num_workers=args.batch_size)
    class_names = train_dataset.classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    face_net = model_pre_proccess(args.state_path, len(class_names), device)
    id_to_attack_imgs = [transforms_orig(Image.open(args.img_path).convert('RGB'))]
    input = torch.stack(id_to_attack_imgs)
    img_probs = face_net(input.to(device))
    out0 = torchvision.utils.make_grid(input)
    pred_class = [class_names[prob.argmax().item()] for prob in img_probs]
    imshow_no_normalization(out0, title=f'predicted class = {pred_class}')
