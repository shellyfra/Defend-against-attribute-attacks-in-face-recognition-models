import os
import time
from datetime import datetime
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
import torch.nn as nn
import torchvision
import numpy as np

def train(model, criterion, optimizer, num_epochs, train_dataloader, train_dataset, test_dataloader, test_dataset, path,
          with_aug=False, free_text='', device=None):
    if device:
        model = model.to(device)

    start_time = time.time()
    train_losses, test_losses = [], []
    train_acc, test_acc = [], []

    for epoch in range(num_epochs):
        """ Training Phase """
        model.train()

        running_loss = 0.
        running_corrects = 0

        # load a batch data of images
        for i, (inputs, labels) in tqdm(enumerate(train_dataloader)):  # labels[0] == attr, labels[1] === identity
            inputs = inputs.to(device)
            labels = labels.to(device)

            # forward inputs and get output
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # get loss value and update the network weights
            optimizer.zero_grad()  # clean the gradients from previous iteration
            loss.backward()  # autograd backward to calculate gradients
            optimizer.step()  # apply update to the weights

            _, preds = torch.max(outputs, 1)

            running_loss += loss.item() * inputs.size(0)

            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_dataset)
        train_losses.append(epoch_loss)
        epoch_acc = running_corrects / len(train_dataset) * 100.
        train_acc.append(epoch_acc)

        epoch_acc = running_corrects / len(train_dataset) * 100.
        print('[Train #{}] Loss: {:.4f} Acc: {:.4f}% Time: {:.4f}s'.format(epoch, epoch_loss, epoch_acc,
                                                                           time.time() - start_time))

        """ Test Phase """
        model.eval()

        with torch.no_grad():
            running_loss = 0.
            running_corrects = 0

            for inputs, labels in test_dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(test_dataset)
            test_losses.append(epoch_loss)

            epoch_acc = running_corrects / len(test_dataset) * 100.
            test_acc.append(epoch_acc)

            print('[Test #{}] Loss: {:.4f} Acc: {:.4f}% Time: {:.4f}s'.format(epoch, epoch_loss, epoch_acc,
                                                                              time.time() - start_time))
        # save model
        if epoch % 1 == 0:
            print('==> Saving model ...')
            state = {
                'net': model.state_dict(),
                'epoch': epoch,
            }
            if not os.path.isdir(f'{path}/models/'):
                os.mkdir(f'{path}/models/')
            file_name_aug = 'aug' if with_aug else 'no_aug'
            torch.save(state, '{}/models/CelebA_HQ_Facenet_{}_{}_epoch{}_acc_{:.4f}_{}.pth'.format(path, file_name_aug,
                                                                                                   free_text, epoch,
                                                                                                   epoch_acc,
                                                                                                   datetime.strftime(
                                                                                                       datetime.now(),
                                                                                                       "%d_%m_%Y_%H_%M_%S")))

    print('==> Finished Training ...')
    return train_losses, test_losses, train_acc, test_acc

def eval_acc(model, test_dataloader, device, criterion=nn.CrossEntropyLoss()):
  model.eval()
  model.classify = True

  with torch.no_grad():
      running_loss = 0.
      running_corrects = 0

      for inputs, labels in test_dataloader:
          inputs = inputs.to(device)
          labels = labels.to(device)

          outputs = model(inputs)
          loss = criterion(outputs, labels)

          _, preds = torch.max(outputs, 1)
          running_loss += loss.item() * inputs.size(0)
          running_corrects += torch.sum(preds == labels.data)/16
          # print(running_corrects)
          # break

      loss = running_loss / len(test_dataloader)

      acc = (running_corrects / len(test_dataloader)) * 100.

  print('Loss: {:.4f} Acc: {:.4f}% '.format(loss, acc))
  return loss, acc


def set_parameter_requires_grad(model, num_classes, feature_extracting=False):
#(last_linear): Linear(in_features=1792, out_features=512, bias=False)
#(last_bn): BatchNorm1d(512, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
# (logits): Linear(in_features=512, out_features=8631, bias=True)
    if feature_extracting:
        # frozen model
        model.requires_grad_(False)
    else:
        # fine-tuning
        for param in model.parameters():
            param.requires_grad = False

        num_ftrs = model.logits.in_features
        model.logits = nn.Linear(num_ftrs, num_classes)
        model.logits.requires_grad = True
        # model.last_linear.requires_grad = True
    model.classify = True
    return model


def get_params_to_update(model):
    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t", name)
    return params_to_update


def imshow(input, title):
    # torch.Tensor => numpy
    input = input.numpy().transpose((1, 2, 0))
    # undo image normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    input = std * input + mean
    input = np.clip(input, 0, 1)
    # display images
    plt.imshow(input)
    plt.title(title)
    plt.show()

def imshow_no_normalization(input, title):
    # torch.Tensor => numpy
    print('got here0?')
    input = input.numpy().transpose((1, 2, 0))
    # display images
    a = np.array(input)
    print(input)
    plt.imshow(a)
    plt.title(title)
    print('got here2?')

    plt.show()
