import time
import datetime
import sys
import os
import torch

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from torch.autograd import Variable
from PIL import Image
from sklearn.metrics import roc_curve, auc

# Converts a Tensor into an image array (numpy)
# |imtype|: the desired type of the converted numpy array
def tensor2im(input_image, imtype=np.uint8):
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def FileName(path):
    name = ''
    filepath, filename = os.path.split(path)
    name_list = filename.split('.')[:-1]
    for sub_name in name_list:
        name += sub_name
        name += '.'

    return name


def g_gan_loss_visualize(epochs, train_g_gan_loss, result_path):
    plt.style.use("ggplot")
    plt.figure(dpi=200)
    plt.subplot(1, 1, 1)
    plt.title("Epoch G GAN Loss")
    plt.plot(np.arange(1, epochs+1), train_g_gan_loss, label='train_g_gan_loss', color='r', linestyle='-', marker='o')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('G GAN Loss')
    plt.savefig(result_path + '/g_gan_loss.png')


def g_mse_loss_visualize(epochs, train_g_mse_loss, result_path):
    plt.style.use("ggplot")
    plt.figure(dpi=200)
    plt.subplot(1, 1, 1)
    plt.title("Epoch G Mse Loss")
    plt.plot(np.arange(1, epochs+1), train_g_mse_loss, label='train_g_mse_loss', color='r', linestyle='-', marker='o')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('G Mse Loss')
    plt.savefig(result_path + '/g_mse_loss.png')


def g_cycle_loss_visualize(epochs, train_g_cycle_loss, result_path):
    plt.style.use("ggplot")
    plt.figure(dpi=200)
    plt.subplot(1, 1, 1)
    plt.title("Epoch G Cycle Loss")
    plt.plot(np.arange(1, epochs+1), train_g_cycle_loss, label='train_g_cycle_loss', color='r', linestyle='-', marker='o')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('G Cycle Loss')
    plt.savefig(result_path + '/g_cycle_loss.png')


def d_loss_visualize(epochs, train_d_loss, result_path):
    plt.style.use("ggplot")
    plt.figure(dpi=200)
    plt.subplot(1, 1, 1)
    plt.title("Epoch D Loss")
    plt.plot(np.arange(1, epochs+1), train_d_loss, label='train_d_loss', color='r', linestyle='-', marker='o')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('D Loss')
    plt.savefig(result_path + '/d_loss.png')


def c_hinge_real_loss_visualize(epochs, train_c_hinge_real_loss, result_path):
    plt.style.use("ggplot")
    plt.figure(dpi=200)
    plt.subplot(1, 1, 1)
    plt.title("Epoch C Hinge Real Loss")
    plt.plot(np.arange(1, epochs+1), train_c_hinge_real_loss, label='train_c_hinge_real_loss', color='r', linestyle='-', marker='o')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('C Hinge Real Loss')
    plt.savefig(result_path + '/c_hinge_real_loss.png')
    

def c_hinge_fake_loss_visualize(epochs, train_c_hinge_fake_loss, result_path):
    plt.style.use("ggplot")
    plt.figure(dpi=200)
    plt.subplot(1, 1, 1)
    plt.title("Epoch C Hinge Fake Loss")
    plt.plot(np.arange(1, epochs+1), train_c_hinge_fake_loss, label='train_c_hinge_fake_loss', color='r', linestyle='-', marker='o')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('C Hinge Fake Loss')
    plt.savefig(result_path + '/c_hinge_fake_loss.png')


def val_loss_visualize(epochs, val_c_loss, result_path):
    plt.style.use("ggplot")
    plt.figure(dpi=200)
    plt.subplot(1, 1, 1)
    plt.title("Epoch C Loss")
    plt.plot(np.arange(1, epochs+1), val_c_loss, label='val_c_loss', color='b', linestyle='-', marker='^')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('C Loss')
    plt.savefig(result_path + '/c_val_loss.png')
    

def val_acc_visualize(epochs, val_c_acc, result_path):
    plt.style.use("ggplot")
    plt.figure(dpi=200)
    plt.subplot(1, 1, 1)
    plt.title("Epoch C Accuracy")
    plt.plot(np.arange(1, epochs+1), val_c_acc, label='val_c_acc', linestyle='-', color='b', marker='^')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('C Accuracy')
    plt.savefig(result_path + '/c_val_acc.png')


def plot_confusion_matrix(class_names, confusion_matrix, result_path):
    cfmt =pd.DataFrame(confusion_matrix, index=class_names, columns=class_names)
    plt.figure(dpi=200)
    plt.title('Confusion Matrix')
    sns.heatmap(cfmt, annot=True, cmap='BuGn', fmt="d")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig(result_path + '/confusion_matrix.png')


def plot_roc_auc_curve(y_true, y_pred_prob, project_name, result_path):
    y_test_true = np.array(y_true)
    y_test_predprob = np.array(y_pred_prob)
    fpr, tpr, thresholds = roc_curve(y_test_true, y_test_predprob, pos_label=1)
    roc_auc = auc(fpr, tpr)

    plt.figure(dpi=200)
    plt.plot(fpr, tpr, color="blue", lw=3, label='{} (area = {})'.format(project_name, roc_auc))
    plt.plot([0, 1], [0, 1], color="grey", lw=3, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.savefig(result_path + '/roc_auc_curve.png')


class Logger():
    def __init__(self, n_epochs, batches_epoch, result_path):
        self.n_epochs = n_epochs             # total number of epochs
        self.batches_epoch = batches_epoch   # batches per epoch
        self.losses = {}                     # keep track of losses
        self.epoch = 1
        self.result_path = result_path       # save train or val figures

        self.G_gan_loss = []
        self.G_mse_loss = []
        self.G_cycle_loss = []
        self.D_loss = []
        self.C_hinge_real_loss = []
        self.C_hinge_fake_loss = []

    def log(self, losses=None, images=None):           # call for each epoch
        sys.stdout.write('\rEpoch %03d/%03d -- ' % (self.epoch, self.n_epochs))

        for i, loss_name in enumerate(losses.keys()):  # print losses
            if loss_name == 'acc_C':
                self.losses[loss_name] = losses[loss_name]
            else:
                self.losses[loss_name] = losses[loss_name].item()
            if (i+1) == len(losses.keys()):
                sys.stdout.write('%s: %.4f \n ' % (loss_name, self.losses[loss_name]/self.batches_epoch))
            else:
                sys.stdout.write('%s: %.4f | ' % (loss_name, self.losses[loss_name]/self.batches_epoch))
        
        for image_name, tensor in images.items():      # save images
            image_numpy = tensor2im(tensor.data)
            save_image(image_numpy, self.result_path + '/epoch{}_{}.png'.format(self.epoch, image_name))

        for loss_name, loss in self.losses.items():    # add losses into list
            if loss_name == 'loss_G_GAN':
                self.G_gan_loss.append(loss/self.batches_epoch)
            if loss_name == 'loss_G_mse':
                self.G_mse_loss.append(loss/self.batches_epoch)
            if loss_name == 'loss_G_cycle':
                self.G_cycle_loss.append(loss/self.batches_epoch)
            if loss_name == 'loss_D':
                self.D_loss.append(loss/self.batches_epoch)
            if loss_name == 'loss_C_hinge_real':
                self.C_hinge_real_loss.append(loss/self.batches_epoch)
            if loss_name == 'loss_C_hinge_fake':
                self.C_hinge_fake_loss.append(loss/self.batches_epoch)

        self.epoch += 1