import argparse
import sys
import os
import torch

import numpy as np
from torchvision.utils import save_image
from torch.autograd import Variable
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import OrderedDict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from utils import plot_confusion_matrix, plot_roc_auc_curve

from init import Modules, MemoryAllocation, HingeLoss
from datasets import TestGDLoader, TestCLoader
from utils import FileName
from feature_show import define_model_trunc, plot_2d_features, plot_3d_features

import warnings
warnings.filterwarnings("ignore")

def test_distance_vertical(module, dataloader, memory_allocation, opt):
    result_test_path = 'results/{}/{}'.format(opt.dataset_name, opt.project_name) +'/test'
    real_distance_B = []
    real_distance_A = []
    distance_output_B = []
    distance_output_A = []
    for i, batch in enumerate(dataloader.test_loader):
        real_A = Variable(memory_allocation.input_A.copy_(batch['A']))  # Set model input realA
        real_B = Variable(memory_allocation.input_B.copy_(batch['B']))  # Set model input realB
             
        # real_B
        distance_B_label = module.aux_C(real_B).squeeze(-1)
        real_distance_B.append(distance_B_label.data.cpu().float().numpy()[0])

        # real_A 
        distance_A_label = module.aux_C(real_A).squeeze(-1)
        real_distance_A.append(distance_A_label.data.cpu().float().numpy()[0])

        fb = module.netG_A2B(real_A, distance_B_label)
        fa = module.netG_B2A(real_B, distance_A_label)
        
        distance_output_B.append(module.aux_C(fb).squeeze(-1).data.cpu().float().numpy()[0])
        distance_output_A.append(module.aux_C(fa).squeeze(-1).data.cpu().float().numpy()[0])

    print(np.min(distance_output_A), np.max(distance_output_A))
    print(np.min(distance_output_B), np.max(distance_output_B))
    plt.plot(real_distance_A, label='target X projection distance')
    plt.plot(distance_output_A, label='reconstructed X projection distance')
    plt.xlabel("training samples")
    plt.ylabel("projection distances")
    plt.legend()
    plt.savefig(result_test_path + '/distance_Y2X.png', dpi=200)
    plt.close()

    plt.plot(real_distance_B, label='target Y projection distance')
    plt.plot(distance_output_B, label='reconstructed Y projection distance')
    plt.xlabel("training samples")
    plt.ylabel("projection distances")
    plt.legend()
    plt.savefig(result_test_path + '/distance_X2Y.png', dpi=200)
    plt.close()


def test_distance_horizontal(module, dataloader, memory_allocation, opt):
    result_test_path = 'results/{}/{}'.format(opt.dataset_name, opt.project_name) +'/test'
    real_distance_B = []
    real_distance_A = []
    distance_output_B = []
    distance_output_A = []
    for i, batch in enumerate(dataloader.test_loader):
        real_A = Variable(memory_allocation.input_A.copy_(batch['A']))
        real_AA = Variable(memory_allocation.input_AA.copy_(batch['AA']))
        real_B = Variable(memory_allocation.input_B.copy_(batch['B']))
        real_BB = Variable(memory_allocation.input_BB.copy_(batch['BB']))
             
        # real_B
        distance_B_label = module.aux_C(real_B).squeeze(-1)
        X_B_coordinate = module.aux_corr(real_B)["coor"].squeeze(-1).squeeze(-1)
        distance_BB_label = module.aux_C(real_BB).squeeze(-1)
        X_BB_coordinate = module.aux_corr(real_BB)["coor"].squeeze(-1).squeeze(-1)

        # real_A 
        distance_A_label = module.aux_C(real_A).squeeze(-1)
        X_A_coordinate = module.aux_corr(real_A)["coor"].squeeze(-1).squeeze(-1)
        distance_AA_label = module.aux_C(real_AA).squeeze(-1)
        X_AA_coordinate = module.aux_corr(real_AA)["coor"].squeeze(-1).squeeze(-1)

        # vertical edge
        vertical_edgeA = torch.abs(distance_A_label - distance_AA_label)
        vertical_edgeB = torch.abs(distance_B_label - distance_BB_label)

        # hypotenuse edge
        hypotenuse_edgeA = ((X_A_coordinate - X_AA_coordinate)**2).sum(dim=1).sqrt()
        hypotenuse_edgeB = ((X_B_coordinate - X_BB_coordinate)**2).sum(dim=1).sqrt()
            
        # horizontal edge
        horizontal_edgeA = (hypotenuse_edgeA**2 - vertical_edgeA**2).sqrt()
        horizontal_edgeB = (hypotenuse_edgeB**2 - vertical_edgeB**2).sqrt()

        fake_A2A = module.netG_A2A(real_A, -horizontal_edgeA.detach())
        fake_B2B = module.netG_B2B(real_B, -horizontal_edgeB.detach())

        output_fakeA2A = module.aux_C(fake_A2A).squeeze(-1)
        coord_fakeAA = module.aux_corr(fake_A2A)["coor"].squeeze(-1).squeeze(-1)
        hypotenuse_edgeAA = ((coord_fakeAA - X_A_coordinate.detach())**2).sum(dim=1).sqrt()
        vertical_edgeAA = torch.abs(output_fakeA2A - distance_A_label.detach())
        horizontal_edgeAA = (hypotenuse_edgeAA**2 - vertical_edgeAA**2).sqrt()

        output_fakeB2B = module.aux_C(fake_B2B).squeeze(-1)
        coord_fakeBB = module.aux_corr(fake_B2B)["coor"].squeeze(-1).squeeze(-1)
        hypotenuse_edgeBB = ((coord_fakeBB - X_B_coordinate.detach())**2).sum(dim=1).sqrt()
        vertical_edgeBB = torch.abs(output_fakeB2B - distance_B_label.detach())
        horizontal_edgeBB = (hypotenuse_edgeBB**2 - vertical_edgeBB**2).sqrt()

        real_distance_B.append(-horizontal_edgeB.data.cpu().float().numpy()[0])
        real_distance_A.append(-horizontal_edgeA.data.cpu().float().numpy()[0])
        
        distance_output_B.append(-horizontal_edgeBB.data.cpu().float().numpy()[0])
        distance_output_A.append(-horizontal_edgeAA.data.cpu().float().numpy()[0])

    print(np.min(distance_output_A), np.max(distance_output_A))
    print(np.min(distance_output_B), np.max(distance_output_B))
    plt.plot(real_distance_A, label='target X projection distance')
    plt.plot(distance_output_A, label='reconstructed X projection distance')
    plt.xlabel("training samples")
    plt.ylabel("projection distances")
    plt.legend()
    plt.savefig(result_test_path + '/distance_X2X.png', dpi=200)
    plt.close()

    plt.plot(real_distance_B, label='target Y projection distance')
    plt.plot(distance_output_B, label='reconstructed Y projection distance')
    plt.xlabel("training samples")
    plt.ylabel("projection distances")
    plt.legend()
    plt.savefig(result_test_path + '/distance_Y2Y.png', dpi=200)
    plt.close()


def test_gd(module, dataloader, data_c_loader, memory_allocation, opt):
    result_test_path = 'results/{}/{}'.format(opt.dataset_name, opt.project_name) +'/test'
    fakeA2B_path = result_test_path + '/fakeA2B'
    fakeB2A_path = result_test_path + '/fakeB2A'
    fakeA2A_path = result_test_path + '/fakeA2A'
    fakeB2B_path = result_test_path + '/fakeB2B'
    realA_path = result_test_path + '/realA'
    realB_path = result_test_path + '/realB'
    if not os.path.exists(fakeA2B_path):
        os.makedirs(fakeA2B_path)
    if not os.path.exists(fakeB2A_path):
        os.makedirs(fakeB2A_path)
    if not os.path.exists(fakeA2A_path):
        os.makedirs(fakeA2A_path)
    if not os.path.exists(fakeB2B_path):
        os.makedirs(fakeB2B_path)
    if not os.path.exists(realA_path):
        os.makedirs(realA_path)
    if not os.path.exists(realB_path):
        os.makedirs(realB_path)

    encoding_array = []  
    x = []
    class_to_idx = data_c_loader.test_set.class_to_idx   
    feature_path = 'results/{}/{}'.format(opt.dataset_name, opt.project_name) +'/test/classification'
    model_trunc = define_model_trunc(opt.project_name, module.netC)  # for plotting features
    for i, batch in enumerate(dataloader.test_loader):
        real_A = Variable(memory_allocation.input_A.copy_(batch['A']))
        real_AA = Variable(memory_allocation.input_A.copy_(batch['AA']))
        real_B = Variable(memory_allocation.input_B.copy_(batch['B']))
        real_BB = Variable(memory_allocation.input_BB.copy_(batch['BB']))
                    
        # real_B
        distance_B_label = module.aux_C(real_B).squeeze(-1)
        X_B_coordinate = module.aux_corr(real_B)["coor"].squeeze(-1).squeeze(-1)
        distance_BB_label = module.aux_C(real_BB).squeeze(-1)
        X_BB_coordinate = module.aux_corr(real_BB)["coor"].squeeze(-1).squeeze(-1)

        # real_A 
        distance_A_label = module.aux_C(real_A).squeeze(-1)
        X_A_coordinate = module.aux_corr(real_A)["coor"].squeeze(-1).squeeze(-1)
        distance_AA_label = module.aux_C(real_AA).squeeze(-1)
        X_AA_coordinate = module.aux_corr(real_AA)["coor"].squeeze(-1).squeeze(-1)

        # vertical edge
        vertical_edgeA = torch.abs(distance_A_label - distance_AA_label)
        vertical_edgeB = torch.abs(distance_B_label - distance_BB_label)

        # hypotenuse edge
        hypotenuse_edgeA = ((X_A_coordinate - X_AA_coordinate)**2).sum(dim=1).sqrt()
        hypotenuse_edgeB = ((X_B_coordinate - X_BB_coordinate)**2).sum(dim=1).sqrt()
            
        # horizontal edge
        horizontal_edgeA = (hypotenuse_edgeA**2 - vertical_edgeA**2).sqrt()
        horizontal_edgeB = (hypotenuse_edgeB**2 - vertical_edgeB**2).sqrt()


        fa2b = module.netG_A2B(real_A, distance_B_label.detach())
        fb2a = module.netG_B2A(real_B, distance_A_label.detach())
        fa2a = module.netG_A2A(real_A, -horizontal_edgeA.detach())
        fb2b = module.netG_B2B(real_B, -horizontal_edgeB.detach())
        
        fake_A2B = 0.5*(module.netG_A2B(real_A, distance_B_label.detach())[0].data + 1.0)
        fake_B2A = 0.5*(module.netG_B2A(real_B, distance_A_label.detach())[0].data + 1.0)
        fake_A2A = 0.5*(module.netG_A2A(real_A, -horizontal_edgeA.detach())[0].data + 1.0)    
        fake_B2B = 0.5*(module.netG_B2B(real_B, -horizontal_edgeB.detach())[0].data + 1.0)
        
        for j in range(opt.batchSize):
            feature = model_trunc(real_A[j].unsqueeze(0))['semantic_feature'].squeeze().detach().cpu().numpy()
            x.append(0)
            encoding_array.append(feature)
            
            feature = model_trunc(real_B[j].unsqueeze(0))['semantic_feature'].squeeze().detach().cpu().numpy()
            x.append(1)
            encoding_array.append(feature)
            
            feature = model_trunc(fb2a[j].unsqueeze(0))['semantic_feature'].squeeze().detach().cpu().numpy()
            x.append(0)
            encoding_array.append(feature)
            
            feature = model_trunc(fa2b[j].unsqueeze(0))['semantic_feature'].squeeze().detach().cpu().numpy()
            encoding_array.append(feature)
            x.append(1)
            
            feature = model_trunc(fa2a[j].unsqueeze(0))['semantic_feature'].squeeze().detach().cpu().numpy()
            encoding_array.append(feature)
            x.append(0)
            
            feature = model_trunc(fb2b[j].unsqueeze(0))['semantic_feature'].squeeze().detach().cpu().numpy()
            encoding_array.append(feature)
            x.append(1)

        A_name = FileName(batch['A_path'][0])                    # Set result A name
        B_name = FileName(batch['B_path'][0])                    # Set result B name
        save_image(fake_A2B, fakeA2B_path + '/X2Y{}png'.format(A_name))  # Save image files fakeA
        save_image(fake_B2A, fakeB2A_path + '/Y2X{}png'.format(B_name))  # Save image files fakeB
        save_image(fake_A2A, fakeA2A_path + '/X2X{}png'.format(A_name))  # Save image files fakeA
        save_image(fake_B2B, fakeB2B_path + '/Y2Y{}png'.format(B_name))  # Save image files fakeB
        save_image(real_A, realA_path + '/{}png'.format(A_name))  # Save image files realA
        save_image(real_B, realB_path + '/{}png'.format(B_name))  # Save image files realB

        sys.stdout.write('\rGenerated images %04d of %04d' % (i+1, len(dataloader.test_loader)))

    encoding_array = np.array(encoding_array)    
    testset_targets = np.array(x)
    print(encoding_array.shape)
    plot_2d_features(encoding_array, class_to_idx, feature_path, testset_targets)
    plot_3d_features(encoding_array, class_to_idx, feature_path, testset_targets)
    print('The 2D and 3D features have been plotted\n')

    sys.stdout.write('\n')


def test_c(module, dataloader, opt):
    test_loss = 0.0                                   # keep track of testing loss
    test_correct = 0                                  # keep track of testing correct numbers

    class_correct = list(0 for i in range(2))         # keep track of each category's correct numbers
    class_total = list(0 for i in range(2))           # acculmulate each category's toal numbers

    classes = dataloader.test_set.classes             # accuracy for each category & confusion matrix

    pred_cm = torch.cuda.FloatTensor([])    # for confusion matrix
    pred_cm = pred_cm.cuda(opt.gpu_ids[0])  # for confusion matrix

    y_true = []  # init for AUC and ROC
    y_prob = []  # init for AUC and ROC

    class_to_idx = dataloader.test_set.class_to_idx                  # for plotting features
    print('The classification objects and indexes: ', class_to_idx)
    encoding_array = []                                              # for plotting features
    model_trunc = define_model_trunc(opt.project_name, module.netC)  # for plotting features

    criterion = HingeLoss()          # define the cost function

    for data, target in tqdm(dataloader.test_loader):
        data, target = data.cuda(opt.gpu_ids[0]), target.cuda(opt.gpu_ids[0])    # move to GPU or cpu
        output = module.netC(data)                                  # forward pass
        loss = criterion(output, target)                     # calculate the loss
        test_loss += loss.item()*data.size(0)                # update testing loss

        predict_y = torch.where(output >= 0, 1, 0)           # output predicted class (i.e., idx)

        test_correct += (predict_y == target).sum().item()   # update validation correct numbers
        
        correct_tensor = (predict_y == target)
        for i in range(opt.batchSize):
            c = target[i]
            class_correct[c] += correct_tensor[i].item()
            class_total[c] += 1

            y_true_ = np.squeeze(target[i].data.cpu().float().numpy())
            y_true.append(int(y_true_))
            y_prob_ = np.squeeze(output[i].data.cpu().float().numpy())
            y_prob.append(y_prob_)

            feature = model_trunc(data[i].unsqueeze(0))['semantic_feature'].squeeze().detach().cpu().numpy()
            encoding_array.append(feature)

        pred_cm = torch.cat((pred_cm, predict_y), dim=0)

    ave_test_loss = test_loss/len(dataloader.test_loader.dataset)      # calculate average loss
    print('\nTesting Loss: {:.6f}'.format(ave_test_loss))

    ave_test_acc = test_correct/len(dataloader.test_loader.dataset)    # calculate average accuracy
    print('Testing Accuracy (Overall): {:.4f} ({}/{})'.format(ave_test_acc, test_correct, len(dataloader.test_loader.dataset)))

    for i in range(2):  #  output accuracy for each category
        if class_total[i] > 0:
            print('Testing accuracy of {}: {:.4f} ({}/{})'.format(classes[i], class_correct[i]/class_total[i], class_correct[i], class_total[i]))
        else:
            print('Testing accuracy of {}: N/A (no training examples)' % (classes[i]))

    print('\nThe Confusion Matrix is plotted and saved:')
    cMatrix = confusion_matrix(torch.tensor(dataloader.test_set.targets), pred_cm.cpu())
    print(cMatrix)
    
    result_path ='results/{}/{}'.format(opt.dataset_name, opt.project_name) +'/test/classification'
    if not os.path.exists(result_path): 
        os.makedirs(result_path)

    plot_confusion_matrix(classes, cMatrix, result_path)

    print('\nThe Classification Report is plotted below:')
    print(classification_report(torch.tensor(dataloader.test_set.targets), pred_cm.cpu()))

    auc = roc_auc_score(y_true, y_prob)
    print('The AUC (Area Under Curve) is: {:.6f}'.format(auc))
    plot_roc_auc_curve(y_true, y_prob, opt.project_name, result_path)
    print('The ROC curve have been plotted\n')
    
    feature_path = 'results/{}/{}'.format(opt.dataset_name, opt.project_name) +'/test/classification'
    if not os.path.exists(feature_path): 
        os.makedirs(feature_path)
    encoding_array = np.array(encoding_array)
    testset_targets = np.array(dataloader.test_set.targets)
    plot_2d_features(encoding_array, class_to_idx, feature_path, testset_targets)
    plot_3d_features(encoding_array, class_to_idx, feature_path, testset_targets)
    print('The 2D and 3D features have been plotted\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSize', type=int, choices=[1], default=1, help='size of the batches')
    parser.add_argument('--dataroot', type=str, default='/mnt/evo1/data/iclr2024/augmented_covid', help='root directory of the dataset')
    parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
    parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
    parser.add_argument('--size', type=int, default=224, help='size of the data (squared assumed)')
    parser.add_argument('--cuda', action='store_true', help='use GPU computation')
    parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
    parser.add_argument('--num_classes', default=1, type=int, help='Number of classes for classification')
    parser.add_argument('--gpu_ids', type=str, default='1')
    parser.add_argument('--generator_A2B', type=str, default='checkpoints/covid/cyclegan_convnext_tiny/best_netG_A2B.pth', help='A2B generator checkpoint file')
    parser.add_argument('--generator_B2A', type=str, default='checkpoints/covid/cyclegan_convnext_tiny/best_netG_B2A.pth', help='B2A generator checkpoint file')
    parser.add_argument('--generator_A2A', type=str, default='checkpoints/covid/cyclegan_convnext_tiny/best_netG_A2A.pth', help='A2B generator checkpoint file')
    parser.add_argument('--generator_B2B', type=str, default='checkpoints/covid/cyclegan_convnext_tiny/best_netG_B2B.pth', help='B2A generator checkpoint file')
    parser.add_argument('--classifier', type=str, default='checkpoints/covid/cyclegan_convnext_tiny/best_netC.pth', help='classifier checkpoint file')
    parser.add_argument('--aux_classifier', type=str, default='/checkpoints/covid/pretrained/convnext_tiny_best_netC.pth', help='classifier checkpoint file')
    parser.add_argument('--dataset_name', default='covid', type=str, help='Choose the dataset name for save results')
    parser.add_argument('--project_name', default='cyclegan_convnext_tiny', type=str, help='Choose the project name for save results')
    parser.add_argument('--model_name', default='convnext_tiny', type=str, 
        choices=['alexnet', 'vgg13', 'vgg16', 'googlenet', 'resnet18', 'resnet34', 'densenet121',
        'mnasnet1_0', 'mobilenet_v3_small', 'efficientnet_b5', 'convnext_tiny'], help='Choose the model you want train')
    opt = parser.parse_args()

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    if opt.cuda:
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)

    module = Modules(opt.input_nc, opt.output_nc, opt.num_classes, opt.model_name)
    module.init_modules(opt.cuda, opt.gpu_ids)
    
    state_dict = torch.load(opt.aux_classifier)
    new_state_dict = OrderedDict()
    corr_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'module' not in k:
            k = 'module.'+k
        else:
            k = k.replace('features.module.', 'module.features.')
        new_state_dict[k]=v
        if opt.model_name == 'densenet121' and 'classifier' not in k:
            corr_state_dict[k]=v
        elif opt.model_name == 'alexnet' and 'classifier.6' not in k:
            corr_state_dict[k]=v
        elif opt.model_name == 'vgg13' and 'classifier.6' not in k:
            corr_state_dict[k]=v
        elif opt.model_name == 'vgg16' and 'classifier.6' not in k:
            corr_state_dict[k]=v
        elif opt.model_name == 'resnet18' and 'fc' not in k:
            corr_state_dict[k]=v
        elif opt.model_name == 'resnet34' and 'fc' not in k:
            corr_state_dict[k]=v
        elif opt.model_name == 'resnet50' and 'fc' not in k:
            corr_state_dict[k]=v
        elif opt.model_name == 'googlenet' and 'dropout' not in k and 'fc' not in k:
            corr_state_dict[k]=v
        elif opt.model_name == 'wide_resnet50_2' and 'fc' not in k:
            corr_state_dict[k]=v
        elif opt.model_name == 'mobilenet_v2' and 'classifier.1' not in k:
            corr_state_dict[k]=v
        elif opt.model_name == 'efficientnet_b5' and 'classifier.1' not in k:
            corr_state_dict[k]=v
        elif opt.model_name == 'convnext_tiny' and 'classifier.1' not in k and 'classifier.2' not in k:
            corr_state_dict[k]=v
        elif opt.model_name == 'mnasnet1_0' and 'classifier.1' not in k:
            corr_state_dict[k]=v
        elif opt.model_name == 'mobilenet_v3_small' and 'classifier.3' not in k:
            corr_state_dict[k]=v

    if opt.model_name == 'mnasnet1_0':
        module.aux_C.load_state_dict(state_dict)
    else:
        module.aux_C.load_state_dict(new_state_dict)
    module.aux_corr.load_state_dict(corr_state_dict)
    
    module.aux_C.eval()
    module.aux_corr.eval()

    # Load state dicts for generators and a classifier
    module.netG_A2B.load_state_dict(torch.load(opt.generator_A2B))
    module.netG_B2A.load_state_dict(torch.load(opt.generator_B2A))
    module.netG_A2A.load_state_dict(torch.load(opt.generator_A2A))
    module.netG_B2B.load_state_dict(torch.load(opt.generator_B2B))
    module.netC.load_state_dict(torch.load(opt.classifier))

    # Set module networks test mode
    module.netG_A2B.eval()
    module.netG_B2A.eval()
    module.netG_A2A.eval()
    module.netG_B2B.eval()
    module.netC.eval()
    
    # test the cyclegan (Generator & Disciminator) performance
    data_gd_loader = TestGDLoader(opt.size, opt.dataroot, opt.batchSize, opt.n_cpu)
    memory_allocation = MemoryAllocation(opt.cuda, opt.batchSize, opt.input_nc, opt.output_nc, opt.size, opt.gpu_ids)
    data_c_loader = TestCLoader(opt.size, opt.dataroot, opt.batchSize, opt.n_cpu)
    test_gd(module, data_gd_loader, data_c_loader, memory_allocation, opt)

    # test the distance reconstruction performance
    test_distance_vertical(module, data_gd_loader, memory_allocation, opt)
    test_distance_horizontal(module, data_gd_loader, memory_allocation, opt)

    # test the classifier performance
    data_c_loader = TestCLoader(opt.size, opt.dataroot, opt.batchSize, opt.n_cpu)
    test_c(module, data_c_loader, opt)
