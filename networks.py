import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator, self).__init__()

        # Initial convolution block       
        model = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc+1, 64, 7),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True) ]

        # Downsampling
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features//2
        for _ in range(2):
            model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2

        # Output layer
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(64, output_nc, 7),
                    nn.Tanh() ]

        self.model = nn.Sequential(*model)

    def forward(self, x, c1):
        c1 = c1.view(c1.size(0), 1, 1, 1)
        c1 = c1.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c1], dim = 1)
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [   nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(128), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(256), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(256, 512, 4, padding=1),
                    nn.InstanceNorm2d(512), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x =  self.model(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)


class Classifier(nn.Module):
    def __init__(self, num_classes):
        super(Classifier, self).__init__()              # input image = 224 x 244 x 3
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)     # 224 x 224 x 3 --> 112 x 112 x 32 maxpool
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)    # 112 x 112x 32 --> 56 x 56 x 64 maxpool
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)   # 56 x 56 x 64 --> 28 x 28 x 128 maxpool    
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)  # 28 x 28 x 128 --> 14 x 14 x 256 maxpool    

        self.pool = nn.MaxPool2d(2, 2)            # maxpool 2 x 2
      
        self.fc1 = nn.Linear(256 * 14 * 14, 512)  # 28 x 28 x 128 vector flat 512
        self.fc2 = nn.Linear(512, num_classes)    # Two categories
      
        self.dropout = nn.Dropout(0.5)            # Try it and use 0.5 After saving the value

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # On the conv1 layer, relu the maxpool. 112 x 112 x 32
        x = self.pool(F.relu(self.conv2(x)))  # On the conv2 layer, relu the maxpool. 56 x 56 x 64
        x = self.pool(F.relu(self.conv3(x)))  # On the conv3 layer, relu the maxpool. 28 x 28 x 128
        x = self.pool(F.relu(self.conv4(x)))  # On the conv4 layer, relu the maxpool. 14 x 14 x 256

        x = x.view(-1, 256 * 14 * 14)         # Expand Image

        x = self.dropout(x)            # Apply dropout
        x = self.fc1(x)
        x = F.relu(x)        # After inserting into the fc layer, relu
        x = self.dropout(x)            # Apply dropout
        coordinate = x
        x = self.fc2(x)

        return x.squeeze(-1), coordinate


def define_pretrained_model(model_name, num_classes, use_corr):
    """
    The following classification models are available, 
    with or without pre-trained weights:
    """
    model = None

    if model_name == 'cnn':
        model = Classifier(num_classes)  # Auxiliary Classifier
        
    #------------------------------------AlexNet (2012)------------------------------------#
    if model_name == 'alexnet':   
        model = models.alexnet(weights = models.AlexNet_Weights.DEFAULT)
        num_fc = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features=num_fc, out_features=num_classes)
        model_corr = create_feature_extractor(model, return_nodes={"classifier.5": "coor"})
        
    #--------------------------------------VGG (2014)--------------------------------------#
    if model_name == 'vgg11':     
        model = models.vgg11(weights = models.VGG11_Weights.DEFAULT)
    if model_name == 'vgg13':     
        model = models.vgg13(weights = models.VGG13_Weights.DEFAULT)
        num_fc = model.classifier[6].in_features            
        model.classifier[6] = nn.Linear(num_fc, num_classes)
        model_corr = create_feature_extractor(model, return_nodes={"classifier.5": "coor"})
    if model_name == 'vgg16':     
        model = models.vgg16(weights = models.VGG16_Weights.DEFAULT)
        num_fc = model.classifier[6].in_features            
        model.classifier[6] = nn.Linear(num_fc, num_classes)
        model_corr = create_feature_extractor(model, return_nodes={"classifier.5": "coor"})

    if model_name == 'vgg19':
        model = models.vgg19(weights = models.VGG19_Weights.DEFAULT)
        num_fc = model.classifier[6].in_features             
        model.classifier[6] = nn.Linear(num_fc, num_classes) 

    #-----------------------------------GoogleNet (2014)-----------------------------------#
    if model_name == 'googlenet':
        model = models.googlenet(weights = models.GoogLeNet_Weights.DEFAULT)
        num_fits = model.fc.in_features
        model.fc = nn.Linear(num_fits, num_classes)
        model_corr = create_feature_extractor(model, return_nodes={"avgpool": "coor"})

    #-------------------------------------ResNet (2015)------------------------------------#
    if model_name == 'resnet18':
        model = models.resnet18(weights = models.ResNet18_Weights.DEFAULT)
        num_fits = model.fc.in_features
        model.fc = nn.Linear(num_fits, num_classes) 
        model_corr = create_feature_extractor(model, return_nodes={"avgpool": "coor"})

    if model_name == 'resnet34':
        model = models.resnet34(weights = models.ResNet34_Weights.DEFAULT)
        num_fits = model.fc.in_features
        model.fc = nn.Linear(num_fits, num_classes)
        model_corr = create_feature_extractor(model, return_nodes={"avgpool": "coor"})

    if model_name == 'resnet50':
        model = models.resnet50(weights = models.ResNet50_Weights.DEFAULT)
        num_fits = model.fc.in_features
        model.fc = nn.Linear(num_fits, num_classes)
        model_corr = create_feature_extractor(model, return_nodes={"avgpool": "coor"})

    if model_name == 'resnet101':
        model = models.resnet101(weights = models.ResNet101_Weights.DEFAULT)

    #----------------------------------Inception v3 (2015)---------------------------------#
    if model_name == 'inception_v3':
        model = models.inception_v3(weights = models.Inception_V3_Weights.DEFAULT)
        model.aux_logits = False
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

    #-----------------------------------SqueezeNet (2016)----------------------------------#
    if model_name == 'squeezenet1_0':
        model = models.squeezenet1_0(weights = models.SqueezeNet1_0_Weights.DEFAULT)
    if model_name == 'squeezenet1_1':
        model = models.squeezenet1_1(weights = models.SqueezeNet1_1_Weights.DEFAULT)

    #------------------------------------DenseNet (2016)-----------------------------------#
    if model_name == 'densenet121':
        model = models.densenet121(weights = models.DenseNet121_Weights.DEFAULT)
        model.features.avgpool = nn.AdaptiveAvgPool2d((1,1))
        model.classifier = nn.Linear(in_features=model.classifier.in_features, out_features=num_classes)
        model_corr = create_feature_extractor(model, return_nodes={"features.avgpool": "coor"})
    if model_name == 'densenet161':
        model = models.densenet161(weights = models.DenseNet161_Weights.DEFAULT)
    if model_name == 'densenet169':
        model = models.densenet169(weights = models.DenseNet169_Weights.DEFAULT)
    if model_name == 'densenet201':
        model = models.densenet201(weights = models.DenseNet201_Weights.DEFAULT)

    #----------------------------------ShuffleNet v2 (2018)--------------------------------#
    if model_name == 'shufflenet_v2_x0_5':  # x0_5, x1_0, x1_5, x2_0
        model = models.shufflenet_v2_x0_5(weights = models.ShuffleNet_V2_X0_5_Weights.DEFAULT)

    #-------------------------------------MnasNet (2018)--------------------------------#
    if model_name == 'mnasnet0_5':     # 0_5, 0_75, 1_0, 1_3
        model = models.mnasnet0_5(weights = models.MNASNet0_5_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(in_features=model.classifier[1].in_features, out_features=num_classes)
    if model_name == 'mnasnet1_0':     # 0_5, 0_75, 1_0, 1_3
        model = models.mnasnet1_0(weights = models.MNASNet1_0_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(in_features=model.classifier[1].in_features, out_features=num_classes)
        model_corr = create_feature_extractor(model, return_nodes={"classifier.0": "coor"})

    #-------------------------------------ResNeXt (2019)-----------------------------------#
    if model_name == 'resnext50_32x4d':
        model = models.resnext50_32x4d(weights = models.ResNeXt50_32X4D_Weights.DEFAULT)

    if model_name == 'wide_resnet50_2':
        model = models.wide_resnet50_2(weights = models.Wide_ResNet50_2_Weights.DEFAULT)
        model_corr = create_feature_extractor(model, return_nodes={"avgpool": "coor"})
        model.fc = nn.Linear(in_features=model.fc.in_features, out_features=num_classes)

    #------------------------------------MobileNet (2019)----------------------------------#
    if model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(weights = models.MobileNet_V2_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(in_features=model.classifier[1].in_features, out_features=num_classes)
        model_corr = create_feature_extractor(model, return_nodes={"classifier.0": "coor"})
    if model_name == 'mobilenet_v3_small':
        model = models.mobilenet_v3_small(weights = models.MobileNet_V3_Small_Weights.DEFAULT)
        model.classifier[3] = nn.Linear(in_features=model.classifier[3].in_features, out_features=num_classes)
        model_corr = create_feature_extractor(model, return_nodes={"classifier.2": "coor"})
    if model_name == 'mobilenet_v3_large':
        model = models.mobilenet_v3_large(weights = models.MobileNet_V3_Large_Weights.DEFAULT)

    #-----------------------------------EfficientNet (2019)--------------------------------#
    if model_name == 'efficientnet_b5':  # b0 ~ b7
        model = models.efficientnet_b5(weights = models.EfficientNet_B5_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(in_features=model.classifier[1].in_features, out_features=num_classes)
        model_corr = create_feature_extractor(model, return_nodes={"classifier.0": "coor"})

    #--------------------------------------RegNet (2020)-----------------------------------#
    if model_name == 'regnet_x_400mf': # X: 400MF, 800MF, 1.6GF, 3.2GF, 8.0GF, 16GF, 32GF
        model = models.regnet_x_400mf(weights = models.RegNet_X_400MF_Weights.DEFAULT)
    if model_name == 'regnet_y_400mf': # Y: 400MF, 800MF, 1.6GF, 3.2GF, 8.0GF, 16GF, 32GF, 128GF
        model = models.regnet_y_400mf(weights = models.RegNet_Y_400MF_Weights.DEFAULT)

    #--------------------------------Vision Transformer (2020)-----------------------------#
    if model_name == 'vit_b_32':  # b_32, b_16, l_32, l_16, h_14
        model = models.vit_b_32(weights = models.vit_b_32)

    #---------------------------------EfficientNet v2 (2021)-------------------------------#
    if model_name == 'efficientnet_v2_s':  # S, M, L
        model = models.efficientnet_v2_s(weights = models.EfficientNet_V2_S_Weights.DEFAULT)

    #-------------------------------------ConvNeXt (2022)----------------------------------#
    if model_name == 'convnext_tiny':  # tiny, small, base, large
        model = models.convnext_tiny(weights = models.ConvNeXt_Tiny_Weights.DEFAULT)
        num_fc = model.classifier[2].in_features 
        model.classifier[2] = nn.Linear(num_fc, out_features = num_classes)
        model_corr = create_feature_extractor(model, return_nodes={"classifier.0": "coor"})
    
    if use_corr:
        return model, model_corr
    else:
        return model