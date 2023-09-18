import random
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D


def define_model_trunc(project_name, model):
    model_trunc = None
    print(project_name)
    if project_name == 'cyclegan_resnet18' or project_name == 'cyclegan_resnet34' or project_name == 'cyclegan_resnet50':
        model = model.module
        model_trunc = create_feature_extractor(model, return_nodes = {'avgpool': 'semantic_feature'})
    if project_name == 'cyclegan_googlenet':
        model = model.module
        model_trunc = create_feature_extractor(model, return_nodes = {'avgpool': 'semantic_feature'})
    if project_name == 'cyclegan_vgg11' or project_name == 'cyclegan_vgg13' or project_name == 'cyclegan_vgg16' or project_name == 'cyclegan_vgg19':
        model = model.module
        model_trunc = create_feature_extractor(model, return_nodes = {'classifier.4': 'semantic_feature'})
    if project_name == 'cyclegan_cnn':
        model = model.module  # get network module from inside its DataParallel wrapper
        model_trunc = create_feature_extractor(model, return_nodes = {'fc1': 'semantic_feature'})
    if project_name == 'cyclegan_mobilenet_v2' or project_name == 'cyclegan_mobilenet_v3_small':
        model = model.module
        model_trunc = create_feature_extractor(model, return_nodes = {'classifier.0': 'semantic_feature'})
    if project_name == 'cyclegan_densenet121':
        model = model.module
        model_trunc = create_feature_extractor(model, return_nodes = {'features.avgpool': 'semantic_feature'})
    if project_name == 'cyclegan_alexnet':
        model = model.module
        model_trunc = create_feature_extractor(model, return_nodes = {'classifier.4': 'semantic_feature'})
    if project_name == 'squeezenet1_0' or project_name == 'squeezenet1_1':
        model_trunc = create_feature_extractor(model, return_nodes = {'classifier.3': 'semantic_feature'})
    if project_name == 'cyclegan_efficientnet_b5' or project_name == 'efficientnet_b4':
        model = model.module
        model_trunc = create_feature_extractor(model, return_nodes = {'avgpool': 'semantic_feature'})
    if project_name == 'cyclegan_convnext_tiny':
        model = model.module
        model_trunc = create_feature_extractor(model, return_nodes = {'avgpool': 'semantic_feature'})
    if project_name == 'cyclegan_mnasnet1_0':
        model = model.module
        model_trunc = create_feature_extractor(model, return_nodes = {'classifier.0': 'semantic_feature'})
    if project_name == 'cyclegan_wide_resnet50_2':
        model = model.module
        model_trunc = create_feature_extractor(model, return_nodes = {'avgpool': 'semantic_feature'})
    return model_trunc


def plot_2d_features(encoding_array, class_to_idx, feature_path, targets):
    marker_list = ['.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', 'P', '*', 'h', 'H', '+', 'x', 'X', 'D', 'd', '|', '_', 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    n_class = len(class_to_idx)          
    palette = sns.hls_palette(n_class)
    sns.palplot(palette)

    random.seed(1234)
    random.shuffle(marker_list)
    random.shuffle(palette)

    tsne = TSNE(n_components=2, learning_rate='auto', n_iter=20000, init='pca')
    X_tsne_2d = tsne.fit_transform(encoding_array)
    
    plt.figure(figsize=(14, 14))
    
    for key, value in class_to_idx.items(): # {'cat': 0, 'dog': 1}
        color = palette[value]
        marker = marker_list[value % len(marker_list)]
    
        indices = np.where(targets==value)
        plt.scatter(X_tsne_2d[indices, 0], X_tsne_2d[indices, 1], color=color, marker=marker, label=key, s=150)
    
    plt.legend(fontsize=16, markerscale=1, bbox_to_anchor=(1, 1))
    plt.xticks([])
    plt.yticks([])
    plt.savefig(feature_path + '/2d_t-sne.png', dpi=200)
    plt.show()


def plot_3d_features(encoding_array, class_to_idx, feature_path, targets):
    marker_list = ['.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', 'P', '*', 'h', 'H', '+', 'x', 'X', 'D', 'd', '|', '_', 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    n_class = len(class_to_idx)          
    palette = sns.hls_palette(n_class)
    sns.palplot(palette)

    random.seed(1234)
    random.shuffle(marker_list)
    random.shuffle(palette)

    tsne = TSNE(n_components=3, learning_rate='auto', n_iter=10000, init='pca')
    X_tsne_3d = tsne.fit_transform(encoding_array)

    fig = plt.figure(figsize=(14, 14))
    ax = Axes3D(fig)

    for key, value in class_to_idx.items(): # {'cat': 0, 'dog': 1}
        color = palette[value]
        marker = marker_list[value % len(marker_list)]
    
        indices = np.where(targets==value)
        ax.scatter(X_tsne_3d[indices, 0], X_tsne_3d[indices, 1], X_tsne_3d[indices, 2], color=color, marker=marker, label=key, s=150)
    
    plt.legend(fontsize=16, markerscale=1, bbox_to_anchor=(1, 1))
    plt.savefig(feature_path + '/3d_t-sne.png', dpi=200)
    plt.show()