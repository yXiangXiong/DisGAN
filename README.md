# Introduction
Conventional transformation methods primarily rely on the intra-domain knowledge. On the other hand, advanced generative adversarial networks generate cross-domain images lacking variety, particularly in small-scale datasets. In light of these challenges, we propose that accurately controlling the variation degrees of synthetic samples can reshape the decision boundary in hyperplane space for the downstream classifications.

We developed a HyperDisGAN effectively controlling the locations of generated cross-domain and intra-domain samples. The locations are respectively defined using the vertical distances of the cross-domain samples to the optimal hyperplane and the horizontal distances of the intra-domain samples to the source samples, which are determined by \emph{Hinge Loss} and \emph{Pythagorean Theorem}.

# linux environment（python=3.8）:
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia  

conda install tqdm  

conda install matplotlib==3.3.4  

conda install seaborn  

conda install scikit-learn  

# Running command
python train.py --cuda  --dataset_name covid_ct --project_name cyclegan_convnext_tiny --aux_classfier pretrained_aux_convnext_tiny.pth --lambda_vertical 0.01 --lambda_horizontal 0.01

python test.py --cuda  --dataset_name covid_ct --project_name cyclegan_convnext_tiny --aux_classfier pretrained_aux_convnext_tiny.pth --classifier downstream_classifier.pth
