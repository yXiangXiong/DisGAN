# Introduction
we developed a novel generative model (HyperDisGAN) that effectively controls the locations of generated cross-domain and intra-domain samples. The locations are respectively defined using the vertical distances of the cross-domain samples to the optimal hyperplane and the horizontal distances of the intra-domain samples to the source samples, which are determined by \emph{Hinge Loss} and \emph{Pythagorean Theorem}.

# linux environment（python=3.8）:
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia  

conda install tqdm  

conda install matplotlib==3.3.4  

conda install seaborn  

conda install scikit-learn  

# Running command
python train.py --cuda  

python test.py --cuda
