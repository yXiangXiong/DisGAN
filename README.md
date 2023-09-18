# linux environment（python=3.8）:
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia  

conda install tqdm  

conda install matplotlib==3.3.4  

conda install seaborn  

conda install scikit-learn  

# Running command
python train.py --cuda  

python test.py --cuda
