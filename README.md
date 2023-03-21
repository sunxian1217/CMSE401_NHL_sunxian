# Software Abstract

TensorFlow is an interface for expressing machine learning algorithms and an implementation for executing such algorithms.It makes machine learning and developing neural networks faster and easier.It provides a flexible and powerful platform for building and training models, and supports a variety of programming languages, including Python, C++, and Java. TensorFlow supports both CPU and GPU computation and can be used on a variety of platforms. Additionally, TensorFlow offers a range of tools and libraries for data preprocessing, visualization, and model deployment.PyTorch is also an open source machine learning framework that is widely used for building and training neural networks and it is relatively easier to use than tensorflow. It is initially developed by Facebook artificial-intelligence research group, and Uber’s Pyro software for probabilistic programming which is built on it.PyTorch also supports both CPU and GPU computation and has a rich set of tools for data loading, preprocessing, analyzing and visualization. Tensorflow and pytorch both can make machine learning and developing neural networks faster and easier. 

# Installation
1. Install Anaconda
```curl -O https://repo.anaconda.com/archive/Anaconda3-2023.03-Linux-x86_64.sh```
```bash Anaconda3-2023.03-Linux-x86_64```
```Do you accept the license terms [yes|no]```? Type ```yes``` to accept
Choose the installation location
```Do you wish the installer to initialize Anaconda3 by running conda init? [yes|no]```, type ```no```
Navigate to your home space and open the .bashrc file with an editor e.g., run the commands
```cd $HOME```
```vim .bashrc```
Check for initialization code starts with ```>>> conda init >>>``` and ends with ```<<< conda init <<<```; while setup code may contain commands like export.Remove or comment out this code by adding # in front of each line.
If the Anaconda installation path is ```$HOME/anaconda3```, skip this line; If not, add the command line ```export CONDA3PATH=<Anaconda3 installation path>```

2. Tensorflow
```module purge```
```module load Conda/3```
```conda create --name tf```
```source activate tf```
```conda install -c conda-forge tensorflow```
```conda deactivate```

3. Pytorch
```module purge```
```module load Conda/3```
```conda create -n tractseg```
```conda activate tractseg```
```conda install -c mrtrix3 mrtrix3```
```conda install pytorch```

# References
In order to install Anaconda, follow the instructions on the icer website: https://docs.icer.msu.edu/Using_conda/
For installing pytorch, follow the instructions in this link: https.://docs.icer.msu.edu/LabNotebook_TractSeg/#lab-notebook-installing-tractseg-on-hpcc-using-conda. For tensorflow, use this link: https://docs.icer.msu.edu/Installing_TensorFlow_using_anaconda/

https://www.tutorialspoint.com/pytorch/pytorch_introduction.htm