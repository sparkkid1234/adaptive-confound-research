# Adaptive Confound
Code files and notebooks for the paper "Discovering and controlling for latent confounds in text classification using adversarial domain adaption" (Landeiro V., Tran T., Culotta A., SDM19)

## Replicate the environment

`R>` indicates an action to execute on the work (remote) server. `L>` indicates that this action should be run on the local machine.
If you do not use a work server, then all the commands should be run on your local machine.

1. If not already done, install anaconda: 
```
R> wget --no-check-certificate https://repo.anaconda.com/archive/Anaconda3-5.2.0-Linux-x86_64.sh
R> bash Anaconda3-5.2.0-Linux-x86_64.sh
```
2. Create the environment and activate it
```
R> conda env create -n py36 -f environment.yml
R> source activate py36
```
3. Note where `jupyter` is located under this environment and use it locally with the jupyter_remote.py script to start a new jupyter notebook server.
```
R> which jupyter
L> jupyter_remote.py <server> --jupyter-path=<jupyter_path> --port=<port> -n
```
