# A Histogram-Driven Generative Adversarial Network for Multimodal Brain Image Synthesis

This repository is the work of "A Histogram-Driven Generative Adversarial Network for Multimodal Brain Image Synthesis" based on **pytorch** implementation. 

Note that, the HisGAN model in `models` file will open when the paper is accept.

You could click the link to access the [paper](https://arxiv.org/). The multimodal FeTS dataset could be acquired from [here](https://www.med.upenn.edu/sbia/brats2018.html).


<!-- <div  align="center">  
 <img src="https://github.com/China-LiuXiaopeng/BraTS-DMFNet/blob/master/fig/Architecture.jpg"
     align=center/>
</div> --> -->

 <!-- <center>Architecture of 3D DMFNet</center>


## Requirements
- python 3.6
- pytorch 1.8.1 or later CUDA version (ARGAN model requires 1.8.1 or later)
- torchvision
- nibabel
- SimpleITK
- matplotlib
- fire
- Pillow


### Training

Multiply gpus training is recommended. The total training time take less than 20 hours in gtxforce 2080Ti. Training like this:

```
python3 -u main.py train --model='ResCycleGANModel' --A='t1' --B='t2'
```

### Test (CPU version)

You could obtain the resutls as paper reported by running the following code:

```
python3 main.py predict --gpu_ids='' --model='HisGAN_EMANet_Histloss' --A='t1' --B='t2' --load_iter=200 --dataroot='{your test set}'
```
Then make a submission to the online evaluation server.

## Citation

If you use our code or model in your work or find it is helpful, please cite the paper:
```
***Unknown***
```

## Acknowledge
None.

