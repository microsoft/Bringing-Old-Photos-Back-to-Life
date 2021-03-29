# Old Photo Restoration (Official PyTorch Implementation)

<img src='imgs/0001.jpg'/>

### [Project Page](http://raywzy.com/Old_Photo/) | [Paper (CVPR version)](https://arxiv.org/abs/2004.09484) | [Paper (Journal version)](https://arxiv.org/pdf/2009.07047v1.pdf) | [Pretrained Model](https://hkustconnect-my.sharepoint.com/:f:/g/personal/bzhangai_connect_ust_hk/Em0KnYOeSSxFtp4g_dhWdf0BdeT3tY12jIYJ6qvSf300cA?e=nXkJH2) | [Colab Demo](https://colab.research.google.com/drive/1NEm6AsybIiC5TwTU_4DqDkQO0nFRB-uA?usp=sharing) :fire:

**Bringing Old Photos Back to Life, CVPR2020 (Oral)**

**Old Photo Restoration via Deep Latent Space Translation, PAMI Under Review**

[Ziyu Wan](http://raywzy.com/)<sup>1</sup>,
[Bo Zhang](https://www.microsoft.com/en-us/research/people/zhanbo/)<sup>2</sup>,
[Dongdong Chen](http://www.dongdongchen.bid/)<sup>3</sup>,
[Pan Zhang](https://panzhang0212.github.io/)<sup>4</sup>,
[Dong Chen](https://www.microsoft.com/en-us/research/people/doch/)<sup>2</sup>,
[Jing Liao](https://liaojing.github.io/html/)<sup>1</sup>,
[Fang Wen](https://www.microsoft.com/en-us/research/people/fangwen/)<sup>2</sup> <br>
<sup>1</sup>City University of Hong Kong, <sup>2</sup>Microsoft Research Asia, <sup>3</sup>Microsoft Cloud AI, <sup>4</sup>USTC

## Notes of this project
The code originates from our research project and the aim is to demonstrate the research idea, so we have not optimized it from a product perspective. And we will spend time to address some common issues, such as out of memory issue, limited resolution, but will not involve too much in engineering problems, such as speedup of the inference, fastapi deployment and so on. **We welcome volunteers to contribute to this project to make it more usable for practical application.**

**We are improving the algorithm so as to process high resolution photos. It takes time and please stay tuned.**

## News
Training code is available and welcome to have a try and learn the training details. 

You can now play with our [Colab](https://colab.research.google.com/drive/1NEm6AsybIiC5TwTU_4DqDkQO0nFRB-uA?usp=sharing) and try it on your photos. 

## Requirement
The code is tested on Ubuntu with Nvidia GPUs and CUDA installed. Python>=3.6 is required to run the code.

## Installation

Clone the Synchronized-BatchNorm-PyTorch repository for

```
cd Face_Enhancement/models/networks/
git clone https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
cp -rf Synchronized-BatchNorm-PyTorch/sync_batchnorm .
cd ../../../
```

```
cd Global/detection_models
git clone https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
cp -rf Synchronized-BatchNorm-PyTorch/sync_batchnorm .
cd ../../
```

Download the landmark detection pretrained model

```
cd Face_Detection/
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
cd ../
```

Download the pretrained model from Azure Blob Storage, put the file `Face_Enhancement/checkpoints.zip` under `./Face_Enhancement`, and put the file `Global/checkpoints.zip` under `./Global`. Then unzip them respectively.

```
cd Face_Enhancement/
wget https://facevc.blob.core.windows.net/zhanbo/old_photo/pretrain/Face_Enhancement/checkpoints.zip
unzip checkpoints.zip
cd ../
cd Global/
wget https://facevc.blob.core.windows.net/zhanbo/old_photo/pretrain/Global/checkpoints.zip
unzip checkpoints.zip
cd ../
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## How to use?

**Note**: GPU can be set 0 or 0,1,2 or 0,2; use -1 for CPU

### 1) Full Pipeline

You could easily restore the old photos with one simple command after installation and downloading the pretrained model.

For images without scratches:

```
python run.py --input_folder [test_image_folder_path] \
              --output_folder [output_path] \
              --GPU 0
```

For scratched images:

```
python run.py --input_folder [test_image_folder_path] \
              --output_folder [output_path] \
              --GPU 0 \
              --with_scratch
```

Note: Please try to use the absolute path. The final results will be saved in `./output_path/final_output/`. You could also check the produced results of different steps in `output_path`.

### 2) Scratch Detection

Currently we don't plan to release the scratched old photos dataset with labels directly. If you want to get the paired data, you could use our pretrained model to test the collected images to obtain the labels.

```
cd Global/
python detection.py --test_path [test_image_folder_path] \
                    --output_dir [output_path] \
                    --input_size [resize_256|full_size|scale_256]
```

<img src='imgs/scratch_detection.png'>

### 3) Global Restoration

A triplet domain translation network is proposed to solve both structured degradation and unstructured degradation of old photos.

<p align="center">
<img src='imgs/pipeline.PNG' width="50%" height="50%"/>
</p>

```
cd Global/
python test.py --Scratch_and_Quality_restore \
               --test_input [test_image_folder_path] \
               --test_mask [corresponding mask] \
               --outputs_dir [output_path]

python test.py --Quality_restore \
 --test_input [test_image_folder_path] \
 --outputs_dir [output_path]
```

<img src='imgs/global.png'>


### 4) Face Enhancement

We use a progressive generator to refine the face regions of old photos. More details could be found in our journal submission and `./Face_Enhancement` folder.

<p align="center">
<img src='imgs/face_pipeline.jpg' width="60%" height="60%"/>
</p>


<img src='imgs/face.png'>

> *NOTE*: 
> This repo is mainly for research purpose and we have not yet optimized the running performance. 
> 
> Since the model is pretrained with 256*256 images, the model may not work ideally for arbitrary resolution.

## How to train?

### 1) Create Training File

Put the folders of VOC dataset, collected old photos (e.g., Real_L_old and Real_RGB_old) into one shared folder. Then
```
cd Global/data/
python Create_Bigfile.py
```
Note: Remember to modify the code based on your own environment.

### 2) Train the VAEs of domain A and domain B respectively

```
cd ..
python train_domain_A.py --use_v2_degradation --continue_train --training_dataset domain_A --name domainA_SR_old_photos --label_nc 0 --loadSize 256 --fineSize 256 --dataroot [your_data_folder] --no_instance --resize_or_crop crop_only --batchSize 100 --no_html --gpu_ids 0,1,2,3 --self_gen --nThreads 4 --n_downsample_global 3 --k_size 4 --use_v2 --mc 64 --start_r 1 --kl 1 --no_cgan --outputs_dir [your_output_folder] --checkpoints_dir [your_ckpt_folder]

python train_domain_B.py --continue_train --training_dataset domain_B --name domainB_old_photos --label_nc 0 --loadSize 256 --fineSize 256 --dataroot [your_data_folder]  --no_instance --resize_or_crop crop_only --batchSize 120 --no_html --gpu_ids 0,1,2,3 --self_gen --nThreads 4 --n_downsample_global 3 --k_size 4 --use_v2 --mc 64 --start_r 1 --kl 1 --no_cgan --outputs_dir [your_output_folder]  --checkpoints_dir [your_ckpt_folder]
```
Note: For the --name option, please ensure your experiment name contains "domainA" or "domainB", which will be used to select different dataset.

### 3) Train the mapping network between domains

Train the mapping without scratches:
```
python train_mapping.py --use_v2_degradation --training_dataset mapping --use_vae_which_epoch 200 --continue_train --name mapping_quality --label_nc 0 --loadSize 256 --fineSize 256 --dataroot [your_data_folder] --no_instance --resize_or_crop crop_only --batchSize 80 --no_html --gpu_ids 0,1,2,3 --nThreads 8 --load_pretrainA [ckpt_of_domainA_SR_old_photos] --load_pretrainB [ckpt_of_domainB_old_photos] --l2_feat 60 --n_downsample_global 3 --mc 64 --k_size 4 --start_r 1 --mapping_n_block 6 --map_mc 512 --use_l1_feat --niter 150 --niter_decay 100 --outputs_dir [your_output_folder] --checkpoints_dir [your_ckpt_folder]
```


Traing the mapping with scraches:
```
python train_mapping.py --no_TTUR --NL_res --random_hole --use_SN --correlation_renormalize --training_dataset mapping --NL_use_mask --NL_fusion_method combine --non_local Setting_42 --use_v2_degradation --use_vae_which_epoch 200 --continue_train --name mapping_scratch --label_nc 0 --loadSize 256 --fineSize 256 --dataroot [your_data_folder] --no_instance --resize_or_crop crop_only --batchSize 36 --no_html --gpu_ids 0,1,2,3 --nThreads 8 --load_pretrainA [ckpt_of_domainA_SR_old_photos] --load_pretrainB [ckpt_of_domainB_old_photos] --l2_feat 60 --n_downsample_global 3 --mc 64 --k_size 4 --start_r 1 --mapping_n_block 6 --map_mc 512 --use_l1_feat --niter 150 --niter_decay 100 --outputs_dir [your_output_folder] --checkpoints_dir [your_ckpt_folder] --irregular_mask [absolute_path_of_mask_file]
```



## To Do
- [x] Clean testing code
- [x] Release pretrained model
- [x] Collab demo
- [ ] Replace face detection module (dlib) with RetinaFace
- [x] Release training code


## Citation

If you find our work useful for your research, please consider citing the following papers :)

```bibtex
@inproceedings{wan2020bringing,
title={Bringing Old Photos Back to Life},
author={Wan, Ziyu and Zhang, Bo and Chen, Dongdong and Zhang, Pan and Chen, Dong and Liao, Jing and Wen, Fang},
booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
pages={2747--2757},
year={2020}
}
```

```bibtex
@article{wan2020old,
  title={Old Photo Restoration via Deep Latent Space Translation},
  author={Wan, Ziyu and Zhang, Bo and Chen, Dongdong and Zhang, Pan and Chen, Dong and Liao, Jing and Wen, Fang},
  journal={arXiv preprint arXiv:2009.07047},
  year={2020}
}
```

If you are also interested in the legacy photo/video colorization, please refer to [this work](https://github.com/zhangmozhe/video-colorization).

## Maintenance

This project is currently maintained by Ziyu Wan and is for academic research use only. If you have any questions, feel free to contact raywzy@gmail.com.

## License

The codes and the pretrained model in this repository are under the MIT license as specified by the LICENSE file. We use our labeled dataset to train the scratch detection model.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
