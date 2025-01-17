** Note: Sheila has modified a few things in this repo + readme to address some compatibility and installation issues. 

## Requirements

* pytorch>=0.4.1 (Note: There will likely be a bunch of imports missing that you'll run into...)

    ```shell
    pip install -r requirements.txt
    ```

## Data Preparation

### COCOA dataset proposed in [Semantic Amodal Segmentation](http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhu_Semantic_Amodal_Segmentation_CVPR_2017_paper.pdf).

1. Download COCO2014 train and val images from [here](http://cocodataset.org/#download) and unzip.

2. Download COCOA annotations from [here](https://github.com/Wakeupbuddy/amodalAPI) and untar. (Note: I have included it in the data folder already in this modified repo)

3. Ensure the COCOA folder looks like:

    ```
    COCOA/
      |-- train2014/
      |-- val2014/
      |-- annotations/
        |-- COCO_amodal_train2014.json
        |-- COCO_amodal_val2014.json
        |-- COCO_amodal_test2014.json
        |-- ...
    ```

4. Create symbolic link:
    ```
    cd data
    ln -s /path/to/COCOA
    ```

### KINS dataset proposed in [Amodal Instance Segmentation with KINS Dataset](http://openaccess.thecvf.com/content_CVPR_2019/papers/Qi_Amodal_Instance_Segmentation_With_KINS_Dataset_CVPR_2019_paper.pdf).

1. Download left color images of object data in KITTI dataset from [here](http://www.cvlibs.net/download.php?file=data_object_image_2.zip) and unzip.

2. Download KINS annotations from [here](https://drive.google.com/drive/folders/1hxk3ncIIoii7hWjV1zPPfC0NMYGfWatr?usp=sharing) corresponding to [this commit](https://github.com/qqlu/Amodal-Instance-Segmentation-through-KINS-Dataset/tree/fb7be3fcedc96d4a6e20d4bb954010ec1b4f3194).

3. Ensure the KINS folder looks like:

    ```
    KINS/
      |-- training/image_2/
      |-- testing/image_2/
      |-- instances_train.json
      |-- instances_val.json
    ```

4. Create symbolic link:
    ```
    cd data
    ln -s /path/to/KINS
    ```

## Train

To train with the default run and the COCOA dataset. 

Note: The following is what worked for Sheila and her team with the CUDA 11.8 configured and 4 GPUs. 
```
torchrun --nproc-per-node 4 --master-port 9918 main.py --config experiments/COCOA/pcnet_m/config_train_std_no_rgb_gaussian.yaml --launcher pytorch --exp_path experiments/COCOA/pcnet_m_std_no_rgb_gaussian
```

### train PCNet-M

1. Train (taking COCOA for example).

    ```
    ./train_pcnet_m_std_no_rgb_gaussian.sh
    ```

2. Monitoring status and visual results using tensorboard.

    ```
    sh tensorboard.sh $PORT
    ```

## Evaluate

* Execute:

    ```shell
    ./test_pcnet_m.sh
    ```
or
```
torchrun --nproc-per-node 1 main.py --config experiments/COCOA/pcnet_m/config_train_std_no_rgb_gaussian.yaml --launcher pytorch --load-iter 30000 --validate --exp_path experiments/COCOA/pcnet_m_std_no_rgb_gaussian
```

## Bibtex for Original work

```
@InProceedings{Nguyen_2021_ICCV,
    author    = {Nguyen, Khoi and Todorovic, Sinisa},
    title     = {A Weakly Supervised Amodal Segmenter With Boundary Uncertainty Estimation},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {7396-7405}
}
```

## Original Acknowledgement

1. We developed our approach based on the code from [https://github.com/XiaohangZhan/deocclusion/](https://github.com/XiaohangZhan/deocclusion/)

2. We used the code and models of [GCA-Matting](https://github.com/Yaoyi-Li/GCA-Matting) in our demo.

3. We modified some code from [pytorch-inpainting-with-partial-conv](https://github.com/naoto0804/pytorch-inpainting-with-partial-conv) to train the PCNet-C.
