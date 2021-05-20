# Learning to Localize Sound Source in Visual Scenes [CVPR 2018,TPAMI 2020]

The codebase is the re-implementation of the code that was used in  CVPR 2018 <a href="https://openaccess.thecvf.com/content_cvpr_2018/papers/Senocak_Learning_to_Localize_CVPR_2018_paper.pdf">Learning to Localize Sound Source in Visual Scenes</a> and TPAMI <a href="https://arxiv.org/abs/1911.09649">Learning to Localize Sound Source in Visual Scenes: Analysis and Applications</a> papers. Original code was written in the early version of Tensorflow so that we re-implemented it in PyTorch for the community.

## Getting started
- tqdm
- scipy

## Preparation

- Training Data

  - We used 144k samples from Flickr-SoundNet dataset for training as it is mentioned in the paper.
  - Sound features are directly obtained from <a href="https://github.com/cvondrick/soundnet">SoundNet</a> implementation. We apply average pooling on the output of "Object" branch of Conv8 layer and use it as sound feature in our architecture.
  - To be able to use our dataloader (Sound_Localization_Dataset.py);
    - Each sample folder should contain frames as `.jpg` and audio features as `.mat` extensions. For details please refer to `Sound_Localization_Dataset.py`
      - `/hdd/SoundLocalization/dataset/12015590114.mp4/frame1.jpg`
      - `/hdd/SoundLocalization/dataset/12015590114.mp4/12015590114.mat`

- The Sound Localization Dataset (Annotated Dataset)

    The Sound Localization dataset can be downloaded from the following link:

    https://drive.google.com/open?id=1P93CTiQV71YLZCmBbZA0FvdwFxreydLt

    This dataset contains 5k image-sound pairs and their annotations in XML format.
    Each XML file has annotations of 3 annotators.

    test_list.txt file includes the id of every pair that is used for testing.

## Training

```
python sound_localization_main.py --dataset_file /hdd3/Old_Machine/sound_localization/semisupervised_train_list.txt  
--val_dataset_file /hdd3/Old_Machine/sound_localization/supervised_test_list.txt 
--annotation_path /hdd/Annotations/xml_box_20  --mode train --niter 10 --batchSize 30 --nThreads 8 --validation_on True 
--validation_freq 1 --display_freq 1 --save_latest_freq 1 --name semisupervised_sound_localization_t1 
--optimizer adam --lr_rate 0.0001 --weight_decay 0.0
```
## Pretrained Model

We provide pre-trained model for semisupervised architecture. Accuracy is slightly lower than reported number in the paper (Because of re-implementation in another framework). You can download the model from <a href="https://drive.google.com/file/d/1JMD-LjHbfZ_yUy-l6tjbI46yYQfH8oS4/view?usp=sharing">here</a>.

If you end up using our code or dataset, we ask you to cite the following papers:

    @InProceedings{Senocak_2018_CVPR,
    author = {Senocak, Arda and Oh, Tae-Hyun and Kim, Junsik and Yang, Ming-Hsuan and So Kweon, In},
    title = {Learning to Localize Sound Source in Visual Scenes},
    booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2018}
    }
    @article{Senocak_2020_TPAMI,
    title = {Learning to Localize Sound Source in Visual Scenes: Analysis and Applications},
    author = {Senocak, Arda and Oh, Tae-Hyun and Kim, Junsik and Yang, Ming-Hsuan and So Kweon, In},
    journal = {TPAMI},
    year = {2020},
    publisher = {IEEE}
    }


Image-sound pairs are collected by using the Flickr-SoundNet dataset. Thus, please cite the Yahoo dataset **[the Yahoo dataset](https://webscope.sandbox.yahoo.com/catalog.php?datatype=i&did=67&guccounter=1)** and **[SoundNet](http://projects.csail.mit.edu/soundnet/)** paper as well.

The dataset and the code must be used for research purposes only.
