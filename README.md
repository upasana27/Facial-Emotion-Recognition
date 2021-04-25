# Facial Emotion Recognition Using Deep CNN'S
This package presents an ensemble of trained CNN'S, to obtain a solution for the Kaggle FER-2013 Challenge: Challenges in Representation Learning. https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/overview 
Testing accuracy obtained is 65.2%, as compared winning score of 71.1%.


## Prerequisites
- Python >=3.7.10
- Keras >=2.4.0
- Tensorflow2.x
- Matplotlib >=3.2.2
- Pandas >=1.1.5
- Numpy >=1.19.5
- Sklearn

### How to Install
```bash
pip install requirements.txt
```

## Dataset 
The models were trained on FER 2013 Dataset. Train.csv has 28,709 samples of 48x48 pixel grayscale images of faces each showing one of the emotions: 
- 0=Angry 
- 1=Disgust
- 2=Fear
- 3=Happy
- 4=Sad 
- 5=Surprise
- 6=Neutral 

You can download the dataset from https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data.


## Method

This project, developed using the Keras API, consists of classifying each input image into one of the emotion categories. 
Inspired from results of [1] and [2], used an ensemble of VGG-16[3]  and Inception-v1[4] models, trained from scratch using different initializations and training methods.  Fine-tuning was carried out on the pre-trained VGG-16 model, to take advantage of previous knowledge to improve our predictions. 

| Model       | Initial Weights           | No. Of Trainable Parameters  | 
| ------------- |:-------------:| -----:|
| VGG-16    | None | 9,416,903 |
| Inception-v1     | None      |   2,149,159 |
| VGG-16 | ImageNet    |    2,773,095 |



## Evaluation


The F1-score metric is generated to get an accurate understanding of the model's performance. The confusion matrix is generated, on the basis of which precision and recall is calculated.  We obtained a *precision of 66.4%* , *recall of 65%* and an **F1-score of 65.12%**. 
#### CONFUSION MATRIX:

| |   |      |   |     |     |        |     |
|----------|-------|---------|------|-------|-------|----------|---------|
| Anger    | 348   | 12      | 47   | 52    | 14    | 0        | 63      |
| Disgust  | 14    | 290     | 6    | 18    | 31    | 0        | 27      |
| Fear     | 120   | 1       | 216  | 27    | 52    | 0        | 114     |
| Happy    | 49    | 12      | 11   | 689   | 5     | 1        | 26      |
| Sad      | 38    | 32      | 63   | 20    | 165   | 1        | 89      |
| Surprise | 1     | 0       | 1    | 2     | 3     | 14       | 19      |
| Neutral  | 69    | 9       | 29   | 20    | 20    | 1        | 332     |
|          | Anger | Disgust | Fear | Happy | Sad   | Surprise | Neutral |



## Training
After downloading the dataset, put train.csv is /data folder and run the following scripts to train the models.

To train inception-type network from scratch;
```bash
python3 train_inception.py
```
To train VGG-type network from scratch;
```bash
python3 train_vgg.py
```
To fine-tune layers added on top of a VGG-16 convolutional base;

```bash
python3 train_vgg_transfer_learning.py
```
If you want to train using a custom dataset, resize pictures to a 48*48 pixel size using /preprocess/data_resize.py.

## Testing
 Pretrained models are placed in the /models folders. If you want to include your own model in the network of models, add model in *model_i.h5* format, and assign number of models. 
#### Edit script at:
 ```bash
 int num_of_models;
 ```
 #### Run the following script to generate confusions matrix and f1-score.
  ```bash
 python3 test_ensemble_model.py
 ```

## References
<a id="1">[1]</a> 
S. Singh and F. Nasoz, "Facial Expression Recognition with Convolutional Neural Networks," 2020 10th Annual Computing and Communication Workshop and Conference (CCWC), 2020, pp. 0324-0328, doi: 10.1109/CCWC47524.2020.9031283.

<a id="2">[2]</a> 
Kusuma Negara, I Gede Putra & Jonathan, Jonathan & Lim, Andreas. (2020). Emotion Recognition on FER-2013 Face Images Using Fine-Tuned VGG-16. Advances in Science, Technology and Engineering Systems Journal. 5. 315-322. 10.25046/aj050638. 

<a id="3">[3]</a> 
Zisserman, Andrew. (2014). Very Deep Convolutional Networks for Large-Scale Image Recognition. arXiv 1409.1556. 

<a id="4">[4]</a> 
C. Szegedy et al., "Going deeper with convolutions," 2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2015, pp. 1-9, doi: 10.1109/CVPR.2015.7298594.
