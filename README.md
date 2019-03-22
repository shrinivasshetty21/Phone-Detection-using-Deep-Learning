## Phone Detection using Deep Learning Model.
In this project I am tasked with identifying the centre pixel location for an object(phone) present in an image. I have used deep learning model with convolutional neural network along with a feed forward network to achieve this task.
### Getting Started

Use the following instructions to make your system ready to run the code.

### Dependencies

Project is run using:
- Windows 10
- Python 3.7.1
- TensorFlow 1.13.1
- Keras 2.2.4

### Installation

A requirements.txt is added to the repository which can be used to install the dependencies using the following code in your virtual environment.

```
pip install -r requirements.txt
```

### Inside the Repo

The repository contains python scripts to train and test the model.
It also contain jupyter notebook, that implements the model for better understanding the architecture and tweaking the model. This repo contains weights of the model created using the jupyter notebook along with visualizations of the results.

### Files
- Find Phone DL Model.ipynb : Jupyter Notebook with code implementations.
- find_phone.py : Python script to test the model.
- train_phone_finder.py : Python scrip to train the model.
- requirements.txt : Contains list of packages required to run the scripts.
- readme.md

### Folders
- Results : Contains weights of the model and visualizations.

### Training Script
- train_phone_finder.py : takes a single command line argument which is a path to the folder with labeled images and labels.txt
```
python train_phone_finder.py ~/find_phone
```

### Testing Script
- find_phone.py : takes a single command line argument which is a path to the jpeg image to be tested.

```
 python find_phone.py ~/find_phone_test_images/108.jpg
```
Make sure the weights created by the training script is located in the above mentioned directory of the test image.
Below is the expected output of the testing script.
```
Phone in image 108.jpg is located at x-y coordinates given below.

0.4523 0.5423
```
### Additional Notes

The performance of this model can be improved using more images and training the model without resizing the images. I trained 100 images of 64x64 dimension to train the model. The model can be improved with a better architecture. This problem can also be solved using Saliency detection algorithm which would locate the salient part of the image from the background. In this case the salient part of the image would be the phone.
