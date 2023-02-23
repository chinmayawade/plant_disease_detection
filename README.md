# Plant Disease Detection using ResMLP Feedforward Networks for Image Classification with Data-Efficient Training

ResMLP, also known as the Residual Multi-layer Perceptron, is a neural network architecture that has shown promising results in various computer vision tasks. To use ResMLP for plant disease detection, we can follow the following steps:

**Collect and preprocess the dataset:**  Collect a large dataset of images of healthy and diseased plants, and preprocess them by resizing, cropping, and normalizing the images.

**Split the dataset:** Divide the dataset into training, validation, and testing sets.

**Train the ResMLP model:** Initialize the ResMLP model and train it using the training set. During training, use data augmentation techniques such as random cropping, flipping, and rotation to increase the robustness of the model.

**Tune hyperparameters:** Tune hyperparameters such as learning rate, batch size, and number of training epochs using the validation set to improve the performance of the model.

**Evaluate the model:** Finally, evaluate the performance of the trained model on the testing set by calculating metrics such as accuracy, precision, recall, and F1-score.

Overall, ResMLP can be a powerful tool for plant disease detection, but the quality and size of the dataset, as well as the effectiveness of the data preprocessing and augmentation, can have a significant impact on the performance of the model.

## Key Points ## 
1. Importing Libraries and Dataset
2. Data Preprocessing and Visualization
3. Data Augmentation
4. Modeling for MobileNetV3Small
5. Modeling using ResMLP
6. Training using Deep Learning Neural Network
7. Plotting Results
8. Confusion Matrix

## Imported Libraries ##


```ruby
import cv2, os, shutil, math
from keras.layers import *
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score, classification_report, confusion_matrix
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
```



## Collected Various Dataset Leaves

| Blotch Leaves  |  Rot Leaves |
| ------------- | ------------- |
| Necrotic leaf blotch is caused by the rapid synthesis of gibberellins triggered by environmental factors. Treating trees with zinc-containing fungicides (e.g., Ziram) or foliar sprays containing zinc nutrients can decrease the severity of necrotic leaf blotch.  |  Black rot is caused by the fungus Diplodia seriata (syn Botryosphaeria obtusa). The fungus can infect dead tissue as well as living trunks, branches, leaves and fruits. The black rot fungi survive Minnesota winters in branch cankers and mummified fruit (shriveled and dried fruit) attached to the tree.  |



| Scab Leaves  | Healthy Leaves |
| ------------- | ------------- |
| Leaves infected with the apple scab fungus usually fall from trees in autumn or early winter. The fungus continues to live within the leaves during winter, forming small, flask-shaped bodies, in which spores (ascospores) develop. These ascospores mature in spring and are forcibly ejected during spring rains  | The leaves are simple, oval in shape, have small serrations along the margin, and are arranged alternately along the branches. A typical leaf is 2 – 5 inches long, 1.5 – 2.5 inches wide, and has an acuminate tip (Petrides 1972). The twigs, buds, and undersides of the leaves usually have white pubescence  |


## Data Preprocessing and Visualization ## 
### Preprocessed the data based on Image Quality, File Extentions such as .png,jpeg,jpg etc labelled the data and sorted it as per the classes
Imagine a large table filled with rows of data representing thousands of images. Each row contains information about a single image, such as its file name, file extension, file path, and labels or classes assigned to it.

As you scroll through the table, you notice that some of the images have *bad quality* or resolution. They may appear *blurry or pixelated, making it difficult to make out the details.* Other images may have *incorrect file extensions or file paths*, which can make it challenging to locate and access them.

However, the *labels* or classes assigned to each image provide useful information about the content of the image. For example, an image might be labeled as "cat" or "dog" if it contains a picture of a cat or a dog. These labels allow you to quickly sort and filter the images based on their content.

You also notice that some images have multiple *labels or classes* assigned to them. For instance, an image might be labeled as both "cat" and "indoor" if it shows a cat inside a house. This additional information can be helpful in training machine learning models to recognize and categorize different types of images.

Overall, visualizing this data allows you to gain a better understanding of the images you are working with and the information associated with them. By examining the image quality, file information, and labels or classes, you can make more informed decisions about how to organize and utilize the images in your project.


## Modeling for MobileNetV3Small ## 
MobileNetV3Small is a type of convolutional neural network architecture that is designed for mobile and embedded devices with limited computational resources. If you want to build a model using MobileNetV3Small, there are a few steps you can follow:

**Choose a framework:** There are several deep learning frameworks that support MobileNetV3Small, such as TensorFlow, PyTorch, and Keras. Choose a framework that you are comfortable with and that supports MobileNetV3Small.

**Preprocess your data:** Preprocessing your data is an important step in any machine learning project. Ensure that your data is in the correct format for MobileNetV3Small, which is typically a series of image tensors with a specific size and number of channels.

**Build the model:** MobileNetV3Small is a pre-trained model, meaning it has already been trained on a large dataset. You can either use the pre-trained weights or fine-tune the model for your specific use case. You will need to add the appropriate number of output layers for your particular classification problem.

**Train the model:** Train your model on your training data. If you are fine-tuning a pre-trained model, you may not need to train for as many epochs as you would with a model trained from scratch.

Evaluate the model: Once you have trained your model, evaluate its performance on your validation set. If the model is not performing well, you may need to adjust the hyperparameters or make changes to the model architecture.

**Deploy the model:** Once you are satisfied with the model's performance, you can deploy it to your mobile or embedded device. Be sure to optimize the model for inference on your specific device to ensure optimal performance.

Overall, MobileNetV3Small is a powerful architecture that can be used to build machine learning models for mobile and embedded devices. By following these steps, you can build and deploy your own model using MobileNetV3Small.

    
**Confusion Marix**
-----------------------
![download](https://user-images.githubusercontent.com/90987160/221009170-e2d433c7-0ef5-4787-a7a3-1be49287babd.png)

**Classification Report:**
----------------------    
| 		      | precision | recall | f1-score | Support |
| ------------- | ------------- | ------------- | ------------- |  ------------- |
| APPLE ROT LEAVES  | 0.3750  | 0.2727  | 0.3158   | 11  |
| HEALTHY LEAVES  | 0.2727  |0.7500  |  0.4000  | 4  |
| LEAF BLOTCH | 0.4286  | 0.5455 |0.4800  | 11  |
| SCAB LEAVES  |0.5556  | 0.3125 | 0.4000  | 16  |
| Accuracy  |  |  | 0.4048    | 42  |
|    macro avg  |  0.4080  | 0.4702  |  0.3989  | 42  |
| weighted avg | 0.4481  | 0.4048 |0.3989  | 42  |



     
