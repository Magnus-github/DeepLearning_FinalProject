# DeepLearning_FinalProject

## Project summary
This project is supposed to explore the concept of transfer learning with already pre-trained networks. In the first part, we will focus on creating a network using a pre-trained backbone provided by torchvision and applying it to the Oxford-IIIT Pet Dataset, where we will start with binary calssification of cats and dogs and later do classification of different cat and dog breeds.

As an extension of the basic project we will first implement some advanced data augmentation, which will be used in the last part, where we implement the FixMatch algorithm to be able to train our network with a limited amount labelled data.

## Data
The network will be trained on the Oxford-IIT pet dataset which can be downloaded [here](https://www.robots.ox.ac.uk/~vgg/data/pets/ "The Oxford-IIIT Pet Dataset"). 

The dataset contains 37 classes with roughly 200 images per class. The classes are different breeds of cats and dogs and the images differ in size, scale, pose and lighting. 

The labelling is done through the image names, i.e. the name of the image indicates its class and image names starting with an uppercase letter are cats, the rest are dogs. 