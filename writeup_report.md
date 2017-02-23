# Behavioral Cloning

---

## Choosing a Data Source 

Initially, I tried to use my own data by using the simulator to drive around the track with the keyboard. However, the data this produced was very jerky and did not work too well in training. I also tried to use the beta simulator with the mouse, but I only have a trackpad, so that didn't work too well either. I ended up using just the Udacity provided data, and making augmentations to that data to produce a good dataset

---

## Exploratory Data Analysis

First, the data was analyzed and plotted out to see what the distribution of steering angles was. Viewing the distribution of the steering angles in a histogram, it was clear that there were too many straight samples, and not enough turning samples. This had to be compensated for through appropriate sampling and augmentation methods in the training to prevent the model from overfitting to the straight steering data. 

---

## Data Distribution

Because the original data source overrepresented steering angles of 0, the data had to be resampled to produce a good model. Four different data distributions were used in training: Uniform, Uniform with upsampled straight (2x the number for the turn bins), Normal (Guassian), and Fat Tail. 

The one that ended up working best was the Uniform with upsampled straights. 

Uniform distributions worked decently, but there was too much oscillation in the car motion. Models with a Normal and Fat Tail distribution had difficulty making sharp turns and would often understeer. Upsampling the straight (zero) steering angles dampened the oscillation of the car on straightaways while retaining the ability of the model to make sharp turns when necessary. 

---

## Data Augmentation 

Four different data augmentations were used to enhance the dataset and provide better performance. 

* Mirorring: Mirroring the images and flipping the steering angle (e.g. -0.5 -> 0.5)
* Brightness: Scaling the brightness of the image
* Angle perturbations: Changing the steering angle without modifying the image, to account for collection error
* Left/Right cameras: Using the left and right camera images with a steering angle of +/- .25 from the true angle

All the augmented data was produced using a Python generator within the Keras model, so unique data was constantly being generated on the fly for each batch.

The split between center/right/left camera data was 50/25/25. The split between real and augmented images was 50/50, each augmented image had independent 50% chances of being each mirorred, brightened, and perturbed. 

---

## Model

2 models were used in training: AlexNet and the model from Nvidia paper. AlexNet did not produce very good results, so it was discarded quite early.

The Nvidia model is a sequential model with the following architecture (in order):

* A Normalization layer to normalize values to [-1,1]
* A 5 x 5 x 25 Convolution with strides of 2
* A 5 x 5 x 36 Convolution with strides of 2
* A 5 x 5 x 48 Convolution with strides of 2
* A 3 x 3 x 64 Convolution with strides of 1
* A 3 x 3 x 64 Convolution with strides of 1
* A dropout layer with dropout probability of 0.25

* A fully-connected 576 x 1164 Dense layer
* A dropout layer with dropout probability of 0.25

* A fully-connected 1164 x 100 Dense layer
* A dropout layer with dropout probability of 0.25

* A fully-connected 100 x 50 Dense layer
* A dropout layer with dropout probability of 0.25

* A fully-connected 50 x 10 Dense layer
* A dropout layer with dropout probability of 0.25

* A fully-connected 10 x 1 Dense layer

All Convolution and Dense layers were followed by a ReLu activation and initialized with a he_normal distribution of weights.

Training was conducted with batch sizes of 128 that were provided by a Python generator and models were trained for anywhere from 3-15 epochs with 32000 samples/per epoch. The best model used 8 epochs for training. An Adam optimizer was used, so learning rates did not have to be set manually. 
