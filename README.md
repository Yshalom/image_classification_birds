# Train A Neural Network On Bird Database For Image Classification

## Introduction
I took image database from HuggingFace (https://huggingface.co/datasets/yashikota/birds-525-species-image-classification),
and I made a task to create model that classify the images on the database.
The database as follow:
 - Train-images: 84635
 - Test-images: 2625
 - Validation-images: 2625
 - Image-Size: 224x224
 - Classes: 525 (+1 for images without birds)

*Examples from the database:*  
<img src="README_files/database-examples/train-99.jpg"/>
<img src="README_files/database-examples/train-186.jpg"/>
<img src="README_files/database-examples/train-427.jpg"/>

## What's in this project?
1. This project contains the workflow I took, to get the best model I could, it details my tries, failures and successes.
At the end the best model is represented to show how it perform.

2. For each model there is a log file along with the model, representing how it perform, and the training epochs' scores.
Also there is a script that takes the data and build a SVG image from it for easier representation.

3. There is a database viewer that can be configured to show the training/test/validation images with/without image transforms.

4. Testing scripts; *note - not all the code is tested, this project is more focused on AI than coding*

> [!NOTE]
> Not every model architecture and every training is presented here, only the core points of the research are kept!


## Before training
* I made database-viewer application to see the database.
* I made a class that hold the database, and convert it to NumPy format (you can look on the database format at its source `https://huggingface.co/datasets/yashikota/birds-525-species-image-classification`, here the database is in .gitignore).
* I applied a cache for the images that holds them pre-convert from image representation to PyTorch Tensors, ready for easy access during training.

</br>

# Model Training

## CNN-1
The model is a simple CNN architecture, small with `114K` parameters.  
Input size: 94x94
After several tries the best results I could get with this model were:
- Loss:
    - train-loss: 2.098122
    - test-loss:  1.869292
    - val-loss:   2.13719
- Accuracy:
    - train-accuracy: 53.301830%
    - test-accuracy:  57.485710%
    - val-accuracy:   53.676189%

<img width="400px" src="models/CNN-1/try-2/log/graph.svg">

> [!WARNING]
> The model is too small for learning complex pattern!

## CNN-2
The model is a simple CNN architecture, small with `137K` parameters.  
Input size: 224x224
The best results:
- Loss:
    - train-loss: 1.790865
    - test-loss:  1.669988
    - val-loss:   1.951408
- Accuracy:
    - train-accuracy: 60.057896%
    - test-accuracy:  62.019043%
    - val-accuracy:   57.333332%

<img width="400px" src="models/CNN-2/log/graph.svg">

> [!WARNING]
> The model better from the previous one, but still it's too small for learning complex pattern!

This models has almost the same parameter count as the previous one, though it out perform it,
we can take into conclusion the bigger image size, that probably responsible for that.

> [!NOTE]
> The full image size improve performance.

## AlexNet-like-1
The model is a AlexNet architecture, with `12M` parameters.  
There is a change from AlexNet architecture at the last Linear Layers' size: 4096 -> 1024  
Input size: 224x224
The best results:
- Loss:
    - train-loss: 0.473686
    - test-loss:  0.98839
    - val-loss:   1.289062
- Accuracy:
    - train-accuracy: 89.238495%
    - test-accuracy:  75.580956%
    - val_accuracy:   70.247620%

<img width="400px" src="models/AlexNet-like-1/try-3/log/graph.svg">

> [!WARNING]
> The results are from epoch training 30, more than that the model start to over-fit the training data

## AlexNet-like-2
The model is a AlexNet architecture, with `10M` parameters.  
There is a change from AlexNet architecture at the last Linear Layers' size: 4096 -> 1024,
and a reduction in channel count at the convolution layers.  
Input size: 224x224
The best results I could get with this model were:
- Loss:
    - train-loss: 0.094126
    - test-loss:  0.823275
    - val-loss:   1.129378
- Accuracy:
    - train-accuracy: 98.961426%
    - test-accuracy:  79.695236%
    - val-accuracy:   75.733337%

<img width="400px" src="models/AlexNet-like-2/log/graph.svg">

> [!WARNING]
> More room for training, but still there is some over-fitting issue

## AlexNet-like-3
The model is a AlexNet architecture, with `3M` parameters.  
There is a change from AlexNet architecture at the last Linear Layers' size: 4096 -> 512,
and a strong reduction in channel count at the convolution layers.  
Input size: 224x224
The best results I could get with this model were:
- Loss:
    - train-loss: 0.124912
    - test-loss:  1.20639
    - val-loss:   1.682443
- Accuracy:
    - train-accuracy: 98.092995%
    - test-accuracy:  72.114281%
    - val_accuracy:   68.380951%

<img width="400px" src="models/AlexNet-like-3/try-2/log/graph.svg">

> [!WARNING]
> The smaller model straggle at the test & evaluation images too, in fact as I make the model smaller it straggle more.
> The solution to the over-fitting must be something else, smaller AlexNet-like model will not solve the issue!

---

Here I though that maybe AlexNet architecture is too old. I try other architecture which try to mix AlexNet with VGGNet ideas (use multiple convolution of 3 instead of 5,7,11).  
The (**CNN-3**, `7M` parameters) apply this idea, I tried different sizes of the network, at the end the best I could reach is not better than the AlexNet architecture and I aborted this idea.  
I didn't try the **VGG** architecture as it is 138M-144M parameters in size, which of course will cause over fitting.

I also tried another network (**AlexNet-like-4**, `7M` parameters), which also doesn't give me the results I'm looking for.  

---

# Image Transforms - Adding Data Variance

I tried to train multiple models, with different sizes, learning-rate, batch-size and epochs.
I got either small-model which don't recognize complex patterns (perform bad on the images), or large models which are over fitted to the training set (and perform bad on the test & evaluation sets).  
I decided to expand the training data variance with `torchvision.transforms.v2`, and than train a model.

Each image is randomly changed by:
- crop of the image (at least 80% of the images is saved).
- 0-2 box erased from the image (a box is 1%-5% of the image).
- flip
- color jitter

*Examples from the database:*  
<img src="README_files/image-transforms/train-99.jpg"/>
<img src="README_files/image-transforms/train-186.jpg"/>
<img src="README_files/image-transforms/train-427.jpg"/>


## Train old networks again
First I start training the old networks again, seeing how the would they perform under the new changes
### AlexNet-like-3
The best results I could get with this model were:
- Loss:
    - train-loss: 0.916084
    - test-loss:  0.816708
    - val-loss:   1.064557
- Accuracy:
    - train-accuracy: 78.806648%
    - test-accuracy:  79.847618%
    - val-accuracy:   76.761902%

> [!NOTE]
> Look the over fitting problem is gone - the test & validation scores is close to the train score.

> [!WARNING]
> The mode can't detect complex patterns - it is too small!

---

I also tried the AlexNet-like-4 again, with no success.  
And also try the AlexNet-like-1 again, with no success either *(files are not in the repo)*

---

## AlexNet-like-5
**I saw that the previous models are too small from getting the full complexity of the new training set they get, therefore I tried to make a larger model, to see how will it perform.**
I made a bigger model with `17M` parameters.  
There is a change from AlexNet architecture at the last Linear Layers' size: 4096 -> 1536  
The results:
- Loss:
    - train-loss: 0.070933
    - test-loss:  0.439326
    - val-loss:   0.791034
- Accuracy:
    - train-accuracy: 98.908257%
    - test-accuracy:  88.761902%
    - val-accuracy:   85.028572%

<img width="400px" src="models/AlexNet-like-5/try-1/log/graph.svg">

---

As I saw that more parameters means better learning with *AlexNet-like-5*, I tried to go even farther with *AlexNet-like-6* which has `22M` parameters. Though I couldn't see any improvement there.

---

**AlexNet-like-5 gave me good enough results, I'm happy with them, and I stop here**