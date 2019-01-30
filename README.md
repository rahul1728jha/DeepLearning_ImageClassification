# DeepLearning_ImageClassification
Flower classification using CNN and data augmentation

# Objective
There are three types of flowers in the dataset: Daisy,Rose,Sunflower. The task is to build a CNN model for classification.

# Challenges
The dataset size is very small hence the model tends to overfit a lot.

# Approach
1. Simple CNN : Overfits a lot due to scarcity of data
2. Simple CNN with data augmentation : Performs well but the losses are a bit erratic
3. Transfer learning with data augmentation : Best model. Performs well with smooth loss curve

# Performance
Softmax cross entropy is used. Accuracy is 85%. Loss is 0.40

# Project structure:
Folders: 
  1. data : Contains the dataset with three sub folders
  2. dataAugmented : Contains the augmented images. Not used for training purpose. Just to visualize the augmented images to be used.
  3. model : The trained model is saved into this folder. 
  4. templates : GUI related files : index.html, style.css, jqueryScript.js, default.png
  5. testImages : Used to store the images along with there predicted class
  
Files:
  1. FlowerClassification.ipynb : Ipython notebook to demonstrate various approaches
  2. train.py : The best model obtained is used to train and persist with this file
  3. test.py : Flask service which is triggered by the GUI to perform the predictions
  
# Usage:

  Create folder data/rose,data/sunflower,data/daisy. Put contents of rose1 and rose2 into rose and sunflower1 and sunflower2 into sunflower. 
  
  Training: python train.py 30
  
  30 : epochs
  
  Test: python test.py
  
  Flask service will start on http://127.0.0.1:5000. This is used in the jqueryScript.js to perform the prediction
  
  Open the index.html in templates folder and trigger the service
  
  
# Reference links:
  Bootstrap:
  
    https://bootsnipp.com/snippets/76778
    
    https://www.w3schools.com/bootstrap/bootstrap_templates.asp
    
  Flask:
  
    https://www.tutorialspoint.com/flask
    
# Refer Image Classification Usage.pdf for complete usage guide with screen schots
