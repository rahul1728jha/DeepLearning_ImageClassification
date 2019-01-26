from PIL import Image
import numpy as np
import glob
from IPython.display import display
from sklearn.model_selection import train_test_split

import os,sys

import keras
from keras.utils import to_categorical
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential,Model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras import backend as K

class ImageObjects():
	def __init__(self):
		self.baseDir = 'data/modified_flowers/'
		self.imageDir = ['daisy','rose','sunflower']
		self.convertedWidth = self.convertedHeight  = 128 
		self.batch_size = 32
		self.epochs=30

		self.images = []
		self.labels = []
		self.classes = 0

def resize_images(obj):
	for obj.subDir in obj.imageDir:
		print('\nBrowsing directory: ' , obj.subDir)
		maxWidth = 0
		maxHeight = 0
		for pic in glob.glob(obj.baseDir + obj.subDir + '/*.*'):
			imageOriginal = Image.open(pic)
			
			width = np.array(imageOriginal).shape[0]
			height = np.array(imageOriginal).shape[1]
			
			if height > maxHeight:
				maxHeight = height
			if width > maxWidth:
				maxWidth = width
			
			imageResized = imageOriginal.resize([obj.convertedWidth, obj.convertedWidth])
			obj.images.append( np.asarray(np.array(imageResized)))
			obj.labels.append(obj.classes)
			
		obj.classes += 1
		print('Max height : ' , maxHeight)
		print('Max width : ' , maxWidth)
	obj.classes -= 1

# Changing the data from list to array.
def getX_Y(obj):
    X = np.asarray(obj.images)
    Y = np.asarray(obj.labels)
    return X,Y

# Test Train Split Function
def returnTestTrainSplitData(X, Y):
	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, shuffle= True)
	y_train_categorized = to_categorical(y_train)
	y_test_categorized = to_categorical(y_test)
	print(' X_train shape %s . Y_train shape %s ' %(X_train.shape,y_train.shape))
	print(' X_test shape %s . Y_test shape %s ' %(X_test.shape,y_test.shape))
	return X_train, X_test,y_train_categorized,y_test_categorized


def executeModel(obj,model,X_train, X_test,y_train_categorized,y_test_categorized,epochs=20):
	print('Running for epochs : ',epochs)
	datagen = augmentation(X_train)
	model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

	history  = model.fit_generator(
		datagen.flow(X_train, y_train_categorized, batch_size=obj.batch_size),
		steps_per_epoch=int(len(X_train) / obj.batch_size),
		epochs=epochs,
		validation_steps= int(len(X_test) / obj.batch_size ),
		validation_data=datagen.flow(X_test,y_test_categorized)
	)

	score = model.evaluate_generator ( datagen.flow(X_test,y_test_categorized),steps=int(len(X_test) / obj.batch_size ))

	print('Saving Model')
	model_json = model.to_json()
	with open("model/model.json", "w") as json_file:
		json_file.write(model_json)
	# serialize weights to HDF5
	model.save_weights("model/model.h5")

	print("-" * 100)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])
	return history

def augmentation(X_train):
	datagen = ImageDataGenerator(
		featurewise_center=True,
		featurewise_std_normalization=True,
		rotation_range=20,
		fill_mode='nearest',
		horizontal_flip=True,
		brightness_range=[0.5, 1.5],
		vertical_flip=True)

	datagen.fit(X_train)
	return datagen

def getModel():

	model = applications.VGG19(weights = "imagenet", include_top=False, input_shape = (128,128,3))

	for layer in model.layers:
		layer.trainable = False

	x = model.output

	x = Flatten()(x)
	x = Dense(256, activation="relu")(x)
	x = Dropout(0.5)(x)
	x = Dense(128, activation="relu")(x)
	x = Dropout(0.5)(x)
	x = Dense(64, activation="relu")(x)
	x = Dropout(0.5)(x)

	predictions = Dense(3, activation="softmax")(x)

	combinedModel = Model(input = model.input, output = predictions)

	#combinedModel.summary()
	return combinedModel
	
	
if __name__ == '__main__':
	epochs = 0
	if len(sys.argv) == 2:
		epochs = int(sys.argv[1])
	else:
		epochs = 20
	obj = ImageObjects()
	print('\n#############################Resizing images#############################')
	resize_images(obj)
	print('\n#############################Train Test Split#############################')
	X,Y = getX_Y(obj)
	X_train, X_test,y_train_categorized,y_test_categorized = returnTestTrainSplitData(X, Y)
	print('\n#############################Get the model#############################')
	model = getModel()
	print('\n#############################Execute the model#############################')
	history = executeModel(obj,model,X_train, X_test,y_train_categorized,y_test_categorized,epochs)
