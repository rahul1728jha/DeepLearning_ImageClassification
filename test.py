from flask import Flask,request,jsonify
from flask_cors import CORS
from PIL import Image
import re
import numpy as np
import base64
from scipy.misc import imsave, imread, imresize
import os,sys
import keras.models
from keras.models import model_from_json
sys.path.append(os.path.abspath("./model"))
from load import * 

app = Flask(__name__)
CORS(app)

global model,graph,mapping,count
model,graph,mapping,count = init()

#decoding an image from base64 into raw representation
def convertImage(b64_string):
	imgData1=b64_string.decode('utf-8')
	imgstr = re.search(r'base64,(.*)',imgData1).group(1)
	with open('testImages/output_'+str(count)+'.jpg','wb') as output:
		output.write(base64.decodebytes(imgstr.encode()))
		
@app.route('/<name>')
def hello_world(name):
	msg = 'HI Hello World. File IS :  ' + str(name)
	
	return jsonify(Message=msg)

@app.route('/predict',methods=['POST'])
def predict():
	global count
	imgData = request.get_data()
	convertImage(imgData)
	
	imageOriginal = Image.open('testImages/output_'+str(count)+'.jpg')
	images=[]
	imageResized = imageOriginal.resize([128, 128])
	images.append( np.asarray(np.array(imageResized)))
	
	X = np.asarray(images)
	with graph.as_default():
		
		#perform the prediction
		out = model.predict(X)
		out = out[0]
		flowerIndex = np.argmax(out)
		flowerType=mapping.get(flowerIndex)
		probabilities = [str(mapping.get(i)) +' : '+ str(out[i])+'||' for i in range(0,len(out))]
		
		#Rename the file: Append the predicted value for reference
		if os.path.isfile('testImages/output_'+str(count)+'.jpg'):
			os.rename('testImages/output_'+str(count)+'.jpg', 'testImages/output_'+str(count)+'_predicted_'+flowerType+'.jpg')
			count +=1
			
		return jsonify(FlowerType=flowerType,Probabilities=probabilities)


if __name__ == '__main__':
	app.run()