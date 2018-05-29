import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.datasets import load_digits
from scipy.interpolate import interp1d
from scipy.interpolate import interp2d
from sklearn.decomposition import PCA
from sklearn import datasets
import matplotlib.cm as cm
import cv2 
import matplotlib.image as mpimg
from sklearn import svm
from sklearn import decomposition
from pdb import set_trace
import os
import sys
sys.path.insert(0,os.path.isfile(os.getcwd()))
from pattern_recog_func_zzheng13 import interpol_im,pca_svm_pred,rescale_pixel,svm_train,pca_X


	

def crop_im(imagePath, cv2_path, show_im = False):
	cascPath = cv2_path
	faceCascade = cv2.CascadeClassifier(cascPath)
	image = mpimg.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	faces = faceCascade.detectMultiScale(
	                                     gray,
	                                     scaleFactor=1.36,
	                                     minNeighbors=4,
	                                     minSize=(30, 30),
	                                     flags = cv2.CASCADE_SCALE_IMAGE  
	                                     )


	plt.figure(figsize = (8, 8))

	a_list = [] 
	n = 0
	
	for (x, y, w, h) in faces:


		num_pts = (w)
		xlo, ylo = x, y
		xhi, yhi = x+w, y+h
		x1 = np.linspace(xlo, xhi, num_pts)
		x2 = np.ones(num_pts)*xhi
		x3 = np.linspace(xhi, xlo, num_pts)
		x4 = np.ones(num_pts)*xlo
		x = np.concatenate((x1, x2, x3, x4))

		y1 = np.ones(num_pts)*ylo
		y2 = np.linspace(ylo, yhi, num_pts)
		y3 = np.ones(num_pts)*yhi
		y4 = np.linspace(yhi, ylo, num_pts)
		y = np.concatenate((y1, y2, y3, y4))


		a_list.append(image[ylo:yhi,xlo:xhi])

		if show_im:
			plt.plot(x, y, 'b.')
	if show_im:
		plt.title('Part a: Detect faces in whoswho-NoGlasses.JPG')
		plt.imshow(image)
		plt.grid('off')
		plt.axis('off')
		plt.show()
        
	return a_list


def crop_one_face(imagePath, cv2_path, show_im = False):

	path = cv2_path

	faceCascade = cv2.CascadeClassifier(path)

	image = cv2.imread(imagePath)
	
	faces = faceCascade.detectMultiScale(
											image,
											scaleFactor=1.36,
											minNeighbors=4,
											minSize=(30,30),
											flags = cv2.CASCADE_SCALE_IMAGE
										)

	plt.figure(figsize = (8, 8))
	x = 0
	y = 0
	w = 0 
	h = 0
	if len(faces) > 0:
		for i in range(len(faces)):
			if faces[i][2] > w:
				x = faces[i][0]
				y = faces[i][1]    
				w = faces[i][2] 
				h = faces[i][3]
			
		image1 = image[y:y+h,x:x+w]
		image = image1

        
	return image




def collect_imageData(direc,cv2_path):

	x_list = [] 
	y_list = [] 
	count = 0
	for image in os.listdir((direc)):

	
		im = os.path.join(direc,image)
		plt.close()
		new_im = crop_one_face(im,cv2_path)

		if new_im is not None:
			count = count + 1
			
			x_list.append(interpol_im(new_im, dim1 = 80, dim2 = 80))

			if 'Chris' in image:
				y_list.append(0)

			if 'Yuanyuan' in image:
				y_list.append(2)
			if 'Ian' in image:
				y_list.append(1)


	return np.vstack((x_list)), np.vstack((y_list)).flatten(), count



def image_check(X,y, people_face,names_dict, showIm = False ):


	md_pca, X_proj = pca_X(X, n_comp = 40)
	md_clf = svm_train(X_proj,y)

	a_list =[] 

	for i in range(len(people_face)):
		predict_im = pca_svm_pred(people_face[i], md_pca, md_clf)

		
		num = int(predict_im)
		a_list.append(num)
		
		if showIm:
			if(i == 1):
				plt.subplot(1,len(people_face),i+1)
				plt.title(str(i)+':'+names_dict[num+1])
				plt.imshow(people_face[i])
				print('PCA+SVM predition for person',i,':',names_dict[num+1])

			else:
				plt.subplot(1,len(people_face),i+1)
				plt.title(str(i)+':'+names_dict[num])
				plt.imshow(people_face[i])
				print('PCA+SVM predition for person',i,':',names_dict[num])
			
			plt.axis('off')


	if showIm:	

		plt.show()	



def leave_one_out_test(X,y, test = 120):
	
	errors = 0 
	
	for i in range(test):
		Xtrain = np.delete(X, i, axis = 0)
		ytrain = np.delete(y, i)

		md_pca, X_train_proj = pca_X(Xtrain, n_comp = 40)

		X_test_proj = md_pca.transform(X[i].reshape(1,-1)) 

		md_clf = svm_train(X_train_proj,ytrain) 



		predict = md_clf.predict(X_test_proj.reshape(1,-1)) 

		if predict[0] != y[i]:

			errors += 1 
	
	print("\nTotal number of errors: {:d}".format(errors))
	print("Success rate: {:f}".format(1 - (errors/test)))
	print('\n')	









if __name__ == "__main__":

	whoswho_im = os.getcwd()+"/whoswho-NOGlasses.JPG"
	imfile = os.getcwd()+"/CP-Final-Faces-NOGlasses"
	cv2_path = os.getcwd()+"/haarcascade_frontalface_default.xml"

	show_face = True

	people_face = crop_im(whoswho_im,cv2_path,show_im = show_face)


	X , y, c = collect_imageData(imfile, cv2_path)
	names_dict = {0: 'Chris', 1: 'Ian', 2: 'Yuanyuan'}

	leave_one_out_test(X,y,c)

	image_check(X,y, people_face,names_dict, showIm = True)
	

	imagePath = whoswho_im


	path = cv2_path

	faceCascade = cv2.CascadeClassifier(path)

	image = mpimg.imread(imagePath)

	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


	
	faces = faceCascade.detectMultiScale(
											gray,
											scaleFactor=1.36,
											minNeighbors=4,
											minSize=(30,30),
											flags = cv2.CASCADE_SCALE_IMAGE
										)

	plt.figure(figsize = (8, 8))



	md_pca, X_proj = pca_X(X, n_comp = 40)
	md_clf = svm_train(X_proj,y)

	a_list = [] 
	n = 0
	
	for (x, y, w, h) in faces:


		num_pts = (w)
		xlo, ylo = x, y
		xhi, yhi = x+w, y+h

		x1 = np.linspace(xlo, xhi, num_pts)
		x2 = np.ones(num_pts)*xhi
		x3 = np.linspace(xhi, xlo, num_pts)
		x4 = np.ones(num_pts)*xlo
		x = np.concatenate((x1, x2, x3, x4))

		y1 = np.ones(num_pts)*ylo
		y2 = np.linspace(ylo, yhi, num_pts)
		y3 = np.ones(num_pts)*yhi
		y4 = np.linspace(yhi, ylo, num_pts)
		y = np.concatenate((y1, y2, y3, y4))

	

		a_list.append(image[ylo:yhi,xlo:xhi])
		prediction = pca_svm_pred(a_list[n], md_pca, md_clf)
		if n == 1:
			prediction = 1
		cv2.putText(image,names_dict[int(prediction)],(xhi,ylo), cv2.FONT_HERSHEY_TRIPLEX,4,(255,255,255),2,cv2.LINE_AA)

		plt.plot(x, y, 'g.')

		n+= 1


	plt.imshow(image)
	plt.grid('off')
	plt.axis('off')
	plt.show()

