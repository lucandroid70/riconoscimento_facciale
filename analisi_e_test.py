#Creato da Luca Sabato sistema di riconosmento facciale con CV2 luca.droi@outlook.it ;)  file analisi_e_test.py


import cv2
import numpy as np  
import os

#KNN algo
def distance(v1,v2):
	return np.sqrt(((v1-v2)**2).sum())
def knn(train,test,k=5):
	dist=[]
	for i in range(train.shape[0]):
		ix=train[i,:-1]
		iy=train[i,-1]

		d=distance(test,ix)
		dist.append([d,iy])
	#sort on the distance ans get top k
	dk=sorted(dist,key=lambda x:x[0])[:k]
	#retrieve only the labels
	labels=np.array(dk)[:,-1]
	#get frequencies of labels
	output=np.unique(labels,return_counts=True)
	#find max frequency and label
	index=np.argmax(output[1])
	return output[0][index]


#Initialize camera
cap=cv2.VideoCapture(0)
#face detection
#face_cascade=cv2.CascadeClassifier("haarCascade_frontalface_alt.xml") #### ERRORE
face_cascade = cv2.CascadeClassifier(os.path.join(cv2.haarcascades, 'haarcascade_frontalface_default.xml'))
skip=0
dataset_path='./data/'

face_data=[]
labels=[]
class_id=0 #labels for the given file
names={} #mapping between id-name

#Data preparation
for fx in os.listdir(dataset_path):
	if fx.endswith('.npy'):
		data_item=np.load(dataset_path+fx)
		face_data.append(data_item)
		names[class_id]=fx[:-4]
		#create labels for the class
		target=class_id*np.ones((data_item.shape[0],))
		class_id+=1
		labels.append(target)

face_dataset=np.concatenate(face_data,axis=0)
face_labels=np.concatenate(labels,axis=0).reshape((-1,1))	
trainset=np.concatenate((face_dataset,face_labels),axis=1)


#testing 

while True:
	ret,frame=cap.read()
	if ret==False:
		continue
	faces=face_cascade.detectMultiScale(frame,1.3,5)
    	
	for face in faces:
		x,y,w,h=face
		#get the face ROI
		offset=10
		face_selection=frame[y-offset:y+offset+h,x-offset:x+w+offset]
		face_selection=cv2.resize(face_selection,(100,100))

		#predicted label(out)
		out=knn(trainset,face_selection.flatten())

		#display rectangle & name on the screen

		pred_name=names[int(out)] 
		cv2.putText(frame,pred_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)

	cv2.imshow("Faces",frame)
	
	key=cv2.waitKey(1) & 0xFF
	if key==ord('q'):
		break

cap.release()
cv2.destroyAllWindows()		







	

		
