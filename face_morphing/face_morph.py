#this file has been modified from the original

from morphing.face_landmark_detection import makeCorrespondence
from morphing.delaunay import makeDelaunay
from morphing.faceMorph import makeMorphs
import subprocess
import argparse
import shutil
import os

def doMorphing(thePredictor,theImage1,theImage2,theDuration,theFrameRate,theResult, email):
	[size,img1,img2,list1,list2,list3]=makeCorrespondence(thePredictor,theImage1,theImage2)
	if(size[0]==0):
		print("Sorry, but I couldn't find a face in the image "+size[1])
		return
	list4=makeDelaunay(size[1],size[0],list3)
	makeMorphs(theDuration,theFrameRate,img1,img2,list1,list2,list4,size,theResult, email)

def morph(which, image1,image2, email):
	if(which=="face"):
		with open(image1,'rb') as img1, open(image2,'rb') as img2:
			doMorphing('./face_morphing/shape_predictor_68_face_landmarks.dat',img1,img2, 2, 25, 'morphing-example.mp4', email)
	elif(which=="body"):
		print("Body Detection Coming Soon!")
	else:
		print("Please enter correct detection type.")