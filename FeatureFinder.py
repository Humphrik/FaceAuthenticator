import numpy as np
import cv2
from matplotlib import pyplot as plt


#Find face in sample image
face_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_eye.xml')

def main():
    number = 0
    validUsers = ["User0.jpg","User1.jpg","User2.jpg", "User3.jpg", "User4.jpg", "User5.jpg", "User6.jpg"]
    similarities = []
    camera = cv2.VideoCapture(0)
    orb = cv2.ORB_create()

    while(True):
        ret, cam = camera.read()
        gray = cv2.cvtColor(cam, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(cam, (x, y), (x+w, y+h), (255, 0, 0), 2)
        #cv2.imsave("CameraInput.jpg", cam)
        
        cv2.imshow("Feed", cam)
        if cv2.waitKey(1) & 0xFF == ord('p'):
            for s in validUsers:
                user = cv2.imread(s)
                user = cropFace(user)
                kp2, des2 = orb.detectAndCompute(user,None)
                bf = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck = True)
                cam = cv2.imread("CameraInput.jpg")
                cam = cropFace(cam)
                kp1, des1 = orb.detectAndCompute(cam,None)
                matches = bf.match(des1, des2)
                matches = sorted(matches, key = lambda x:x.distance)
                result = cv2.drawMatches(cam,kp1,user,kp2,matches, cam, flags = 2)
                print(len(matches))
                similarities.append(len(matches))
                plt.imshow(result), plt.show()
            print( userFound(similarities))
            similarities = 0
        elif cv2.waitKey(1) & 0xFF == ord('q'):
            break
        elif cv2.waitKey(1) & 0xFF == ord('t'):
            cv2.imwrite("User%d.jpg"%(number),cam)
            number+=1
            print "user confirmed!"
    cv2.destroyAllWindows()



def cropFace(image):
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(image, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        #roi_gray = gray[y:y+h, x:x+w]
        roi_color = image[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_color)
        #for(ex, ey, ew, eh) in eyes:
         #   cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        corners = [int(y+h*0.1),int(y+h*0.9),int(x+h*0.2),int(x+h*0.8)]
        image = image[corners[0]:corners[1], corners[2]:corners[3]]
        return image

def userFound(arr):
    for i in arr:
        if(i > 80):
             return "Welcome Back User!"
    return "User not found!"
            


main()
    

