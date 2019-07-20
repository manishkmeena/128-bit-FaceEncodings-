import pandas as pd
import face_recognition as fc
import cv2

a=fc.load_image_file('/home/manish/Desktop/ML/Projects/Face vs Encodings/IMG_20180821_020242_394_1.jpg')
fl=fc.face_locations(a)
enc=fc.face_encodings(a,fl)

df=pd.read_csv('Enc1.csv')
print(df)

for i in fl:
    [x1,y1,x2,y2]=i
    cv2.rectangle(a,(y1,x2),(y2,x1),(0,0,255),5)
#cv2.imshow('a',a)
#k=cv2.waitKey()
#cv2.destroyAllWindows()


var=0
nameList=[]
encList=[]

for i in range(0,len(fl)):
    faceName=input('Enter the Name of Person(In Sequence Left to Righ)-')
    nameList.append(faceName)
    encList.append(enc[var])
    print(var)
    print(nameList)
    print(encList)
    var=var+1

df1=pd.DataFrame({'Names':nameList,'Enc':encList})
df1.to_csv('Enc1.csv')









