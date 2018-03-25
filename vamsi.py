import cv2
import numpy as np
import math
import time
cap=cv2.VideoCapture(0)

while True:
    ret,frame=cap.read()
    frame=cv2.flip(frame,1)
    array=frame.shape[:2]
    #print array
    cv2.putText(frame,'Cover squares with your palm',(40,50),cv2.FONT_HERSHEY_COMPLEX,1,(255,170,0))
    cv2.rectangle(frame,(320,240),(335,255),(0,255,0),2)
    cv2.rectangle(frame,(360,160),(375,175),(0,255,0),2)
    cv2.rectangle(frame,(380,200),(395,215),(0,255,0),2)
    cv2.rectangle(frame,(420,230),(435,245),(0,255,0),2)
    cv2.rectangle(frame,(360,310),(375,325),(0,255,0),2)
    cv2.rectangle(frame,(380,260),(395,275),(0,255,0),2)
    cv2.rectangle(frame,(420,300),(435,315),(0,255,0),2)
    cv2.imshow('Our Live Sketcher',frame)
    if cv2.waitKey(1)==13:
        sample=frame
        break
cap.release()
cv2.destroyAllWindows()
u_corner=[(320,240),(360,160),(380,200),(420,230),(360,310),(380,260),(420,300)]
l_corner=[(335,255),(375,175),(395,215),(435,245),(375,325),(395,275),(435,315)]
cv2.imshow('Captured Sample',sample)
cv2.waitKey()
cv2.destroyAllWindows()
for i in range(0,7):
    print(u_corner[i],l_corner[i])
rectangle=sample[240:255,320:355]
cv2.imshow('sample',sample)
cv2.waitKey()
cv2.imshow('rectangle',rectangle)
cv2.waitKey()
hls_image=cv2.cvtColor(frame,cv2.COLOR_BGR2HLS)
#print(hls_image.shape)
#print hls_image[1:5]
cv2.imshow('HLS image',hls_image)
cv2.waitKey()
cv2.destroyAllWindows()
avgColor=np.zeros((7,3))

def getAvgColor(a,b,k):
    h=[]
    l=[]
    s=[]
    for i in range(a[1]+2,b[1]-2):
        for j in range(a[0]+2,b[0]-2):
            h.append(hls_image[i][j][0])
            l.append(hls_image[i][j][1])
            s.append(hls_image[i][j][2])
    avgColor[k][0]=np.median(h)
    avgColor[k][1]=np.median(l)
    avgColor[k][2]=np.median(s)
            
for i in range(0,7):
    getAvgColor(u_corner[i],l_corner[i],i)

c_lower=np.zeros((7,3))
c_upper=np.zeros((7,3))


for i in range(0,7):
    for j in range(0,3):
        if j==0:
            c_lower[i][0]=7
            c_upper[i][0]=12
        elif j==1:
            c_lower[i][1]=30
            c_upper[i][1]=40
        else:
            c_lower[i][2]=80
            c_upper[i][2]=80

for i in range(0,7):
    for j in range(0,3):
        if avgColor[i][j]-c_lower[i][j]<0:
            c_lower[i][j]=avgColor[i][j]
            
for i in range(0,7):
    for j in range(0,3):
        if avgColor[i][j]+c_upper[i][j]>255:
            c_upper[i][j]=255-avgColor[i][j]
            
lower_bound=[]
upper_bound=[]
for i in range(0,7):
    lower_bound.append([avgColor[i][0]-c_lower[i][0],avgColor[i][1]-c_lower[i][1],avgColor[i][2]-c_lower[i][2]])
    upper_bound.append([avgColor[i][0]+c_lower[i][0],avgColor[i][1]+c_lower[i][1],avgColor[i][2]+c_lower[i][2]])
    
print(lower_bound)
print(upper_bound)

def getDistance(s,e):
    return math.sqrt(((s[1]-e[1])**2)+((s[0]-e[0])**2))

def getAngle(s,f,e):
    d1=getDistance(s,f)
    d2=getDistance(f,e)
    dot=(s[1]-f[1])*(e[1]-f[1]) + (s[0]-f[0])*(e[0]-f[0])
    return math.acos(dot/(d1*d2)) * 57

def isHand(h,w,defects):
    if(defects>4):
        return False
    elif(h/w>4 or w/h>4):
        return False
    elif(h==0 or w==0):
        return False
    else:
        return True

start_time_2=round(time.time())

def checkForOneFinger(h,cntr,hull):
    tolerance=h/2
    idx=0
    m=480
    for i in range(0,len(cntr)):
        if(cntr[i][0][1]<m):
            m=cntr[i][0][1]
            idx=i
    n=0
#     global start_time_2
#     if(round(time.time())-start_time_2>5):
#         print "maximum point for contours is "+str(cntr[idx][0][0])+" "+str(cntr[idx][0][1])
#         print "Points in a hull"
#         for i in range(0,len(hull)):
#             print str(hull[i][0][0])+"  "+str(hull[i][0][1])
#             if(m+tolerance>hull[i][0][1]):#and cntr[idx][0][0]!=hull[i][0][0] and cntr[idx][0][1]!=hull[i][0][1]
#                 n=n+1
    extLeft = tuple(cntr[cntr[:, :, 0].argmin()][0])
    extRight = tuple(cntr[cntr[:, :, 0].argmax()][0])
    extTop = tuple(cntr[cntr[:, :, 1].argmin()][0])
    extBot = tuple(cntr[cntr[:, :, 1].argmax()][0])
    if(extTop[1]+tolerance>extRight[1] or extTop[1]+tolerance>extLeft[1]):
        n=n+1
#         start_time_2=round(time.time())
#         print "n is "+str(n)
#         print "extLeft ",extLeft
#         print "extRight ",extRight
#         print "extTop ",extTop
#         print "tolerance ",tolerance
       
    if n==0:
        return True
    else:
        return False
        
start_time=round(time.time())
        
cap=cv2.VideoCapture(0)

while 1:
    _, frame1 = cap.read()
#     cv2.imshow('frame',frame)
#     cv2.waitKey()
    frame=cv2.blur(frame1,(3,3))
    hls=cv2.cvtColor(cv2.flip(frame,1),cv2.COLOR_BGR2HLS)
    cv2.imshow('Hls',hls)
    l=[]
    mask = cv2.inRange(hls, np.array(lower_bound[0]), np.array(upper_bound[0]))
    l.append(mask)
    res = mask
    
    for i in range(1,7):
        mask=cv2.inRange(hls, np.array(lower_bound[i]), np.array(upper_bound[i]))
        l.append(mask)
        cv2.imshow('Mask '+str(i),mask)
        res=cv2.bitwise_or(res,mask)
        
    res=cv2.medianBlur(res,7)
    frame=cv2.flip(frame,1)
    #cv2.imshow('frame',frame)
    cv2.imshow('original',cv2.flip(frame1,1))
    cv2.imshow('mask',res)
    img,contours,hierarchy=cv2.findContours(res,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    sorted_contours=sorted(contours,key=cv2.contourArea,reverse=True)
    hull_image=frame.copy()
    approx_image=frame.copy()
    final_image=frame.copy()
    max_cntr=sorted_contours[0]    
    x,y,w,h = cv2.boundingRect(max_cntr)
    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)    
    cv2.imshow('Bounding Rectangle', frame)
    #cv2.imshow('frame',frame)
    hull1=cv2.convexHull(max_cntr,clockwise=False,returnPoints=False)
    hull2=cv2.convexHull(max_cntr,clockwise=False,returnPoints=True)
    cv2.drawContours(hull_image,[hull2],-1,(0,255,0),2)
    cv2.imshow('HULL',hull_image)
#     approx=cv2.approxPolyDP(hull2,5,True)
#     cv2.drawContours(approx_image,[approx],-1,(0,255,0),2)
    defects=cv2.convexityDefects(sorted_contours[0],hull1)
    new_defects=[]
    disTolerance=h/5
    angleTolerance=90
    for i in range(0,defects.shape[0]):
        s,e,f,d=defects[i,0]
        start=max_cntr[s][0]
        end=max_cntr[e][0]
        far=max_cntr[f][0]
        if(getDistance(start,far)>disTolerance and getDistance(end,far)>disTolerance and getAngle(start,far,end)<angleTolerance):
            if(start[1]<y+h-h/4 and end[1]<y+h-h/4):
                cv2.circle(approx_image, tuple(far), 3, [255,0,255], -1)
                cv2.line(approx_image,tuple(start),tuple(end),[0,255,255], 2)
                new_defects.append(defects[i])
    cv2.imshow('Convexity Defects',approx_image)
    
    #Detecting if it is a hand
    
    if(isHand(h,w,len(new_defects))==False):
        cv2.putText(final_image,'Hand is not Detected',(50,50),cv2.FONT_HERSHEY_COMPLEX,2,(255,100.100))
    else:
        convexity_defects=len(new_defects)
        if convexity_defects>0:
            number_of_fingers=convexity_defects+1
        else:
            if(checkForOneFinger(h,max_cntr,hull2)==True):
                number_of_fingers=1
            else:
                number_of_fingers=0
    end_time=round(time.time())
    if(end_time-start_time>5):
        print("number of fingers "+str(number_of_fingers))
        start_time=end_time
    cv2.imshow('Final Image',final_image)
    
    #end
    
    #remove endpoint of convexity defects if they are at the same fingertip
    
    for i in range(0,len(new_defects)):
        for j in range(i,len(new_defects)):
            disTolerance=w/6
            s,e,f,d=new_defects[i][0]
            s1,e1,f1,d1=new_defects[j][0]
            start=max_cntr[s][0]
            end=max_cntr[e][0]
            start1=max_cntr[s1][0]
            end1=max_cntr[e1][0]
            if(getDistance(start,end1)<disTolerance):
                max_cntr[s][0]=end1
                break
            if(getDistance(end,start1)<disTolerance):
                max_cntr[s1][0]=end
    #end
    
    if cv2.waitKey(1)==13:
        break
cap.release()
cv2.destroyAllWindows()