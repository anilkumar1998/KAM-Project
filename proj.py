import cv2
import numpy
import math
import matplotlib.pyplot as plt
import time

cam4 = cv2.VideoCapture(0)
fr = 1
start_time = round(time.time(),)
while True:
    ret,frame = cam4.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    cv2.rectangle(frame, (30,30), (330,330), (255,0,0),2)
    roi = gray[30:330,30:330]
    gray_blur = cv2.GaussianBlur(roi,(7,7),0)
    _,thresh = cv2.threshold(gray_blur,100,255,cv2.THRESH_BINARY_INV)
    cv2.imshow("Threshold",thresh)
    cv2.imshow("Original",frame)
    img,cntr,hir = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    if len(cntr) == 0:
        img = frame.copy()
        cv2.putText(img,"No Object Found",(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
    else:
        max_cntr = max(cntr,key=cv2.contourArea)
        img = cv2.drawContours(frame,[max_cntr+(30,30)],-1,(255,0,0),2)
        hull = cv2.convexHull(max_cntr)
        img = cv2.drawContours(img,[hull+(30,30)],-1,(0,255,0),2)
        hull1 = cv2.convexHull(max_cntr,returnPoints=False)
        defects = cv2.convexityDefects(max_cntr,hull1)
        count_defects = 0
        if defects is not None:
            for i in range(defects.shape[0]):
                s,e,f,d = defects[i,0]

                start = tuple(max_cntr[s][0])
                end = tuple(max_cntr[e][0])
                far = tuple(max_cntr[f][0])
                start = tuple([x+30 for x in list(start)])
                end = tuple([x+30 for x in list(end)])
                far = tuple([x+30 for x in list(far)])

                # find length of all sides of triangle
                a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)

                # apply cosine rule here
                angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57

                # ignore angles > 90 and highlight rest with red dots
                if angle <= 90:
                    count_defects += 1
                    cv2.circle(img, far, 3, [255,0,255], -1)
                #dist = cv2.pointPolygonTest(cnt,far,True)

                # draw a line from start to end i.e. the convex points (finger tips)
                # (can skip this part)
                cv2.line(img,start, end, [0,255,0], 2)
            end_time = round(time.time())
            if (end_time - start_time) > 5:
                print("Number of fingers are: ",count_defects+1)
                start_time = end_time
    cv2.imshow("Contours+CH+CD",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cam4.release()
cv2.destroyAllWindows()
