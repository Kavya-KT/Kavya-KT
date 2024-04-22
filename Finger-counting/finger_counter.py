import cv2
import mediapipe

medhand=mediapipe.solutions.hands
draw=mediapipe.solutions.drawing_utils


hand=medhand.Hands(max_num_hands=1)

video=cv2.VideoCapture(0)
while True:
    suc,img=video.read()
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    res=hand.process(img)
    img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    tipids=[4,8,12,16,20]
    lmlst=[]
    #cv2.rectangle(img,(30,300),(100,450),(255,255,0),cv2.FILLED)
    cv2.rectangle(img,(30,320),(100,430),(255,255,0),2)
    if res.multi_hand_landmarks:
        for handlm in res.multi_hand_landmarks:
            for id,lm in enumerate(handlm.landmark):
                #print(id,lm)
                cx=lm.x
                cy=lm.y
                lmlst.append([id,cx,cy])
                #print(lmlst)
                if len(lmlst)!=0 and len(lmlst)==21:
                    fingerlst=[]

                    #thumb
                    if lmlst[0][1]<lmlst[1][1]:
                        if lmlst[4][1]<lmlst[3][1]:
                            fingerlst.append(0)
                        else:
                            fingerlst.append(1)
                    else:
                        if lmlst[4][1]>lmlst[3][1]:
                            fingerlst.append(0)
                        else:
                            fingerlst.append(1)


                    #other fingers
                    for i in range(1,5):
                        if lmlst[tipids[i]][2]>lmlst[tipids[i]-2][2]:
                            fingerlst.append(0)
                        else:
                            fingerlst.append(1)
                    #print(fingerlst)
                    if len(fingerlst)!=0:
                        fingercount=fingerlst.count(1)
                        print(fingercount)
            cv2.putText(img,str(fingercount),(50,400),cv2.FONT_HERSHEY_COMPLEX,2,(0,250,0),2)
            draw.draw_landmarks(img,handlm,medhand.HAND_CONNECTIONS,draw.DrawingSpec(color=(255,255,0),thickness=1,circle_radius=5),draw.DrawingSpec(color=(0,255,255),thickness=5))
    

    cv2.imshow('hand',img)
    if cv2.waitKey(1)&0XFF==ord('p'):
        break
video.release()
cv2.destroyAllWindows()