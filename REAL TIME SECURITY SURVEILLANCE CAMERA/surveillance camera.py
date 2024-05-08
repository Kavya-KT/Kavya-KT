import cv2
from datetime import datetime
import numpy as np
import pandas as pd
import os
import re
import easyocr
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'



# initialize the HOG descriptor
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')


detected_plate=[]
detected_plates_data=[]
serial_number=1
processed_plates=set()

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = None

video=cv2.VideoCapture('demo.mp4')

while True:
    success, img = video.read()#Reading each frame
    
    if not success:
        break

    current_time = datetime.now().strftime('%Y-%m-%d %I:%M:%S %p')
    cv2.putText(img, current_time, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)

    img = cv2.resize(img, (640,480))
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#face detection
    faces = face_cascade.detectMultiScale(gray, minNeighbors=7)
    print('Number of detected faces:', len(faces))
    for (x2,y2,w2,h2) in faces:
        cv2.rectangle(img,(x2,y2),(x2+w2,y2+h2),(0,0,255),4)

# detect humans in input image
    (humans, _) = hog.detectMultiScale(gray, winStride=(8,8),padding=(50,50), scale=1.1)
    print('Human Detected : ', len(humans))
    for (x1, y1, w1, h1) in humans:
        pad_w, pad_h = int(0.15 * w1), int(0.01 * h1)
        cv2.rectangle(img, (x1 + pad_w, y1 + pad_h), (x1+ w1 - pad_w, y1 + h1 - pad_h), (0, 255, 0), 2)

#number plate

    plates = cascade.detectMultiScale(gray, 1.2, 5)
    print('Number of detected license plates:', len(plates))
    for (x,y,w,h) in plates:
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
        gray_plates = gray[y:y+h, x:x+w]
        color_plates = img[y:y+h, x:x+w]

        cv2.imshow('Number Plate', gray_plates)
        # cv2.imshow('Number Plate Image', img)

        reader = easyocr.Reader(['en'])
        result = reader.readtext(gray_plates)
        if result:
            plate_texts = [detection[1] for detection in result]
            combined_text = ''.join(plate_texts)
            clean_text = re.sub(r'[^a-zA-Z0-9]', '', combined_text)
            detected_plate.append(clean_text)
        print(detected_plate)
    

    data = {
        'VehicleNumber': ['R183JF', 'N894Jv', 'L656XH','H644LX','K884RS','66HH07','L605HZ','R197GB','21BH2345AA','21BH0001AA','BP199SN','B2228HM','RJ14CV0002','22BH6517A'],
        'OwnerName': ['John Doe', 'Jane Smith', 'Michael Johnson', 'Emily Brown', 'David Wilson', 'Sarah Miller','Emily ','David Brown', 'Chris Davis', 'Amy Taylor','kavya','jishnu','nandana','ammu'],
        'OwnerContact': ['123-456-7890', '987-654-3210', '555-555-5555', '111-222-3333', '444-444-4444','98975987952','789-456-123', '777-777-7777', '888-888-8888', '999-999-9999','123-456-89','9856-789-645','159-852-789','158-745-625']
    }
    df_vehicle_department = pd.DataFrame(data)
    # print(df_vehicle_department)

    def get_owner_details(vehicle_number, df):
            # Search for the vehicle number in the DataFrame
            matching_row = df[df['VehicleNumber'] == vehicle_number]
            
            if not matching_row.empty:
                # Retrieve owner details from the matching row
                owner_name = matching_row['OwnerName'].values[0]
                owner_contact = matching_row['OwnerContact'].values[0]
                return owner_name, owner_contact
            else:
                
                return None,None
    for plate in detected_plate:
            if plate not in processed_plates:  # Check if the plate has already been processed
                owner_name, owner_contact = get_owner_details(plate, df_vehicle_department)
                #If owner details are found, printing the plate number, owner name, owner contact, and detection time.
                if owner_name:
                    if owner_name is not None:
                        print("Plate:", plate)
                        print("Owner Name:", owner_name)
                        print("Owner Contact:", owner_contact)    
                        print("Detection Time:",current_time)
                        print()  
                    
                    # Add the detected plate, owner details, and detection time to the list
                    
                        detected_plates_data.append({
                            "sl.no:":serial_number,
                            'Detection Time': current_time,
                            'Plate': plate, 'Owner Name': owner_name, 
                            'Owner Contact': owner_contact})
                    
                        serial_number+=1
                processed_plates.add(plate)
    
  
    cv2.imshow('face',img)
   
    if out is None:
            out = cv2.VideoWriter('output3.avi', fourcc, 10.0, (img.shape[1], img.shape[0]))
    out.write(img)
    
    if cv2.waitKey(1) & 0XFF== ord('p'):
        break


df_detected_plates = pd.DataFrame(detected_plates_data)
df_detected_plates.to_excel('detected_plates31.xlsx', index=False)


video.release()
if out is not None:
    out.release()
cv2.destroyAllWindows()
