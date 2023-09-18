import face_recognition
import cv2
import glob
print("Start face recognition.")
#print(face_locations)
#print(face_locations[0][3])
cnt = 0
j = -1
for filename in glob.glob('./textbook/*.png'):
    j = j + 1
    image = face_recognition.load_image_file(filename)
    face_locations = face_recognition.face_locations(image)
    image = cv2.imread(filename)
    #image = cv2.resize(image, dsize=(0,0), fx=3, fy=3, interpolation = cv2.INTER_LINEAR)
    for i in range(len(face_locations)):
        if face_locations[i][0]>face_locations[i][2]:
            t1 = face_locations[i][2]
            t2 = face_locations[i][0]
        else:
            t2 = face_locations[i][2]
            t1 = face_locations[i][0]
            
            
        if face_locations[i][1]>face_locations[i][3]:
            t3 = face_locations[i][3]
            t4 = face_locations[i][1]
        else:
            t3 = face_locations[i][1]
            t4 = face_locations[i][3]

#       print(t1,t2,t3,t4)
        gap1 = (t2-t1)
        gap2 = (t4-t3)
        t1 = t1 - (gap1 // 2)
        t2 = t2 + (gap1 // 2)
        t3 = t3 - (gap2 // 2)
        t4 = t4 + (gap2 // 2)
        cropped_image = image[int(t1):int(t2), int(t3):int(t4)]
#        print(image.shape, cropped_image.shape)
        cnt = cnt + 1
        cv2.imwrite('textbook_crop/'+str(cnt)+'.png',cropped_image)
        f = open('textbook_recovery/'+str(cnt)+'.txt', 'w')
        f.write(str(j+1)+','+str(t1)+','+str(t2)+','+str(t3)+','+str(t4))
#        print(t4,t1,t3,t2,face_locations[i][1],face_locations[i][0],face_locations[i][3],face_locations[i][2])
        image = cv2.rectangle(image, (t4,t1), (t3,t2),(255,0,0))
    #print(len(face_locations))
    #cv2.imwrite('recog_result/'+str(j+1)+'.png',image)

cnt = 0
j = -1
for filename in glob.glob('./target/*.png'):
    j = j + 1
    image = face_recognition.load_image_file(filename)
    face_locations = face_recognition.face_locations(image)
    image = cv2.imread(filename)
    #image = cv2.resize(image, dsize=(0,0), fx=3, fy=3, interpolation = cv2.INTER_LINEAR)
    for i in range(len(face_locations)):
        if face_locations[i][0]>face_locations[i][2]:
            t1 = face_locations[i][2]
            t2 = face_locations[i][0]
        else:
            t2 = face_locations[i][2]
            t1 = face_locations[i][0]
            
            
        if face_locations[i][1]>face_locations[i][3]:
            t3 = face_locations[i][3]
            t4 = face_locations[i][1]
        else:
            t3 = face_locations[i][1]
            t4 = face_locations[i][3]

#        print(t1,t2,t3,t4)
        gap1 = (t2-t1)
        gap2 = (t4-t3)
        t1 = t1 - (gap1 // 2)
        t2 = t2 + (gap1 // 2)
        t3 = t3 - (gap2 // 2)
        t4 = t4 + (gap2 // 2)
        cropped_image = image[int(t1):int(t2), int(t3):int(t4)]
#        print(image.shape, cropped_image.shape)
        cnt = cnt + 1
        cv2.imwrite('target_crop/'+str(cnt)+'.png',cropped_image)
        f = open('target_recovery/'+str(cnt)+'.txt', 'w')
        f.write(str(j+1)+','+str(t1)+','+str(t2)+','+str(t3)+','+str(t4))
#        print(t4,t1,t3,t2,face_locations[i][1],face_locations[i][0],face_locations[i][3],face_locations[i][2])
        image = cv2.rectangle(image, (t4,t1), (t3,t2),(255,0,0))
    #print(len(face_locations))
    #cv2.imwrite('recog_result/'+str(j+1)+'.png',image)