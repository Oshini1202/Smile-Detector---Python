import cv2

#Face and smile classifier
face_detector = cv2.CascadeClassifier("images and xml/haarcascade_frontalface_default.xml")
smile_detector = cv2.CascadeClassifier("images and xml/haarcascade_smile.xml")

#accsessing the webcam
webcam = cv2.VideoCapture(0)

while True:
    #Read the webcam foorage
    sucessful_frame_read, frame = webcam.read()

    if not sucessful_frame_read:
        break

    #Change to Gray Scale
    black_n_white = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    #Detect the face
    faces = face_detector.detectMultiScale(black_n_white)

    #Drawing rectangles on faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 200, 20), 4)

        #the frame is splited
        the_face = frame[y:y+h, x:x+h]

        # Change to Gray Scale
        face_black_n_white = cv2.cvtColor(the_face, cv2.COLOR_RGB2GRAY)

        smile = smile_detector.detectMultiScale(face_black_n_white, scaleFactor=1.7, minNeighbors=20)

        if len(smile)> 0:
            cv2.putText(frame,'smiling',(x, y+h+40), fontScale=3, fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255, 255, 255))

    #Show the current frame
    cv2.imshow("Smile Detector", frame)

    #Display
    key = cv2.waitKey(1)

    #To stop the program
    if key == 81 or key == 113:
        break

webcam.release





