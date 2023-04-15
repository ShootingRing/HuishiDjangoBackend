import cv2
cap = cv2.VideoCapture("C:/Users/11251/Desktop/FaceDetectionRecognition/video.avi")
fourcc = cv2.VideoWriter_fourcc(*'XVID')
# fps = cap.get(cv2.CAP_PROP_FPS)
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
out = cv2.VideoWriter('video1.avi', fourcc, 10.0, size)
while True:
    ret, frame = cap.read()
    out.write(frame)
    # 在图像上显示 Press Q to save and quit
    cv2.putText(frame,
                "Press Q to save and quit",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 255, 0), 2)
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()