import cv2
capss= 'rtsp://admin:12345678@192.168.0.132:10554//udp//av0_0'
cap = cv2.VideoCapture(capss)# 调整参数实现读取视频或调用摄像头
while (cap.isOpened):
    ret, frame = cap.read()
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()