# # -*- coding: utf-8 -*-
# """
# Created on Thu Nov 22 18:41:33 2018
# #QQ群：476842922（欢迎加群讨论学习
# @author: Administrator
# """
# #以下是最常用的读取视频流的方法
import cv2
url = 'rtsp://admin:admin123@192.168.0.64:554/'#根据摄像头设置IP及rtsp端口
# url = 'rtsp://admin:888888@172.23.205.235:10554/udp/av0_0.mp4 '
cap = cv2.VideoCapture(url)#读取视频流
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


# import cv2
#
# import time
# import multiprocessing as mp

"""
Source: Yonv1943 2018-06-17
https://github.com/Yonv1943/Python/tree/master/Demo
"""

# def image_put(q, name, pwd, ip, channel=1):
#     cap = cv2.VideoCapture("rtsp://%s:%s@%s//stream/realtime?channel=%d&streamtype=0" % (name, pwd, ip, channel))
#     if cap.isOpened():
#         print('HIKVISION')
#     while True:
#         q.put(cap.read()[1])
#         q.get() if q.qsize() > 1 else time.sleep(0.0001)
#
# def image_get(q, window_name):
#     cv2.namedWindow(window_name, flags=cv2.WINDOW_FREERATIO)
#     while True:
#         frame = q.get()
#         cv2.imshow(window_name, frame)
#         cv2.waitKey(1)
#
# def run_multi_camera():
#     # user_name, user_pwd = "admin", "password"
#     user_name, user_pwd = "admin", "admin123"
#     camera_ip_l = [
#         "192.168.0.64",  # ipv4
#     ]
#
#     mp.set_start_method(method='spawn')  # init
#     queues = [mp.Queue(maxsize=4) for _ in camera_ip_l]
#
#     processes = []
#     for queue, camera_ip in zip(queues, camera_ip_l):
#         processes.append(mp.Process(target=image_put, args=(queue, user_name, user_pwd, camera_ip)))
#         processes.append(mp.Process(target=image_get, args=(queue, camera_ip)))
#
#     for process in processes:
#         process.daemon = True
#         process.start()
#     for process in processes:
#         process.join()
#
# if __name__ == '__main__':
#     run_multi_camera()