import os
import cv2
import time
import argparse
import multiprocessing
import numpy as np
import tensorflow as tf
from utils.app_utils import FPS, WebcamVideoStream, HLSVideoStream
from multiprocessing import Queue, Pool
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import pygame  # pip install pygame
from playsound import playsound
#语言播报
from aip import AipSpeech
import pyttsx3
# coding: utf-8
import urllib.request
import urllib.parse
import json

""" 你的 APPID AK SK """
APP_ID = '16126605'
API_KEY = '4LMBg2SzOGOaSu5OONNUkUYw'
SECRET_KEY = 'LOZcgIxnmG4kXdxGaGvEtia1vQ2zaoLl'
client = AipSpeech(APP_ID, API_KEY, SECRET_KEY)#Api常规设置



CWD_PATH = os.getcwd()

# Path to frozen detection graph. This is the actual model that is used for the object detection.
MODEL_NAME = 'ssd_mobilenet_v1_coco_2018_01_28'
PATH_TO_CKPT = os.path.join(CWD_PATH, 'object_detection', MODEL_NAME, 'frozen_inference_graph.pb')

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(CWD_PATH, 'object_detection', 'data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)






def get_data(words):
    data = {}
    data["type"] = "AUTO"
    data["i"] = words
    data["doctype"] = "json"
    data["xmlVersion"] = "1.8"
    data["keyfrom:fanyi"] = "web"
    data["ue"] = "UTF-8"
    data["action"] = "FY_BY_CLICKBUTTON"
    data["typoResult"] = "true"
    data = urllib.parse.urlencode(data).encode('utf-8')
    return data


def url_open(url, data):
    req = urllib.request.Request(url, data)
    req.add_header("User-Agent",
                   "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/44.0.2403.157 Safari/537.36")
    response = urllib.request.urlopen(req)
    html = response.read()
    html = html.decode("utf-8")
    return html


def get_json_data(html):
    result = json.loads(html)
    result = result['translateResult']
    result = result[0][0]['tgt']
    return result


def main(words):
    # words = input("please input words: ")
    url = "http://fanyi.youdao.com/translate?smartresult=dict&smartresult=rule&smartresult=ugc&sessionFrom=dict.top"

    data = get_data(words)
    html = url_open(url, data)
    return  get_json_data(html)


# if __name__ == "__main__":
#     while True:
#         main()


def strings(str1):
    result = '前方发现'+str(str1)
    engine = pyttsx3.init()
    engine.say(result)
    engine.runAndWait()
#     print(str1)
#     # 识别正确返回语音二进制 错误则返回dict 参照下面错误码
#     while 1:
#         if not isinstance(result, dict):
#             with open('auido.mp3', 'wb') as f:
#                 f.write(result)
#         playsound("auido.mp3")
#             # playMusic('auido.mp3')
#
# # 貌似只能播放单声道音乐，可能是pygame模块限制
# def playMusic(filename, loops=0, start=0.0, value=0.5):
#     """
#     :param filename: 文件名
#     :param loops: 循环次数
#     :param start: 从多少秒开始播放
#     :param value: 设置播放的音量，音量value的范围为0.0到1.0
#     :return:
#     """
#     flag = False  # 是否播放过
#     pygame.mixer.init()  # 音乐模块初始化
#     while 1:
#         if flag == 0:
#             pygame.mixer.music.load(filename)
#             # pygame.mixer.music.play(loops=0, start=0.0) loops和start分别代表重复的次数和开始播放的位置。
#             pygame.mixer.music.play(loops=loops, start=start)
#             pygame.mixer.music.set_volume(value)  # 来设置播放的音量，音量value的范围为0.0到1.0。
#         if pygame.mixer.music.get_busy() == True:
#             flag = True
#         else:
#             if flag:
#                 pygame.mixer.music.stop()  # 停止播放
#                 break

def detect_objects(image_np, sess, detection_graph):
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Each box represents a part of the image where a particular object was detected.
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Actual detection.
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})

    # Visualization of the results of a detection.检测结果的可视化
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8)
    return image_np


def worker(input_q, output_q):
    # Load a (frozen) Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)

    fps = FPS().start()
    while True:
        fps.update()
        frame = input_q.get()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output_q.put(detect_objects(frame_rgb, sess, detection_graph))

    fps.stop()
    sess.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-str', '--stream', dest="stream", action='store', type=str, default=None)
    parser.add_argument('-src', '--source', dest='video_source', type=int,
                        default=0, help='Device index of the camera.')
    parser.add_argument('-wd', '--width', dest='width', type=int,
                        default=480, help='Width of the frames in the video stream.')
    parser.add_argument('-ht', '--height', dest='height', type=int,
                        default=360, help='Height of the frames in the video stream.')
    parser.add_argument('-num-w', '--num-workers', dest='num_workers', type=int,
                        default=2, help='Number of workers.')
    parser.add_argument('-q-size', '--queue-size', dest='queue_size', type=int,
                        default=5, help='Size of the queue.')
    args = parser.parse_args()

    logger = multiprocessing.log_to_stderr()
    logger.setLevel(multiprocessing.SUBDEBUG)

    input_q = Queue(maxsize=args.queue_size)
    output_q = Queue(maxsize=args.queue_size)
    pool = Pool(args.num_workers, worker, (input_q, output_q))

    # webcamera
    # cam_url = 'rtsp://admin:admin123@192.168.0.64:554/stream/realtime?channel=1&streamtype=0'
    cam_url = 'rtsp://admin:yin7372175240000@192.168.0.108:554/cam/realmonitor?channel=1&subtype=1'
    video_capture = cv2.VideoCapture(cam_url)
    # video_capture = WebcamVideoStream(src=args.video_source,
    #                                 width=args.width,
    #                                height=args.height).start()
    fps = FPS().start()
    t_start = time.time()
    # out = None
    while True:  # fps._numFrames < 120
        if time.time() - t_start > 0.19:  # 因为识别速度和视频帧率相差过大，为了使输出图像与q摄像头输入保持同步，所以每两秒输出一次识别结果。该参数可以根据计算性能加以调整
            t_start = time.time()
            _, frame = video_capture.read()
            frame = cv2.resize(frame, (3 * args.width, 800), interpolation=cv2.INTER_AREA)
            input_q.put(frame)

            t = time.time()
            output_rgb = cv2.cvtColor(output_q.get(), cv2.COLOR_RGB2BGR)

            cv2.imshow('Video', output_rgb)
            fps.update()

            print('[INFO] elapsed time: {:.2f}'.format(time.time() - t))

            if cv2.waitKey(1) == ord('r'):
                message = open('1.txt').read()
                print(message)
                message = main(message)
                strings(message)

        else:
            _, frame = video_capture.read()

            # if out is not None:
            #     cv2.imshow("Video", out)
            # else:
            #     cv2.imshow("Video", frame)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break







    if (args.stream):
        print('Reading from hls stream.')
        video_capture = HLSVideoStream(src=args.stream).start()
    else:
        print('Reading from webcam.')
        video_capture = WebcamVideoStream(src=args.video_source,
                                      width=args.width,
                                      height=args.height).start()


    # fps = FPS().start()
    #
    # while True:  # fps._numFrames < 120
    #     frame = video_capture.read()
    #     frame=cv2.flip(frame,1)
    #     input_q.put(frame)
    #
    #     t = time.time()
    #
    #     output_rgb = cv2.cvtColor(output_q.get(), cv2.COLOR_RGB2BGR)
    #     cv2.imshow('Video', output_rgb)
    #     fps.update()
    #
    #     print('[INFO] elapsed time: {:.2f}'.format(time.time() - t))
    #
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break



    fps.stop()
    print('[INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))
    print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))

    pool.terminate()
    video_capture.stop()
    cv2.destroyAllWindows()


