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

#语言播报
import pyttsx3
# coding: utf-8




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







def strings(str1):
    result = '前方发现'+str(str1)
    engine = pyttsx3.init()
    engine.say(result)
    engine.runAndWait()

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
    cam_url = 'rtsp://admin:yin7372175240000@192.168.0.140:554/cam/realmonitor?channel=1&subtype=1'
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



    # fps.stop()
    # print('[INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))
    # print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))

    pool.terminate()
    video_capture.stop()
    cv2.destroyAllWindows()


