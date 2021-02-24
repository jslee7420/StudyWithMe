import light_remover as lr
import make_train_data as mtd

from threading import Timer
from threading import Thread
from imutils import face_utils
from scipy.spatial import distance as dist
import dlib
import timeit
import time
import imutils
import json
from channels.generic.websocket import AsyncWebsocketConsumer
import base64

from io import BytesIO
from PIL import Image
import cv2
import os
import urllib.request
import numpy as np
from django.conf import settings
face_detection_videocam = cv2.CascadeClassifier(os.path.join(
    settings.BASE_DIR, 'opencv_haarcascade_data/haarcascade_frontalface_default.xml'))


class ChatConsumer(AsyncWebsocketConsumer):
    def __init__(self):
        #####################################################################################################################
        # 1. Variables for checking EAR.
        # 2. Variables for detecting if user is asleep.
        # 3. When the alarm rings, measure the time eyes are being closed.
        # 4. When the alarm is rang, count the number of times it is rang, and prevent the alarm from ringing continuously.
        # 5. We should count the time eyes are being opened for data labeling.
        # 6. Variables for trained data generation and calculation fps.
        # 7. Detect face & eyes.
        # 8. Run the cam.
        # 9. Threads to run the functions in which determine the EAR_THRESH.

        # 1.
        # self.OPEN_EAR = 0  # For init_open_ear()
        # self.EAR_THRESH = 0  # Threashold value

        # # 2.
        # # It doesn't matter what you use instead of a consecutive frame to check out drowsiness state. (ex. timer)
        # self.EAR_CONSEC_FRAMES = 20
        # self.COUNTER = 0  # Frames counter.

        # # 3.
        # self.closed_eyes_time = []  # The time eyes were being offed.
        # # Flag to activate 'start_closing' variable, which measures the eyes closing time.
        # self.TIMER_FLAG = False
        # # Flag to check if alarm has ever been triggered.
        # self.ALARM_FLAG = False

        # # 4.
        # self.ALARM_COUNT = 0  # Number of times the total alarm rang.
        # # Variable to prevent alarm going off continuously.
        # self.RUNNING_TIME = 0

        # # 5.
        # # Variable to measure the time eyes were being opened until the alarm rang.
        # self.PREV_TERM = 0

        # # 6. make trained data
        # np.random.seed(9)
        # # actually this three values aren't used now. (if you use this, you can do the plotting)
        # self.power, self.nomal, self.short = mtd.start(25)
        # # The array the actual test data is placed.
        # self.test_data = []
        # # The array the actual labeld data of test data is placed.
        # self.result_data = []
        # # For calculate fps
        # self.prev_time = 0

        # 7.
        print("loading facial landmark predictor...")
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(
            "shape_predictor_68_face_landmarks.dat")

        (self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (self.rStart,
         self.rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]\
            # self.th_open.deamon = True
        # self.th_open.start()
        # self.th_close = Thread(target=self.init_close_ear)
        # self.th_close.deamon = True
        # self.th_close.start()

    async def connect(self):
        self.room_name = self.scope['url_route']['kwargs']['room_name']
        self.room_group_name = 'chat_%s' % self.room_name

        # Join room group
        await self.channel_layer.group_add(
            self.room_group_name,
            self.channel_name
        )

        await self.accept()

    async def disconnect(self, close_code):
        # Leave room group
        await self.channel_layer.group_discard(
            self.room_group_name,
            self.channel_name
        )

    # # Receive message from WebSocket
    # async def receive(self, text_data):

    #     # Send message to room group
    #     await self.channel_layer.group_send(
    #         self.room_group_name,
    #         {
    #             'type': 'chat_message',
    #             'message': text_data
    #         }
    #     )

    # # Receive message from room group
    # async def chat_message(self, event):
    #     message = event['message']

    #     # Send message to WebSocket
    #     await self.send({
    #         'message': message
    #     })

    # Receive message from WebSocket
    async def receive(self, text_data):

        text_data_json = json.loads(text_data)
        data_url = text_data_json['message']

        if len(data_url) < 30:   # 정상적인 이미지 데이터가 오지 않은 경우 하단 코드 수행 생략
            return

        # 얼굴탐색
        message = self.face_decting(data_url)

        # Send message to room group
        await self.channel_layer.group_send(
            self.room_group_name,
            {
                'type': 'chat_message',
                'message': message
            }
        )

    # Receive message from room group
    async def chat_message(self, event):
        message = event['message']

        # Send message to WebSocket
        await self.send(text_data=json.dumps({
            'message': message
        }))

    def face_decting(self, data_url):
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.

        # 앞에 base:~ 없애고 이미지 bytes만 추출
        offset = data_url.index(',')+1
        # Decoding base64 string to bytes object
        img_bytes = base64.b64decode(data_url[offset:])

        img = Image.open(BytesIO(img_bytes))
        image = np.array(img)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # faces_detected = face_detection_videocam.detectMultiScale(
        #     gray, scaleFactor=1.3, minNeighbors=5)
        # for (x, y, w, h) in faces_detected:
        #     cv2.rectangle(image, pt1=(x, y), pt2=(x + w, y + h),
        #                   color=(255, 0, 0), thickness=2)
        # image = cv2.flip(image, 1)  # 이미지 좌우 반전
        # # cv2 는 BGR로 변환 시킴 따라서 RGB로 다시 변환
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = cv2.imencode('.jpg', image)[1]
        # # 원래 포멧으로 변경
        # img_as_base64 = 'data:image/jpg;base64,' + \
        #     base64.b64encode(image).decode('UTF-8')

        # return img_as_base64

        #####################################################################################################################
        # frame = img

        # L, gray = lr.light_removing(frame)

        rects = self.detector(gray, 0)

        # checking fps. If you want to check fps, just uncomment below two lines.
        #self.prev_time, fps = check_fps(self.prev_time)
        #cv2.putText(frame, "fps : {:.2f}".format(fps), (10,130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,30,20), 2)

        for rect in rects:
            shape = self.predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            leftEye = shape[self.lStart:self.lEnd]
            rightEye = shape[self.rStart:self.rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            # (leftEAR + rightEAR) / 2 => both_ear.
            # I multiplied by 1000 to enlarge the scope.
            both_ear = (leftEAR + rightEAR) * 500

            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            image = cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            image = cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            # if both_ear < EAR_THRESH :
            #     if not self.TIMER_FLAG:
            #         start_closing = timeit.default_timer()
            #         self.TIMER_FLAG = True
            #     self.COUNTER += 1

            #     if self.COUNTER >= self.EAR_CONSEC_FRAMES:

            #         mid_closing = timeit.default_timer()
            #         closing_time = round((mid_closing-start_closing),3)

            #         if closing_time >= self.RUNNING_TIME:
            #             if self.RUNNING_TIME == 0 :
            #                 CUR_TERM = timeit.default_timer()
            #                 OPENED_EYES_TIME = round((CUR_TERM - self.PREV_TERM),3)
            #                 self.PREV_TERM = CUR_TERM
            #                 self.RUNNING_TIME = 1.75

            #             self.RUNNING_TIME += 2
            #             self.ALARM_FLAG = True
            #             self.ALARM_COUNT += 1

            #             print("{0}st ALARM".format(self.ALARM_COUNT))
            #             print("The time eyes is being opened before the alarm went off :", OPENED_EYES_TIME)
            #             print("closing time :", closing_time)
            #             self.test_data.append([OPENED_EYES_TIME, round(closing_time*10,3)])
            #             result = mtd.run([OPENED_EYES_TIME, closing_time*10], self.power, self.nomal, self.short)
            #             self.result_data.append(result)
            #             t = Thread(target = alarm.select_alarm, args = (result, ))
            #             t.deamon = True
            #             t.start()

            # else :
            #     self.COUNTER = 0
            #     self.TIMER_FLAG = False
            #     self.RUNNING_TIME = 0

            #     if self.ALARM_FLAG :
            #         end_closing = timeit.default_timer()
            #         self.closed_eyes_time.append(round((end_closing-start_closing),3))
            #         print("The time eyes were being offed :", self.closed_eyes_time)

            #     self.ALARM_FLAG = False

            image = cv2.putText(image, "EAR : {:.2f}".format(
                both_ear), (300, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 30, 20), 2)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.imencode('.jpg', image)[1]
            # 원래 포멧으로 변경
            img_as_base64 = 'data:image/jpg;base64,' + \
                base64.b64encode(image).decode('UTF-8')
            print('hello')
            return img_as_base64

    # def eye_aspect_ratio(self, eye):
    #     A = dist.euclidean(eye[1], eye[5])
    #     B = dist.euclidean(eye[2], eye[4])
    #     C = dist.euclidean(eye[0], eye[3])
    #     ear = (A + B) / (2.0 * C)
    #     return ear

    # def init_open_ear(self):
    #     time.sleep(5)
    #     print("open init time sleep")
    #     ear_list = []
    #     th_message1 = Thread(target=self.init_message)
    #     th_message1.deamon = True
    #     th_message1.start()
    #     for i in range(7):
    #         ear_list.append(both_ear)
    #         time.sleep(1)
    #     self.OPEN_EAR
    #     self.OPEN_EAR = sum(ear_list) / len(ear_list)
    #     print("open list =", ear_list, "\nOPEN_EAR =", self.OPEN_EAR, "\n")

    # def init_close_ear(self):
    #     time.sleep(2)
    #     self.th_open.join()
    #     time.sleep(5)
    #     print("close init time sleep")
    #     ear_list = []
    #     th_message2 = Thread(target=self.init_message)
    #     th_message2.deamon = True
    #     th_message2.start()
    #     time.sleep(1)
    #     for i in range(7):
    #         ear_list.append(self.both_ear)
    #         time.sleep(1)
    #     CLOSE_EAR = sum(ear_list) / len(ear_list)
    #     # EAR_THRESH means 50% of the being opened eyes state
    #     self.EAR_THRESH = (((self.OPEN_EAR - CLOSE_EAR) / 2) + CLOSE_EAR)
    #     print("close list =", ear_list, "\nCLOSE_EAR =", CLOSE_EAR, "\n")
    #     print("The last EAR_THRESH's value :", self.EAR_THRESH, "\n")

    # def init_message(self):
    #     print("init_message")
