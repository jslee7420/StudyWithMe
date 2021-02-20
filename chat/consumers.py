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
        faces_detected = face_detection_videocam.detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces_detected:
            cv2.rectangle(image, pt1=(x, y), pt2=(x + w, y + h),
                          color=(255, 0, 0), thickness=2)
        image = cv2.flip(image, 1)  # 이미지 좌우 반전
        # cv2 는 BGR로 변환 시킴 따라서 RGB로 다시 변환
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.imencode('.jpg', image)[1]
        # 원래 포멧으로 변경
        img_as_base64 = 'data:image/jpg;base64,' + \
            base64.b64encode(image).decode('UTF-8')

        return img_as_base64
