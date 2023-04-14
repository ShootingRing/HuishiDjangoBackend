import base64
import json
import os
import sys

from rest_framework.decorators import api_view

import response

from django.http import HttpResponse, HttpResponseBadRequest, StreamingHttpResponse
from django.shortcuts import render
from django.views.decorators.http import require_http_methods
from APP import models

from . import statuscode

# 将 G:/algorithm 和 G:/algorithm/DeepFake_FaceRecognition 添加到 sys.path
# sys.path.append(os.path.abspath('G:/algorithm'))
# sys.path.append(os.path.abspath('G:/algorithm/DeepFake_FaceRecognition'))
sys.path.append(os.path.abspath('G:/DeepFake_FaceRecognition (3)'))
sys.path.append(os.path.abspath('G:/DeepFake_FaceRecognition (3)/DeepFake_FaceRecognition'))
sys.path.append(os.path.abspath('G:/DeepFake_FaceRecognition (3)/DeepFake_FaceRecognition/LiveAdv'))
from DeepFake_FaceRecognition.DETECT import detect


@api_view(['POST'])
def login(request):
    print(request.body)
    if request.body:
        try:
            data = json.loads(request.body.decode('utf-8'))
            username = data.get('username')
            password = data.get('password')
            print(username + '  ' + password)
            try:
                print(models.User.objects.all())
                user = models.User.objects.get(username=username)
                # check if password matches
                if user.password != password:
                    return response.login.WrongPassword

                # return HttpResponse('True', status=statuscode.statusCode.OKCode)
                return response.login.LoginSuccess(username, password)

            except models.User.DoesNotExist:
                return response.login.UserNotFound

        except json.decoder.JSONDecodeError:
            return response.login.InvalidJSON

    else:
        return response.login.MissingJSON


@api_view(['GET'])
def userlist(request):
    userQueryset = models.User.objects.all()
    return render(request, 'userlist.html', {'userQueryset': userQueryset})


@api_view(['POST'])
def detection(request):
    if request.body:
        try:
            data = json.loads(request.body.decode('utf-8'))
            video = data.get('data')
            print(video)
            video = video.split(',')[2]
            print(len(video))
            # base64 string to bytes
            # video = bytes(video, encoding='utf-8')
            binary_video = base64.b64decode(video)
            # base64 to mp4 file and save to /video/stream.mp4
            with open('video/stream.mp4', 'wb') as f:
                f.write(binary_video)
            # detect
            detect()

        except json.decoder.JSONDecodeError:
            return HttpResponseBadRequest("Invalid JSON request body", status=statuscode.statusCode.InvalidJSONCode)
    # detect(request.GET['data'])
    # return StreamingHttpResponse(detect(), content_type='multipart/x-mixed-replace; boundary=frame')
    return HttpResponse('True', status=statuscode.statusCode.OKCode)
