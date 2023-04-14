from rest_framework.response import Response
from djangoProjectVueProject.statuscode import statusCode
from APP.token import get_token, out_token
from APP import models


class login:

    def LoginSuccess(username, password):
        token = get_token(username, 60)
        models.User.objects.filter(username=username).update(token=token)
        return Response(
            data={
                'code': statusCode.OKCode,
                'message': 'Login Success',
                'data': {
                    'username': username,
                    'token': token
                }
            },
            status=statusCode.OKCode
        )

    WrongPassword = Response(
        data={
            'code': statusCode.PasswordWrongCode,
            'message': 'Wrong Password',
            'data': {}
        },
        status=statusCode.PasswordWrongCode
    )

    UserNotFound = Response(
        data={
            'code': statusCode.UserNotFoundCode,
            'message': 'User Not Found',
            'data': {}
        },
        status=statusCode.UserNotFoundCode
    )

    InvalidJSON = Response(
        data={
            'code': statusCode.InvalidJSONCode,
            'message': 'Invalid JSON request body',
            'data': {}
        },
        status=statusCode.InvalidJSONCode
    )

    MissingJSON = Response(
        data={
            'code': statusCode.MissingJSONCode,
            'message': 'Missing JSON request body',
            'data': {}
        },
        status=statusCode.MissingJSONCode
    )
