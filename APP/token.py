import time
import base64
import hmac


def get_token(key, expire=3600):
    """
    生成token
    :param key: 用户的唯一标识
    :param expire: 过期时间
    :return: token
    """
    ts_str = str(int(time.time()) + expire)  # 过期时间
    ts_byte = ts_str.encode('utf-8')
    # 生成签名
    sha1_tshexstr = hmac.new(key.encode("utf-8"), ts_byte, 'sha1').hexdigest()
    # 生成token
    token = ts_str + ':' + sha1_tshexstr
    b64_token = base64.urlsafe_b64encode(token.encode('utf-8'))
    return b64_token.decode('utf-8')


def out_token(key, token):
    """
    校验token
    :param key: 用户的唯一标识
    :param token: 生成的token
    :return: True or False
    """
    # token是前端传过来的token字符串
    try:
        token_str = base64.urlsafe_b64decode(token).decode('utf-8')
        token_list = token_str.split(':')
        if len(token_list) != 2:
            return False
        ts_str = token_list[0]
        if float(ts_str) < time.time():  # time.time()返回当前时间的时间戳
            # token expired
            return False
        known_sha1_tsstr = token_list[1]
        sha1 = hmac.new(key.encode("utf-8"), ts_str.encode('utf-8'), 'sha1')
        calc_sha1_tsstr = sha1.hexdigest()
        if calc_sha1_tsstr != known_sha1_tsstr:
            # token certification failed
            return False
        # token certification success
        return True
    except Exception as e:
        print(e)
