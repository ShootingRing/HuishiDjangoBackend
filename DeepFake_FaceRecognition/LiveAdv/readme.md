# LiveDetection部分：
    直接运行MtcnnLiveDetect.py文件；
    pre_precessing（）函数内包括保存视频的路径
if args.video:
    print('开启视频录制...')
    now = time.strftime("%m-%d %H-%M-%S", time.localtime())
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('D:/dataset/video/{}.mp4'.format(now), fourcc, 15, (1280, 720))

# AdvDetection部分：
    直接运行views.py文件
    data中包括读取路径和保存路径（需要调整）
data = {
    'type':'video',
    'file_path':r'D:\dataset\video\02-13 20-40-38.mp4',
    'model_name':'xception',
    'save_file':r'D:\dataset\video\02-13 20-40-38detect.mp4',
}