import multiprocessing as mp
import cv2
import time
import pickle
import numpy as np
import socket
import time
# 服务器用Server，本地主机用Client

GNum = 4 # 每个步态帧发送的图像数目
SNum = 5 # 每个动作帧隐藏的图像数目
NumPerS = 15 # 每秒读取图片

def queue_img_put(q1, q2, max_q_num=NumPerS):  # 通过 rtsp 协议读取IP摄像头的视频流
    # q2保存离散帧，q1保存所有帧
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(r"F:\Desktop\Yolov5_StrongSORT_OSNet-test\person.mp4")
    i = 0  # 用于给视频帧标序号

    while True:
        frame = cap.read()[1]
        input = [i, frame]  # 消息组成为[序号，帧]
        q1.put(input)
        if i%SNum == 0:  # i为5的倍数时，q2序列读取
            q2.put(input)
            # print("q2", i)

        if q1.qsize() > NumPerS-SNum:  # q1 5帧 q2 1帧 保持队列内数据同步
            q1.get()  # 超出最大范围时，删除队列最早的数据
            q1.get()
            q1.get()
            q1.get()
            q1.get()
            q2.get()

        time.sleep(1/NumPerS)  # 单位秒，多进程中留给其他程序时间，20帧/秒

        # print("q1", i)
        i += 1


def ScreenShot(cropsArchors, frames):  # 截图，输出 编号+图 序列
    first, other = cropsArchors
    screenshot = []

    for i in range(SNum):
        img = frames[i][1]
        if i == 0:
            _s = []
            for k in range(first.shape[0]):
                _s.append([first[k, 4], img[first[k, 1]:first[k, 3], first[k, 0]:first[k, 2], :]])
            screenshot.append(_s)

        else:
            j = i-1
            _s = []
            for k in range(other.shape[1]):
                _s.append([other[j, k, 4], img[other[j, k, 1]:other[j, k, 3], other[j, k, 0]:other[j, k, 2], :]])
            screenshot.append(_s)

    return screenshot


def PredictArchor(outputs, preoutputs):  # 调整格式，输出为列表[第一帧,后4帧]，前四个值锚点，最后一个值id
    AllArchor = []  # 5张图，最多15个人框，（1人序号+4边界点）
    coID = np.intersect1d(outputs[:,4], preoutputs[:,4])
    if len(coID) == 0:
        AllArchor = np.zeros([1,outputs.shape[0],5])
        AllArchor[0,:,:] = outputs[:, :5]
        return AllArchor
    OtherArchors = np.zeros([SNum-1,len(coID),5])

    AllArchor.append(outputs[:, :5])
    for k in range(SNum-1):
        for i in range(outputs.shape[0]):
            if outputs[i, 4] in coID:
                for j in range(preoutputs.shape[0]):
                    if outputs[i, 4] == preoutputs[j, 4]:
                        gap = (preoutputs[j, :5] - outputs[i, :5]) / SNum
                        for x in range(len(coID)):
                            OtherArchors[k, x, :] = outputs[i, :5] + gap * (x+1)

    AllArchor.append(OtherArchors.astype(np.int16))

    return AllArchor


def queue_img_get(q1, q2, window_name, host, port):  # 通过 rtsp 协议读取IP摄像头的视频流
    # cv2.namedWindow(window_name, flags=cv2.WINDOW_FREERATIO)
    # q2保存离散帧，q1保存所有帧

    from multiprocessing.connection import Client
    client = Client((host, port))

    # frame = q2.get()
    # shape = np.array(frame[1].shape[:2]) // 2  # 图像压缩一半宽高
    # shape = tuple(shape[::-1])
    frames = []
    RGBCrops5 = []
    preoutputs = []
    while True:
        frame = q2.get()
        if frame[0]%SNum == 0 and frame[0] != 0:
            frames.append(q1.get())
            frames.append(q1.get())
            frames.append(q1.get())
            frames.append(q1.get())
            frames.append(q1.get())

        cropsArchors = []
        ActionCrops = []
        if len(preoutputs) != 0 and len(outputs) != 0:  # 截图不能是第一张
            cropsArchors = PredictArchor(outputs.astype(np.int16), preoutputs.astype(np.int16))
            ActionCrops = ScreenShot(cropsArchors, frames)  # 截图 list:5  list:1 list:2 [id,frame]

        # frame = cv2.resize(frame, shape)
        data_string = pickle.dumps([frame, ActionCrops])  # 第一次发送，frame全图， ActionCrops人像截图
        print("send", frame[0])
        client.send(data_string)


        data_string = client.recv()  # 第一次接收
        outputs = pickle.loads(data_string)  # ndarray(n,7)
        # data_string = pickle.dumps(0)  # 第二次发送
        # client.send(data_string)


        preoutputs = outputs
        frames.clear()

        if cv2.waitKey(10) & 0xff == 27:  # 输入esc退出程序
            break

    client.close()
    print("imgGet___over")


def run_client(host, port):
    user_name, user_pwd, camera_ip = "admin", "你网络摄像头的密码", "172.20.114.26"

    mp.set_start_method(method='spawn')  # init
    max_q_num = 20
    queue1 = mp.Queue(maxsize=max_q_num)  # 队列最大值
    queue2 = mp.Queue(maxsize=max_q_num)
    processes = [
        mp.Process(target=queue_img_put, args=(queue1, queue2, max_q_num)),
        mp.Process(target=queue_img_get, args=(queue1, queue2, camera_ip, host, port)),
    ]

    [setattr(process, "daemon", True) for process in processes]  # process.daemon = True
    [process.start() for process in processes]
    [process.join() for process in processes]
    print("client___over")


if __name__ == '__main__':
    # server_host = '10.101.166.188'
    server_host = 'localhost'
    server_port = 33928  # if [Address already in use], use another port


    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    run_client(server_host, server_port)  # then, run this function only in client
