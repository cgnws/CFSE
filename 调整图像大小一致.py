import os
import cv2


def resize_img(DATADIR, data_k, img_size):
    w = img_size[0]
    h = img_size[1]
    path = os.path.join(DATADIR, data_k)
    # 返回path路径下所有文件的名字，以及文件夹的名字，
    img_list = os.listdir(path)#os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表。

    for i in img_list:
        if i.endswith('.jpg'):#判断图片名称是否以".jpg"结尾
            # 调用cv2.imread读入图片，读入格式为IMREAD_COLOR
            # img_array = cv2.imread(r"人框照片\\CLPD_1200\1.jpg", cv2.IMREAD_COLOR)
            img_array = cv2.imread((path + '/' + i), cv2.IMREAD_COLOR)
            Oh = img_array.shape[0]
            Ow = img_array.shape[1]
            # 调用cv2.resize函数resize图片
            _w = int(Ow * h/Oh)
            # 最近邻插值可以有效防范边缘模糊
            new_array = cv2.resize(img_array, (_w, h), interpolation=cv2.INTER_NEAREST)
            # if h > Oh:
            #     new_array = cv2.resize(img_array, (_w, h), interpolation=cv2.INTER_CUBIC)
            # elif h < Oh:
            #     new_array = cv2.resize(img_array, (_w, h), interpolation=cv2.INTER_AREA)
            img_name = str(i)
            '''生成图片存储的目标路径'''
            save_path = path + 'CLPD_1200_new/'
            if os.path.exists(save_path):
                print(i)
                '''调用cv.2的imwrite函数保存图片'''
                save_img = save_path + img_name
                cv2.imwrite(save_img, new_array)
            else:
                os.mkdir(save_path)
                save_img = save_path + img_name
                cv2.imwrite(save_img, new_array)


if __name__ == '__main__':
    # 设置图片路径
    DATADIR = "peopleresize"
    data_k = 'CLPD_1200'
    '''设置目标像素大小，此处设为512 * 512'''
    img_size = [88, 128]
    resize_img(DATADIR, data_k, img_size)
