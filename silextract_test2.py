from sklearn.cluster import MiniBatchKMeans  #提升聚类速度
from imfillCanny import IMFILL
import numpy as np
import cv2
import time

class SILEXTRACT(object):
    def __init__(self,BGin,BGout):
        self.BGin = BGin
        self.BGout = BGout
        self.H = 128
        self.W = 88
        self.conv_kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    def uniformIMG(self,img):
        Oh = img.shape[0]
        Ow = img.shape[1]
        # 调用cv2.resize 函数 resize 图片
        _w = int(Ow * self.H/Oh)
        # 最近邻插值可以有效防范边缘模糊
        new_img = cv2.resize(img, (_w, self.H), interpolation=cv2.INTER_NEAREST)

        return _w, self.H, new_img

    def prepareIMG(self,gray_img,KNNimg, seg_img):
        org_shape0 = KNNimg.shape[0]
        org_shape1 = KNNimg.shape[1]

        # 处理实例分割图像 若为两轮廓则去掉外侧轮廓
        _, seg_img = cv2.threshold(seg_img, 127, 255, cv2.THRESH_BINARY)

        img = cv2.erode(seg_img, self.conv_kernel3)  # 腐蚀
        SEGimgin = self.RESERVE_MAX_CONTOUR(img)
        SEGimgin = SEGimgin/255

        img = cv2.dilate(SEGimgin*255, self.conv_kernel3)  # 膨胀
        SEGimgout = self.RESERVE_MAX_CONTOUR(img)
        SEGimgout = SEGimgout/255

        BGin = cv2.resize(self.BGin, [org_shape1, org_shape0], interpolation=cv2.INTER_CUBIC)
        _, BGin = cv2.threshold(BGin, 127, 1, cv2.THRESH_BINARY)
        BGout = cv2.resize(self.BGout, [org_shape1, org_shape0], interpolation=cv2.INTER_CUBIC)
        _, BGout = cv2.threshold(BGout, 127, 1, cv2.THRESH_BINARY)

        sum = np.sum(seg_img, axis=0)  # 对每列求和，判断白色区域是否贴近边缘
        if sum[0]>10 or sum[-1]>10:
            newBGin = BGin*SEGimgin
            newBGout = (1-SEGimgout) #+ (1-BGout) # 贴外侧说明内侧足够大，无需人工外侧
        else:
            newBGin = SEGimgin
            newBGout = (1-SEGimgout)# + (1-BGout)  # 必须有人工外侧
        newBGin[newBGin>0] = 1
        newBGout[newBGout>0] = 1
        newBGout = 1- newBGout

        # 绝对人体外侧图
        newBGout_out = 1-newBGout
        # 绝对人体内侧图
        newBGin_in = newBGin
        newBGin_out = 1-newBGin_in
        # 对knn进行内部填充
        _ = IMFILL(newBGin_in)
        CannyBG = _.BGjoin(gray_img, KNNimg)

        # 真正的背景外侧分割应该是 WH外侧+填充图黑色区域-WH2内侧
        BG = (newBGout_out + CannyBG) * newBGin_out
        _, BG = cv2.threshold(BG, 0.5, 1, cv2.THRESH_BINARY)
        # BG = cv2.erode(BG, self.conv_kernel3)  # 腐蚀
        _BG_out = BG#.reshape(-1)

        return _BG_out, newBGin_in  # 背景分割外侧，绝对人体内侧

    def SELECT_DEL_COLOR(self, color_out_rate, color_in_rate, color_num, INcolor_thresh=0.01):  # 外部颜色占比，内部颜色占比，颜色类别数
        all_color = np.array(range(color_num))
        del_color = []
        for i in range(color_num):
            # 外侧大于内侧，或内侧占比过低
            if color_out_rate[i] > color_in_rate[i] or color_in_rate[i] < INcolor_thresh:
                del_color.append(all_color[i])

        return del_color

    def COLOR_DEL(self, bgOut, bgIn, img, color_num, INcolor_thresh):  # 统计并删除颜色
        # 外侧背景与人体绝对内部颜色统计
        maskBGout = bgOut != 0
        img_out = img[maskBGout]
        _color_out_num = np.bincount(img_out)  # 统计元素数目，输入为整数,输出为颜色0-max数目
        color_out_num = np.zeros(color_num)  # 防止内外颜色数目不一致
        color_out_num[:len(_color_out_num)] = _color_out_num
        out_sum = np.sum(bgOut)+1e-4
        color_out_rate = color_out_num/out_sum

        maskWH2in = bgIn != 0
        img_in = img[maskWH2in]
        _color_in_num = np.bincount(img_in)
        color_in_num = np.zeros(color_num)
        color_in_num[:len(_color_in_num)] = _color_in_num
        in_sum = np.sum(bgIn)+1e-4
        color_in_rate = color_in_num/in_sum  # 内部颜色占比翻倍，增强内部颜色比重

        # 需删除颜色
        del_color = self.SELECT_DEL_COLOR(color_out_rate, color_in_rate, color_num, INcolor_thresh)

        #颜色消除-------------------------------------------------------------------
        # 消除全图
        fin_data = 1-np.isin(img, del_color).astype(np.uint8)

        return fin_data*255

    def COLOR_DEL2(self, bgOut, bgIn, img, color_num, INcolor_thresh):  # 统计并删除颜色

        mask = (bgIn.reshape(-1)).astype(bool)
        img_in = (img.reshape(-1))[mask]
        _color_in_num = np.bincount(img_in)
        color_in_num = np.zeros(color_num)
        color_in_num[:len(_color_in_num)] = _color_in_num
        in_sum = np.sum(bgIn)+1e-4
        color_in_rate = color_in_num/in_sum  # 内部颜色占比翻倍，增强内部颜色比重

        # 需删除颜色
        all_color = np.array(range(color_num))
        del_color = []
        for i in range(color_num):
            # 内侧占比过低
            if color_in_rate[i] < INcolor_thresh:
                del_color.append(all_color[i])

        #颜色消除-------------------------------------------------------------------
        # 消除全图
        fin_data = 1-np.isin(img, del_color).astype(np.uint8)

        return fin_data*255

    def RESERVE_MAX_CONTOUR(self, img):  # 仅保留最大轮廓
        contours, _ = cv2.findContours(img.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        num_cpoint = [x.shape[0] for x in contours]
        cmax = max(num_cpoint)
        id_cmax = num_cpoint.index(cmax)
        cv_contour = contours[id_cmax]
        NEWimg = np.zeros(img.shape)
        cv2.fillPoly(NEWimg, [cv_contour], (255, 255, 255))
        return NEWimg

    def RESERVE_CONTOUR_AREA(self, img, threshold):  # 按面积大小保留轮廓
        contours, _ = cv2.findContours(img.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv_contour = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > threshold:
                cv_contour.append(contour)
        NEWimg = np.zeros(img.shape)
        cv2.fillPoly(NEWimg, cv_contour, (255, 255, 255))
        return NEWimg

    def CREATE_NEW_IMG(self, cut_num, bgOut, bgIn, org_img, INcolor_thresh):
        h = self.H/cut_num
        bgOut = bgOut.reshape(self.H, -1)
        bgIn = bgIn.reshape(self.H, -1)
        NEWimg = np.zeros(org_img.shape)
        for i in range(cut_num):
            s = int(i*h)
            e = int((i+1)*h)
            color_map, color_num = self.setCOLORmap(org_img[s:e, :])
            img = self.COLOR_DEL(bgOut[s:e, :], bgIn[s:e, :], color_map, color_num, INcolor_thresh)
            NEWimg[s:e, :] = img

        return NEWimg

    def CREATE_NEW_IMG2(self, cut_num, bgOut, bgIn, color_map, colors_use, INcolor_thresh):
        h = self.H/cut_num
        NEWimg = np.zeros(color_map.shape)
        bgOut = bgOut.reshape(self.H, -1)
        bgIn = bgIn.reshape(self.H, -1)
        color_map = color_map.reshape(self.H, -1)
        for i in range(cut_num):
            s = int(i*h)
            e = int((i+1)*h)
            img = self.COLOR_DEL2(bgOut[s:e,:], bgIn[s:e, :], color_map[s:e, :], colors_use, INcolor_thresh)
            NEWimg[s:e, :] = img

        return NEWimg

    def FIX_BG(self, BG):
        MAXcontour = self.RESERVE_MAX_CONTOUR(BG)
        NEWBG = BG/255 + MAXcontour/255
        NEWBG[NEWBG>0] = 255

        return NEWBG

    def CLEAR_RESULT(self, result):
        img = cv2.erode(result, self.conv_kernel3)  # 腐蚀
        MAXcontour = self.RESERVE_MAX_CONTOUR(img)
        NEWresult = cv2.dilate(MAXcontour, self.conv_kernel3)  # 膨胀

        return NEWresult

    def setgap(self, std):
        gap = std//3 + 1

        return int(gap)

    def setCOLORmap(self,img):  # 单一色图 简单颜色表颜色量化，颜色聚类太耗时
        std = np.std(img)
        gap = self.setgap(std)
        color_num = 256//gap+1
        color_map = img//gap
        return color_map, color_num

    def color_filter(self, bgOut, humanIn, org_img):
        # 第一次结果，图像按横向切割，分成1条
        first_cut_num = 1
        INcolor_thresh1 = 0.05
        _first_result = self.CREATE_NEW_IMG(first_cut_num, bgOut, humanIn, org_img, INcolor_thresh1)
        first_result = self.FIX_BG(_first_result) * (1-bgOut)  # 叠加最大轮廓填补空洞, 背景外侧直接置零

        # 第一次结果取最大边框可以作为第二次的背景
        second_BGin = first_result

        # 第二次结果，图像按横向切割，分成8条
        second_BGin = second_BGin/255
        second_BGout = 1-second_BGin

        second_cut_num = 8
        INcolor_thresh2 = 0.04
        _second_result = self.CREATE_NEW_IMG(second_cut_num, second_BGout, second_BGin, org_img, INcolor_thresh2)
        second_result = self.FIX_BG(_second_result)  # 叠加最大轮廓填补空洞

        final = self.RESERVE_MAX_CONTOUR(second_result)
        # final = final/255 * seg_img

        return final

    def silextract(self, org_img, knn_img, seg_img):
        # 统一图像尺寸，高为128
        width,height,org_img = self.uniformIMG(org_img)
        width,height,knn_img = self.uniformIMG(knn_img)
        width,height,seg_img = self.uniformIMG(seg_img)

        # 数据准备，背景分割外侧，绝对人体内侧
        bgOut, humanIn = self.prepareIMG(org_img,knn_img,seg_img)

        R_img = self.color_filter(bgOut, humanIn, org_img[:,:,0])/255
        G_img = self.color_filter(bgOut, humanIn, org_img[:,:,1])/255
        B_img = self.color_filter(bgOut, humanIn, org_img[:,:,2])/255
        avg_img= np.mean(org_img,axis=2).astype(np.uint8)
        A_img = self.color_filter(bgOut, humanIn, avg_img)/255

        fin_img = R_img + G_img + B_img + A_img
        fin_img[fin_img>1] = 255
        fin_img[fin_img==1] = 0

        cv2.imwrite(r'Siltest/B1.jpg', 255 * R_img)
        cv2.imwrite(r'Siltest/B2.jpg', 255 * G_img)
        cv2.imwrite(r'Siltest/B3.jpg', 255 * B_img)
        cv2.imwrite(r'Siltest/B4.jpg', 255 * A_img)
        cv2.imwrite(r'Siltest/A2.jpg', org_img[:, :, 0])
        cv2.imwrite(r'Siltest/A3.jpg', org_img[:, :, 1])
        cv2.imwrite(r'Siltest/A4.jpg', org_img[:, :, 2])
        cv2.imwrite(r'Siltest/A5.jpg', avg_img)
        cv2.imwrite(r'Siltest/colors_use_10.jpg', fin_img)

        return fin_img


if __name__ == "__main__":
    org_img = cv2.imread(r"Siltest/10ORG.jpg", 1)
    knn_img = cv2.imread(r'Siltest/10KNN.jpg', 0)
    seg_img = cv2.imread(r"Siltest/10SEG.jpg", 0)
    BGin = cv2.imread(r"Siltest/BGin.jpg", 0)
    BGout = cv2.imread(r"Siltest/BGout.jpg", 0)  # 绝对人体区域分割

    t1 = time.time()
    _ = SILEXTRACT(BGin, BGout)
    fin_img = _.silextract(org_img, knn_img, seg_img)
    t2 = time.time()
    print(t2-t1)

    cv2.imwrite(r'Siltest/colors_use_10.jpg', fin_img)
    imfill1 = cv2.imread(r"Siltest/colors_use_10.jpg", 1)
    width,height,org_img = _.uniformIMG(org_img)
    cv2.imwrite(r'Siltest/A1.jpg', imfill1/255*org_img)

