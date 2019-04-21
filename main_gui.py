# -*- coding: utf-8 -*-

from MainWindows import *

import time
import sys

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from cv2 import *
from config import opt
from face_recognition import fr
from PIL import Image
import numpy as np


class MainWindowCore(QMainWindow, Ui_MainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)

        # 初始变量
        self.video_path = None
        self.video_type = None  # 0: offline  1: webcamera
        self.status = opt.STATUS_INIT  # 0: init 1:playing 2: pause
        self.video_label_size = None
        self.video_size = None
        self.video_show_size = None
        self.video_center_box = None
        self.fr_ready = False
        self.fr_input_img = None
        opt.display_mw_view = self.viewList  # 更新输出控件
        opt.display_mw_sys = self.infoList  # 更新输出控件
        # 线程设置
        self.timer = None
        self.fr_worker = None
        self.load_worker = None
        self.dot_worker = None

        # 视频初始参数设置
        self.playCapture = VideoCapture()
        # self.videoWriter = VideoWriter('*.mp4', VideoWriter_fourcc('M', 'J', 'P', 'G'), self.fps, size)
        # 欢迎信息
        opt.print_system_gui('欢迎进入系统！')

    # 初始视频参数 ==========================================================
    def set_video(self, video_path='', video_type=opt.VIDEO_TYPE_WEBCAMERA):
        self.reset()
        self.video_path = video_path
        self.video_type = video_type
        print('视频类型 : ', video_type)
        print('视频路径 : ', video_path)
        print('获取视频信息中...')
        self.set_fps_get_size()
        self.cal_video_center_box()
        self.chk_label_size_change()
        # 更改按钮状态
        self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.playButton.setEnabled(True)
        print('播放准备状态\n')

    def set_fps_get_size(self):
        if self.video_type is opt.VIDEO_TYPE_WEBCAMERA:
            self.playCapture = VideoCapture(0)
        else:
            self.playCapture.open(self.video_path)
        # 获得帧率
        fps = self.playCapture.get(CAP_PROP_FPS)
        print('视频流帧率：', fps)
        self.timer.set_fps(fps)
        self.fr_worker.set_fps(fps)
        # 获得尺寸
        size = (int(self.playCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self.playCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        print('视频流输入尺寸：', size)
        self.video_size = size
        self.playCapture.release()

    def cal_show_size(self):
        max_size = np.max([self.videoLabel.width(), self.videoLabel.height()])
        width, height = self.video_size
        ratio_w = width / max_size
        ratio_h = height / max_size
        ratio = np.max((ratio_w, ratio_h))
        width, height = int(width / ratio), int(height / ratio)
        self.video_show_size = (width, height)
        print('视频流显示尺寸：', self.video_show_size)

    def cal_video_center_box(self):
        width, height = self.video_size
        tar_size = opt.video_show_box_size
        half_size = tar_size / 2
        y, x = height / 2, width / 2
        top = int(y - half_size)
        left = int(x - half_size)
        bottom = int(y + half_size)
        right = int(x + half_size)
        self.video_center_box = (top, left, bottom, right)

    def chk_label_size_change(self):
        now_video_label_size = (self.videoLabel.width(), self.videoLabel.height())
        if now_video_label_size != self.video_label_size:
            print('显示区域大小改变，更改显示尺寸')
            self.video_label_size = now_video_label_size
            self.cal_show_size()

    # 状态控制 ============================================================
    def reset(self):
        print('\n清空视频缓存信息 状态初始化')
        # 注意先后暂停线程
        if self.fr_worker is not None:
            self.fr_worker.stop()
        if self.timer is not None:
            self.timer.stop()
        self.playCapture.release()
        # 重置为初始状态
        if self.status is not opt.STATUS_INIT:
            self.status = opt.STATUS_INIT
        if self.fr_ready:
            # 线程设置
            self.timer = VideoTimer()
            self.timer.show_video_signal.connect(self.show_video_images)
            self.fr_worker = FRWorker()
            self.fr_worker.do_fr_Signal.connect(self.sample_and_fr)

    def chk_video_type(self):
        if self.fr_ready:
            if self.offlineButton.isChecked() and self.video_type != opt.VIDEO_TYPE_OFFLINE:
                self.switch_offline()
            if self.webcamButton.isChecked() and self.video_type != opt.VIDEO_TYPE_WEBCAMERA:
                self.switch_webcam()
            if self.onlineButton.isChecked() and self.video_type != opt.VIDEO_TYPE_ONLINE:
                self.switch_online()
        else:
            print('未加载网络数据')
            opt.print_system_gui('未加载网络数据！请先加载网络数据')

    def switch_offline(self):
        self.set_video(video_path='./video/f40.mp4', video_type=opt.VIDEO_TYPE_OFFLINE)
        opt.print_system_gui('视频模式：offline')

    def switch_webcam(self):
        self.set_video(video_path='', video_type=opt.VIDEO_TYPE_WEBCAMERA)
        opt.print_system_gui('视频模式：webcam')

    def switch_online(self):
        self.set_video(video_path='rtmp://58.200.131.2:1935/livetv/hunantv', video_type=opt.VIDEO_TYPE_ONLINE)
        opt.print_system_gui('视频模式：online')

    def switch_video(self):
        if self.video_path == '' and self.video_type is opt.VIDEO_TYPE_OFFLINE:
            print('路径设置错误！')
            return
        if self.status is opt.STATUS_INIT:
            self.chk_video_type()
            self.play()
        elif self.status is opt.STATUS_PLAYING:
            if self.video_type is opt.VIDEO_TYPE_OFFLINE:
                self.pause()
            if self.video_type is opt.VIDEO_TYPE_WEBCAMERA:
                self.pause()
            if self.video_type is opt.VIDEO_TYPE_ONLINE:
                self.pause()
        elif self.status is opt.STATUS_PAUSE:
            self.chk_label_size_change()
            self.re_play()
        elif self.status is opt.STATUS_STOP:
            self.play()

    def play(self):
        if not self.playCapture.isOpened():
            if self.video_type is opt.VIDEO_TYPE_WEBCAMERA:
                self.playCapture = VideoCapture(0)
            else:
                print('获取在线流媒体中...')
                self.playCapture.open(self.video_path)
        # 启动线程
        self.timer.start()
        # 改为播放状态
        self.status = opt.STATUS_PLAYING
        self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        print('开始显示')
        opt.print_system_gui('开始监测')
        # 延后启动线程
        time.sleep(0.01)
        self.fr_worker.start()

    def stop(self):
        if self.playCapture.isOpened():
            # 注意先后暂停线程
            if self.fr_worker is not None:
                self.fr_worker.stop()
            if self.timer is not None:
                self.timer.stop()
            self.playCapture.release()
            # 改为停止状态
            self.status = opt.STATUS_STOP
            self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
            print('停止显示')
            opt.print_system_gui('停止监测')

    def pause(self):
        if self.playCapture.isOpened():
            # 停止线程
            self.timer.stop()
            self.fr_worker.stop()
            # 改为暂停状态
            self.status = opt.STATUS_PAUSE
            self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
            print('暂停显示')
            opt.print_system_gui('暂停监测')

    def re_play(self):
        if self.playCapture.isOpened():
            # 启动线程
            self.timer.start()
            # 改为播放状态
            self.status = opt.STATUS_PLAYING
            self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
            print('恢复显示')
            opt.print_system_gui('恢复监测')
            # 延后启动线程
            time.sleep(0.01)
            self.fr_worker.start()

    # 功能函数 ============================================================
    def load_net_data_start(self):

        self.load_worker = LoadWorker()
        self.load_worker.load_signal.connect(fr.login_main)

        self.dot_worker = DotWorker()
        self.dot_worker.update_print_signal.connect(opt.add_dot_print_system_gui)
        self.dot_worker.finish_signal.connect(self.load_net_data_done)

        self.load_worker.start()
        self.dot_worker.start()

    def load_net_data_done(self):
        self.load_worker.stop()
        self.dot_worker.stop()

        # 改变标志和状态
        self.fr_ready = True
        self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.playButton.setEnabled(True)

    def show_video_images(self):
        if self.playCapture.isOpened():
            success, frame = self.playCapture.read()
            if success:
                width, height = self.video_size
                top, left, bottom, right = self.video_center_box
                if frame.ndim == 3:
                    rgb = cvtColor(frame, COLOR_BGR2RGB)
                elif frame.ndim == 2:
                    rgb = cvtColor(frame, COLOR_GRAY2BGR)
                # 存储变量
                ori_rgb = rgb.copy()
                self.fr_input_img = ori_rgb
                # 画矩形
                rgb = cv2.rectangle(rgb, (left, top), (right, bottom), (255, 0, 0), 2)
                # 送给显示控件
                temp_image = QImage(rgb.flatten(), width, height, QImage.Format_RGB888)
                temp_pixmap = QPixmap.fromImage(temp_image).scaled(self.video_show_size[0],
                                                                   self.video_show_size[1])
                self.videoLabel.setPixmap(temp_pixmap)
            else:
                print("read failed, no frame data")
                success, frame = self.playCapture.read()
                if not success and self.video_type is opt.VIDEO_TYPE_OFFLINE:
                    print("play finished")  # 判断本地文件播放完毕
                    self.reset()
                    self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaStop))
                return
        else:
            print("open file or capturing device error, init again")
            self.reset()

    def sample_and_fr(self):
        img = self.fr_input_img
        if img is not None:
            time_str = time.strftime('%m月%d日-%H时%M分%S秒')
            top, left, bottom, right = self.video_center_box
            crop_img = img[top:bottom, left:right]
            test_img = Image.fromarray(crop_img)
            # save 裁剪图
            print('当前图片采集', np.shape(crop_img))
            save_img = cvtColor(crop_img, COLOR_RGB2BGR)
            cv2.imwrite('./sample/' + time_str + '_检测图像.bmp', save_img)
            # 为列表准备的图片
            cv2.imwrite('./sample/now.bmp', save_img)
            # 送给检测器
            fr.who_is_it(test_img)
        else:
            print('no img')


# 数据加载线程类
class LoadWorker(QThread):
    load_signal = pyqtSignal(str)

    def __init__(self, frequent=1):
        QThread.__init__(self)
        self.stopped = False
        self.frequent = frequent
        self.mutex = QMutex()

    def run(self):
        with QMutexLocker(self.mutex):
            self.stopped = False
        self.load_signal.emit("1")
        while True:
            if self.stopped:
                return
            time.sleep(1 / self.frequent)

    def stop(self):
        with QMutexLocker(self.mutex):
            self.stopped = True

    def is_stopped(self):
        with QMutexLocker(self.mutex):
            return self.stopped

    def set_fps(self, fps):
        self.frequent = fps


# 显示更新线程类
class DotWorker(QThread):
    update_print_signal = pyqtSignal(str)
    finish_signal = pyqtSignal(str)

    def __init__(self, frequent=2):
        QThread.__init__(self)
        self.stopped = False
        self.frequent = frequent
        self.mutex = QMutex()

    def run(self):
        with QMutexLocker(self.mutex):
            self.stopped = False
        while True:
            if self.stopped:
                return
            if not opt.is_net_load_done:
                self.update_print_signal.emit("1")
            else:
                self.finish_signal.emit("1")
            time.sleep(1 / self.frequent)

    def stop(self):
        with QMutexLocker(self.mutex):
            self.stopped = True

    def is_stopped(self):
        with QMutexLocker(self.mutex):
            return self.stopped

    def set_fps(self, fps):
        self.frequent = fps


# 视频播放线程类
class VideoTimer(QThread):
    show_video_signal = pyqtSignal(str)

    def __init__(self, frequent=20):
        QThread.__init__(self)
        self.stopped = False
        self.frequent = frequent
        self.mutex = QMutex()

    def run(self):
        with QMutexLocker(self.mutex):
            self.stopped = False
        while True:
            if self.stopped:
                return
            self.show_video_signal.emit("1")
            time.sleep(1 * 3.0 / self.frequent)

    def stop(self):
        with QMutexLocker(self.mutex):
            self.stopped = True

    def is_stopped(self):
        with QMutexLocker(self.mutex):
            return self.stopped

    def set_fps(self, fps):
        self.frequent = fps


# 人脸识别线程类
class FRWorker(QThread):
    do_fr_Signal = pyqtSignal(str)

    def __init__(self, parent=None, frequent=20):
        super(FRWorker, self).__init__(parent)
        self.stopped = False
        self.frequent = frequent
        self.mutex = QMutex()

    def run(self):
        with QMutexLocker(self.mutex):
            self.stopped = False
        while True:
            if self.stopped:
                return
            self.do_fr_Signal.emit("1")
            time.sleep(1 * 120 / self.frequent)

    def stop(self):
        with QMutexLocker(self.mutex):
            self.stopped = True

    def is_stopped(self):
        with QMutexLocker(self.mutex):
            return self.stopped

    def set_fps(self, fps):
        self.frequent = fps


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mw = MainWindowCore()
    mw.show()
    sys.exit(app.exec_())
