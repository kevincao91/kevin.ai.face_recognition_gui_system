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


# 登录窗口类
class LoginWindow(QWidget):

    def __init__(self, mw):
        QWidget.__init__(self)
        # 初始变量
        self.login_size = opt.login_size
        # 初始控件
        self.print_list = QListWidget()
        self.login_button = QPushButton('登录系统')
        opt.display_lg_info = self.print_list
        self.main_window = mw
        # 线程设置
        # self.login_worker = LoginWorker()
        # self.login_worker.load_signal.connect(self.login_do)
        # self.login_worker.load_finish_signal.connect(self.login_finish)
        # 初始界面
        self.init_gui()

    def init_gui(self):
        # 组件设置
        self.login_button.clicked.connect(self.login_fun)
        # 栅格布局
        grid = QGridLayout()
        # 布局组件
        grid.addWidget(self.print_list, 0, 0)
        grid.addWidget(self.login_button, 1, 0)
        self.setLayout(grid)
        # 窗口设置
        self.resize(self.login_size, self.login_size)
        self.place_scr_center()
        self.setWindowTitle('人脸监测系统')
        # 窗口显示
        self.show()

    def place_scr_center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    # 功能函数 ====================================
    def login_fun(self):
        fr.login_main()
        self.main_window.show()
        opt.print_system_gui('登录系统')
        self.close()

    def login_start(self):
        self.login_worker.start()

    def login_do(self):
        fr.login_main()
        self.login_worker.load_finish_signal.emit('1')

    def login_finish(self):
        self.main_window.show()
        opt.print_system_gui('登录系统')
        self.close()

    def show_dot(self, dot_num):
        opt.print_gui(str(dot_num))
        pass


# 主窗口类
class MainWindow(QWidget):
    def __init__(self, video_path='', video_type=opt.VIDEO_TYPE_WEBCAMERA):
        QWidget.__init__(self)

        # 初始变量
        self.video_path = video_path
        self.video_type = video_type  # 0: offline  1: webcamera
        self.status = opt.STATUS_INIT  # 0: init 1:playing 2: pause
        self.video_size = None
        self.video_show_size = None
        self.video_center_box = None
        self.video_show_max_size = opt.video_show_max_size
        self.fr_input_img = None

        # 初始控件
        self.title_label = QLabel('Title Info')
        self.playLabel = QLabel('开始监测')
        self.video_label = QLabel('主监视摄像内容')
        self.list_label = QLabel('监测来访人员列表')
        self.pictureLabel = QLabel()
        self.playButton = QPushButton()
        self.view_list = QListWidget()
        self.sys_info_list = QListWidget()
        opt.display_mw_view = self.view_list  # 更新输出控件
        opt.display_mw_sys = self.sys_info_list  # 更新输出控件

        # 初始界面
        self.init_gui()

        # 线程设置
        self.timer = VideoTimer()
        self.timer.show_video_signal.connect(self.show_video_images)
        self.fr_worker = FRWorker()
        self.fr_worker.do_fr_Signal.connect(self.show_people_name)

        # 视频初始参数设置
        self.playCapture = VideoCapture()
        # self.videoWriter = VideoWriter('*.mp4', VideoWriter_fourcc('M', 'J', 'P', 'G'), self.fps, size)

    def init_gui(self):
        # 组件设置
        init_image = QPixmap("resource/cat.jpeg").scaled(self.video_show_max_size, self.video_show_max_size)
        self.pictureLabel.setPixmap(init_image)
        self.playButton.setEnabled(True)
        self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.playButton.clicked.connect(self.switch_video)
        # 栅格布局
        grid = QGridLayout()
        # 布局组件
        grid.addWidget(self.title_label, 0, 0)
        grid.addWidget(self.playLabel, 0, 3)
        grid.addWidget(self.playButton, 0, 4)
        grid.addWidget(self.video_label, 1, 0)
        grid.addWidget(self.list_label, 1, 3)
        grid.addWidget(self.pictureLabel, 2, 0, 3, 2)
        grid.addWidget(self.view_list, 2, 3, 3, 2)
        grid.addWidget(self.sys_info_list, 5, 0, 2, 5)
        self.setLayout(grid)
        # 窗口设置
        if opt.full_window:
            self.full_screen_size()
        else:
            self.resize(opt.main_window_size[0], opt.main_window_size[1])
        self.place_scr_center()
        self.setWindowTitle('人脸监测系统')

    def full_screen_size(self):
        qr = self.frameGeometry()
        w = QDesktopWidget().availableGeometry().width()
        h = QDesktopWidget().availableGeometry().height()
        self.resize(w, h)

    def place_scr_center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    # 初始参数 ============================================================
    def set_video(self, video_path='', video_type=opt.VIDEO_TYPE_WEBCAMERA):
        self.reset()
        self.video_path = video_path
        self.video_type = video_type
        self.set_fps_get_size()
        self.cal_show_size()
        self.cal_video_center_box()

    def set_fps_get_size(self):
        if self.video_type is opt.VIDEO_TYPE_WEBCAMERA:
            self.playCapture = VideoCapture(0)
        else:
            self.playCapture.open(self.video_url)
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
        max_size = self.video_show_max_size
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

    # 状态控制 ============================================================
    def reset(self):
        self.timer.stop()
        self.playCapture.release()
        # 重置为初始状态
        self.status = opt.STATUS_INIT
        self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))

    def switch_video(self):
        if self.video_path == '' and self.video_type is opt.VIDEO_TYPE_OFFLINE:
            print('路径设置错误！')
            return
        if self.status is opt.STATUS_INIT:
            self.play()
        elif self.status is opt.STATUS_PLAYING:
            if self.video_type is opt.VIDEO_TYPE_OFFLINE:
                self.pause()
            if self.video_type is opt.VIDEO_TYPE_WEBCAMERA:
                self.stop()
        elif self.status is opt.STATUS_PAUSE:
            self.re_play()
        elif self.status is opt.STATUS_STOP:
            self.play()

    def play(self):
        if not self.playCapture.isOpened():
            if self.video_type is opt.VIDEO_TYPE_WEBCAMERA:
                self.playCapture = VideoCapture(0)
            else:
                self.playCapture.open(self.video_path)
        # 启动线程
        self.timer.start()
        time.sleep(0.2)
        self.fr_worker.start()
        # 改为播放状态
        self.status = opt.STATUS_PLAYING
        self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))

    def stop(self):
        if self.playCapture.isOpened():
            # 停止线程
            self.timer.stop()
            self.fr_worker.stop()
            self.playCapture.release()
            # 改为停止状态
            self.status = opt.STATUS_STOP
            self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))

    def pause(self):
        if self.playCapture.isOpened():
            # 停止线程
            self.timer.stop()
            self.fr_worker.stop()
            # 改为暂停状态
            self.status = opt.STATUS_PAUSE
            self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))

    def re_play(self):
        if self.playCapture.isOpened():
            # 启动线程
            self.timer.start()
            time.sleep(0.2)
            self.fr_worker.start()
            # 改为播放状态
            self.status = opt.STATUS_PLAYING
            self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))

    # 功能函数 ============================================================
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
                self.pictureLabel.setPixmap(temp_pixmap)
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

    def show_people_name(self):
        img = self.fr_input_img
        if img is not None:
            time_str = time.strftime('%m月%d日-%H时%M分%S秒')
            top, left, bottom, right = self.video_center_box
            crop_img = img[top:bottom, left:right]
            test_img = Image.fromarray(crop_img)
            # save 裁剪图
            print('当前图片采集', np.shape(crop_img))
            save_img = cvtColor(crop_img, COLOR_RGB2BGR)
            cv2.imwrite('sample/' + time_str + '_检测图像.bmp', save_img)
            # 送给检测器
            fr.who_is_it(test_img)
        else:
            print('no img')


# 参数加载线程类
class LoginWorker(QThread):
    load_signal = pyqtSignal(str)
    load_finish_signal = pyqtSignal(str)

    def __init__(self):
        QThread.__init__(self)
        self.stopped = False

    def run(self):
        self.load_signal.emit('1')

    def stop(self):
        self.stopped = True

    def is_stopped(self):
        return self.stopped


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
            time.sleep(1 * 3 / self.frequent)

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


if __name__ == "__main__":
    app = QApplication(sys.argv)

    mw = MainWindow()
    mw.set_video(video_path='', video_type=opt.VIDEO_TYPE_WEBCAMERA)

    lw = LoginWindow(mw)

    sys.exit(app.exec_())
