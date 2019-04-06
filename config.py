from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import time


class Config:
    # 规定代号
    VIDEO_TYPE_OFFLINE = 0
    VIDEO_TYPE_WEBCAMERA = 1
    VIDEO_TYPE_ONLINE = 3

    STATUS_INIT = 0
    STATUS_PLAYING = 1
    STATUS_PAUSE = 2
    STATUS_STOP = 3

    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

    # 配置程序所有的设置数据
    def __init__(self):
        #  初始化程序的设置参数
        self.net_data_path = './model/0313133934_0.958_0.1149_resnet18_net_params.pkl'
        self.net_best_margin = 8.60
        self.net_input_img_size = 200
        self.database_dir = './database/'
        self.video_show_box_size = 200

        self.is_net_load_done = False

        #  初始化程序的设置变量
        self.display_mw_view = None
        self.display_mw_view_index = 0
        self.display_mw_sys = None
        self.display_mw_sys_index = 0

    # 主窗口GUI上显示的viewlist滚动行
    def print_view_gui(self, string):
        item = QListWidgetItem(QIcon('./sample/now.bmp'), string)
        self.display_mw_view.insertItem(self.display_mw_view_index, item)
        self.display_mw_view.setCurrentRow(self.display_mw_view_index)
        self.display_mw_view.repaint()
        self.display_mw_view.show()
        self.display_mw_view_index += 1

    # 主窗口GUI上显示的systemlist滚动行
    def print_system_gui(self, string):
        time_str = time.strftime('%m月%d日-%H时%M分%S秒')
        string = time_str + ' : ' + string
        self.display_mw_sys.insertItem(self.display_mw_sys_index, string)
        self.display_mw_sys.setCurrentRow(self.display_mw_sys_index)
        self.display_mw_sys.repaint()
        self.display_mw_sys.show()
        self.display_mw_sys_index += 1

    # 主窗口GUI上显示的systemlist滚动行  更新显示
    def add_dot_print_system_gui(self):
        string = self.display_mw_sys.currentItem().text() + '.'
        self.display_mw_sys.takeItem(self.display_mw_sys_index-1)
        self.display_mw_sys.insertItem(self.display_mw_sys_index-1, string)
        self.display_mw_sys.setCurrentRow(self.display_mw_sys_index-1)
        self.display_mw_sys.repaint()
        self.display_mw_sys.show()


opt = Config()
