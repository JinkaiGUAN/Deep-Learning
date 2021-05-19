# -*- coding: UTF-8 -*-
"""
@Project : 国药FLC冷库出库分析2019.9.18.xlsx 
@File    : plots.py
@IDE     : PyCharm 
@Author  : Peter
@Date    : 18/05/2021 16:00 
@Brief   : 
"""
import os
import time
import numpy as np
import math
from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib import dates
from matplotlib import rcParams

from src.single_day import SingleDay
from src.full_year import FullYear
from src.utils import time_count

config = {
    "font.family": 'FangSong',
    "font.size": 20,
    "mathtext.fontset": 'stix',
    "axes.titlesize": 24,
    'axes.unicode_minus': False  # 解决保存图像是负号'-'显示为方块的问题
}
rcParams.update(config)


class PlotGraphs(SingleDay):
    r"""Full year statistics, some data will be stored in the excel. Remember
    that, to better unite the format, in this code, we use Mandarin instead of
    English to comment some key properties.

    :params
        @ROOT (String): The output folder path, where all caches store.
        @data_path (String): The path where the original data stores.
        @current_time (datetime.datetime): Date and time once you initialize
         such instance.

    :arguments:
        @log_folder_path (string): The path storing all the excels, i.e., cashes
         folder path.
    """

    def __init__(self, ROOT, datapath, current_time):
        super(PlotGraphs, self).__init__(ROOT, datapath, current_time)

    def plot_outbound_items(self):
        outbound_items_date = self.data.loc[:, ['日期', '数量']]
        # self.unique_datetimes 所有日期
        outbound_items = []
        for date in self.unique_datetimes:
            item_num = np.sum(outbound_items_date[outbound_items_date['日期'] == date].loc[:, '数量'])
            outbound_items.append(item_num)

        sorted_items = sorted(outbound_items, reverse=True)
        eighty_line = sorted_items[math.ceil(len(sorted_items) * 0.2)-1]

        fig = plt.figure(figsize=(16, 9))
        plt.plot(self.unique_datetimes, outbound_items)
        plt.axhline(eighty_line)
        plt.xlabel('日期')
        plt.ylabel('出库量')
        plt.title('出库量与日期分布对比图')
        fig_path = os.path.join(self.log_folder_path, '出库量与日期分布对比图.svg')
        plt.savefig(fig_path, dpi=1600, bbox_inches='tight')


if __name__ == '__main__':
    time_start = time.time()
    root = os.path.abspath(os.path.join(os.getcwd(), ".."))  # os.getcwd()
    current_time = datetime.now()
    plot_graph = PlotGraphs(root,'../国药FLC冷库出库分析2019.9.18.xlsx',current_time)
    plot_graph.plot_outbound_items()

    time_end = time.time()
    time_count(time_start, time_end)



