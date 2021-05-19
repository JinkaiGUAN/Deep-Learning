# -*- coding: UTF-8 -*-
"""
@Project : work_13_05 
@File    : single_day.py
@IDE     : PyCharm 
@Author  : Peter
@Date    : 14/05/2021 10:59 
@Brief   : This is a process calculating a single day all information.
"""
import os
import xlwings
import xlsxwriter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from datetime import datetime

config = {
    "font.family": 'FangSong',
    "font.size": 20,
    "mathtext.fontset": 'stix',
    "axes.titlesize": 24,
    'axes.unicode_minus': False  # 解决保存图像是负号'-'显示为方块的问题
}
rcParams.update(config)


class SingleDay(object):
    r"""All data structures in this class are default to be sorted.

    :arg
        @data_path (String): In this path, we suppose the user could input the
         absolute path of the excel file defined by our specific structure.
        @mode (String): Picking from IQ or EQ.

    :params
        @data (pd.Dataframe): A dataframe structure storing all the information
         assigned from the excel file
        @datetimes (list): [Pd.Timestamp('2019-03-12 00:00:00'), ...]. In this
         list, all timestamps have been store, including the repeated ones. For
         example, we have 1000 rows in the excel file, thus the length of this
         param is also 1000.
        @unique_datetimes (np.array): A set of date stored in this excel file.
         For example, we have 1000 valid rows in the excel file, but there are
         only 50 distinctive days, thus the length of this array is 50.
        @statistics_date_info (dict): key (pd.Timestap), value (list). The
         structure is {Timestamp('2019-03-12 00:00:00'): [0, 1, 2, 3 ...], ....},
         namely, The list of the indexes of this excel file corresponding to
         the specific date. Of course, the date cannot be repetitive.
        @orders_per_day (dict): The key is the order number, the value
         is the order quantity. For example,
         {114800199: 1, 114800203: 20, 114800205: 2,....}.
        @sorted_orders_list_per_day (list): the sorted list according
         to the order quantity in :param goods_order_per_day. The structure
         is as follows, [(114800269, 80),...]， where the sorted rule is
         from the maximum value to the minimum ones.
        @cum_orders_array_per_day (np.array): the cumulative summation of
         the order quantity according to the
         :param sorted_goods_order_list_per_day. However, zero is appended
         to the beginning of the array to form a new array.

    """

    def __init__(self, ROOT,  data_path, current_time, idx=0, mode='IQ'):
        self.ROOT = ROOT
        self.data_path = data_path
        self.current_time = current_time
        self.mode = mode
        self.data = pd.read_excel(data_path, sheet_name='FLC')
        self.log_folder_path = self._create_log_folder()
        self.datetimes, self.unique_datetimes = self._get_datetimes()
        self.statistics_date_info = self._statistics_datetimes()
        self.orders_per_day, self.sorted_orders_list_per_day, self.cum_orders_array_per_day = self.get_goods_info(
            idx)

    def _create_log_folder(self):
        r"""This helper function will create a log folder containing all the
        excel we need."""
        base_path = os.path.join(self.ROOT, 'results')
        log_name = 'EIQ {}'.format(
            datetime.strftime(self.current_time, '%Y-%m-%d-%H-%M'))
        log_path = os.path.join(base_path, log_name)
        if not os.path.exists(log_path):
            os.mkdir(log_path)

        return log_path

    def _get_datetimes(self):
        r"""Helper function to gain all dates stored in the file.

        :return
            @datetimes (list): [Pd.Timestamp('2019-03-12 00:00:00'), ...]"""

        def refactor_datetime_format(x):
            if isinstance(x, pd.Timestamp):
                return True
            else:
                raise Exception(
                    r"The date-time should be writen as the format of" \
                    " year/month/day hour:minute in row {}. Please use the date format in Excel!".format(
                        x))

        datetimes = list(filter(lambda x: refactor_datetime_format(x),
                                self.data.loc[:, '日期']))  # 清单中所有日期时间戳， 包含重复

        return datetimes, np.unique(datetimes)

    def _statistics_datetimes(self):
        r"""Helper function to append the index of the same day to a list which
        stored in a dictionary.

        :return
            @filter_data (dict): key (pd.Timestap), value (list)
        """

        filter_data = {}
        for i, date in enumerate(self.unique_datetimes):
            filter_data[date] = []
            for idx, datetime in enumerate(self.datetimes):
                if datetime.month == date.month and datetime.day == date.day:
                    filter_data[date].append(idx)

        return filter_data

    @staticmethod
    def _truncated_max_lim(sorted_value_list):
        ax_lim_max = int(np.max(sorted_value_list) * 1.1)
        if not (ax_lim_max % 10 == 0):
            ax_lim_max = (10 - ax_lim_max % 10) + ax_lim_max
        return ax_lim_max

    def get_goods_info(self, idx):
        r"""Get the goods data (order number -> order quantity) of a specific
        date.

        :arg
            @idx (int):

        :returns
            @goods_order_per_day (dict): The key is the order number, the value
             is the order quantity. For example,
             {114800199: 1, 114800203: 20, 114800205: 2,....}.
            @sorted_goods_order_list_per_day (list): the sorted list according
             to the order quantity in :param goods_order_per_day. The structure
             is as follows, [(114800269, 80),...]， where the sorted rule is
             from the maximum value to the minimum ones.
            @cum_order_array_per_day (np.array): the cumulative summation of
             the order quantity according to the
             :param sorted_goods_order_list_per_day. However, zero is appended
             to the beginning of the array to form a new array.
        """
        mask = self.statistics_date_info[
            self.unique_datetimes[idx]]  # the index list of the specific day
        data = self.data.iloc[mask, :]
        goods_bill = np.unique(data.loc[:, '货品编号'])

        # 现在可以做， 统计某一类货品的所有数量
        goods_order_per_day = {}  # storing the outbound item number of the certain goods

        for goods_name in goods_bill:
            goods_order_per_day[goods_name] = 0
            for idx, goods_num in zip(mask, data.loc[:, '货品编号']):
                if goods_num == goods_name:
                    goods_order_per_day[goods_name] += data.loc[idx, '数量']

        sorted_goods_order_list_per_day = sorted(goods_order_per_day.items(),
                                                 key=lambda d: d[1],
                                                 reverse=True)  # [(114800269, 80),...]
        cum_order_list_per_day = np.cumsum(
            [i[1] for i in sorted_goods_order_list_per_day])
        cum_order_array_per_day = np.append(0, cum_order_list_per_day)

        return goods_order_per_day, sorted_goods_order_list_per_day, cum_order_array_per_day

    def plot_plato(self, sorted_orders_list, cum_orders_array):
        r"""This function can plot the Plato, so that salesman can do the plato
        analysis in EIQ analysis. """

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()  # mirror the x-axis
        ax1.bar([i for i in range(len(self.sorted_orders_list_per_day))],
                [i[1] for i in self.sorted_orders_list_per_day],
                color='#009966',
                label='单品出货量')
        ax2.plot([i for i in range(len(self.cum_orders_array_per_day))],
                 self.cum_orders_array_per_day, c='#FF0033', label='累计出货量')

        ax1.set_xlabel('产品类别')
        ax1.set_ylabel('出货箱数', color='k')
        ax2.set_ylabel('累计出货箱数', color='k')
        ax1.set_ylim(
            [0, SingleDay._truncated_max_lim(
                [i[1] for i in self.sorted_orders_list_per_day])])
        ax2.set_ylim(
            [0, SingleDay._truncated_max_lim(self.cum_orders_array_per_day)])

        fig.legend(loc=1, bbox_to_anchor=(2, 1), bbox_transform=ax1.transAxes)
        if self.mode == 'IQ':
            plt.title('产品出库的IQ分布')  # 每个单品的订货量
        elif self.mode == 'EQ':
            plt.title('每张订单的订货数量EQ分析')
        else:
            raise Exception("Please input valid mode, i.e. EQ/IQ.")

        plt.savefig('./results/plato.svg', dpi=1600, bbox_inches='tight')
        plt.show()

