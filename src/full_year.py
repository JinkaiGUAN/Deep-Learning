# -*- coding: UTF-8 -*-
"""
@Project : work_13_05 
@File    : single_day.py
@IDE     : PyCharm 
@Author  : Peter
@Date    : 14/05/2021 10:59 
@Brief   : This is a process calculating a single day all information.
"""
__version__ = '0.0.2'


import os
import time
# import xlwings
# import xlsxwriter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from datetime import datetime

from single_day import SingleDay

from utils import time_count

config = {
    "font.family": 'FangSong',
    "font.size": 20,
    "mathtext.fontset": 'stix',
    "axes.titlesize": 24,
    'axes.unicode_minus': False  # 解决保存图像是负号'-'显示为方块的问题
}
rcParams.update(config)


class FullYear(SingleDay):
    r"""Full year statistics, some data will be stored in the excel. Remember
    that, to better unite the format, in this code, we use Mandarin instead of
    English to comment some key properties.

    :params
        @ROOT (String): The output folder path, where all caches store.
        @data_path (String): The path where the original data stores.
        @split_num (Int): We use such :param to split the order_row into several
         ranges, the certain number is :param split_num. This is related to the
         function `top_twenty_order`. Since the order_row isn't large in this
         example, we can use default value 6.
        @current_time (datetime.datetime): Date and time once you initialize
         such instance.

    :arguments:
        @log_folder_path (string): The path storing all the excels, i.e., cashes
         folder path.
    """

    def __init__(self, ROOT, data_path, current_time, split_num=6, idx=0, mode='IQ'):
        super(FullYear, self).__init__(ROOT=ROOT, data_path=data_path, current_time=current_time, idx=idx, mode=mode)
        # self.current_time = datetime(2021, 5, 18, 15, 14)  #datetime.now()
        self.current_time = datetime.now()
        self.ROOT = ROOT
        # self.log_folder_path = self._create_log_folder()
        self.split_num = split_num

    def _get_goods_bill(self):
        r"""This helper function will give all the goods ID (货品编号) for full
        year."""
        return np.unique(self.data.loc[:, '货品编号'])

    def _get_picking_order_bill(self):
        r"""This helper function will give all the picking order ID (拣货单号)
        for full year."""
        return np.unique(self.data.loc[:, '拣货单号'])

    def _save_to_excel(self, dataframe, sheet_name):
        r"""Helper function to save the dataframe to the excel file with a
        specific sheet name."""
        excel_path = os.path.join(self.log_folder_path, '{}.xlsx'.format(sheet_name))
        # writer = pd.ExcelWriter(excel_path)
        # dataframe.to_excel(writer, sheet_name=sheet_name, encoding='utf-8')
        # writer.save()
        index_mark = False
        if sheet_name == '动碰原表':
            index_mark = True
        dataframe.to_excel(excel_path, encoding='utf-8', index=index_mark, sheet_name=sheet_name)

    def _extract_from_excel_to_dataframe(self, sheet_name):
        excel_path = os.path.join(self.log_folder_path,
                                  '{}.xlsx'.format(sheet_name))
        return pd.read_excel(excel_path, sheet_name=sheet_name)

    def merge_excel(self):

        file_paths = []
        names = []
        for excel in os.listdir(self.log_folder_path):
            if excel.endswith('.xlsx'):
                names.append(excel.split('.')[0])
                file_paths.append(os.path.join(self.log_folder_path, excel))

        # 循环读取并，生成多列
        final_excel_path = os.path.join(self.log_folder_path, 'Final EIQ.xlsx')
        writer = pd.ExcelWriter(final_excel_path)

        index_mark = False

        for idx, excel in enumerate(file_paths):
            df = pd.read_excel(excel)
            if names[idx] == '动碰原表':
                index_mark = True
            df.to_excel(writer, sheet_name=names[idx], index=index_mark)

        writer.save()

    @staticmethod
    def _self_sort(value_list, reverse_mark=True):
        r"""Convert a list into a sorted list, but return the position
        information. For example, if the number 123 is the 35th largest in a
        list, the return list will have 35 at the corresponding place."""

        position_list = [i for i in range(1, len(value_list) + 1)]
        sorted_value = sorted(value_list, reverse=reverse_mark)

        position_info = []
        for num in value_list:
            for pos_mark, num_mark in zip(position_list, sorted_value):
                if int(num) == int(num_mark):
                    position_info.append(pos_mark)
                    break
        return position_info

    def flc_row_items_statistics(self):
        r"""Make the ‘FCL行件’ sheet."""
        dates = self.unique_datetimes  # 日期
        dates_str = [x.strftime('%Y/%m/%d') for x in dates]
        # 订单行数
        # order_rows = {timestamp: len(row_list)  for (timestamp, row_list) in self.statistics_date_info}
        order_rows = [len(row_list) for _, row_list in
                      self.statistics_date_info.items()]
        # 件数, SKU数量
        order_quantities = []  # 件数
        sku_nums = []  # SKU数量
        for idx in range(len(dates)):
            orders_per_day, _, order_quantity = super(FullYear,
                                                      self).get_goods_info(idx)
            order_quantities.append(order_quantity[-1])
            sku_nums.append(len(orders_per_day))

        r"""We need to pay attention to when the order row is zero."""
        # 行件比
        ratio_order_rows2order_quantities = [
            round(float(order_quantity) / float(order_row + 1e-6), 1) for
            order_row, order_quantity in zip(order_rows, order_quantities)]
        # 日均动碰
        daily_average_touches = [
            round(float(order_row) / float(sku_num + 1e-6), 1) for
            order_row, sku_num in zip(order_rows, sku_nums)]
        # 重复率
        repetition_rates = np.round(
            (1 - np.array(sku_nums, dtype=np.float64) / (np.array(order_rows) + 1e-6)) * 100, 2)
        # 序号
        serial_number = FullYear._self_sort(order_rows, reverse_mark=True)

        # save data, 日期出现问题
        row_item = {'序号': serial_number, '日期': dates_str, '订单行数': order_rows,
                    '件数': order_quantities,
                    '行件比': ratio_order_rows2order_quantities, 'SKU数量': sku_nums,
                    '日均动碰': daily_average_touches, '重复率/%': repetition_rates}
        row_item_dataframe = pd.DataFrame(row_item)
        self._save_to_excel(row_item_dataframe, sheet_name='FLC行件')

    def flc_IQ(self):
        r"""FLC 冷库IQ, The structure please refer to the method get_goods_info
        in super class. In this method, we have to achieve the following things,
        i.e. 序号， 货品编号， 出库量， SKU占比， SKU累计占比， 出库量占比， 出库量累计占比,
        SKU总数， 出库总量.

        在此过程中， 我们按照出库量从高到底排序。
        """

        # 货品编号
        goods_bill = self._get_goods_bill()
        # 出库量, 即该商品的出库数量
        goods_order = {}  # storing the outbound item number of the certain goods

        for goods_ID in goods_bill:
            goods_order[goods_ID] = 0
            for idx, goods_num in enumerate(self.data.loc[:, '货品编号']):
                if goods_num == goods_ID:
                    goods_order[goods_ID] += self.data.loc[idx, '数量']

        outbound_items_num_list = [goods_num[1] for goods_num in
                                   goods_order.items()]  # 出库量

        # 对出库量进行排序， 后续输入按照排序后的出库量输入
        sorted_idx = np.argsort(-np.array(outbound_items_num_list))
        sorted_outbound_items_num_list = sorted(outbound_items_num_list, reverse=True)
        goods_bill = [goods_bill[sorted_idx[i]] for i in range(len(goods_bill))]

        # 序号
        serial_num = FullYear._self_sort(sorted_outbound_items_num_list)

        # SKU占比， actually， all the ratio is the same
        sku_ratio = np.round((1 / np.array(
            [float(len(goods_bill)) for _ in range(len(goods_bill))])) * 100, 2)
        # SKU累计占比
        sku_ratio_cum = np.cumsum(sku_ratio)
        # 出库总量
        total_items = sum(sorted_outbound_items_num_list)

        # 出库量占比
        outbound_items_ratio = (np.array(sorted_outbound_items_num_list,
                                         dtype=np.float64) / total_items) * 100
        # 出库量累计占比
        """Some problem here!"""
        outbound_items_ratio_sum = np.cumsum(outbound_items_ratio)

        # save data,
        IQ = {'序号': serial_num, '货品编号': goods_bill,
              '出库量': sorted_outbound_items_num_list,
              'SKU占比/%': sku_ratio,
              'SKU累计占比/%': sku_ratio_cum, '出库量占比/%': np.round(outbound_items_ratio, decimals=2),
              '出库量累计占比/%': np.round(outbound_items_ratio_sum, decimals=2)}
        IQ_dataframe = pd.DataFrame(IQ)
        IQ_dataframe['SKU总数'] = len(goods_bill)
        IQ_dataframe['出库总量'] = total_items

        self._save_to_excel(IQ_dataframe, sheet_name='FLC冷库IQ')

    def flc_EQ(self):
        r"""FLC 冷库EQ.

        需要字段: 序号， 拣货单号， 出库量， 订单行数， 订单占比， 订单累计占比， 出库量占比， 出库量累计占比，
        订单总数， 出库总量， 订单行总数， 平均订单行数， 平均行件数， 平均订单出库量
        """
        # 拣货单号
        picking_order_nums = self._get_picking_order_bill()
        # 出库量
        out_stock_per_picking_ID = {}  # 每个订单号的出库量
        order_lines_per_picking_ID = {}  # 每个订单的总行数
        for picking_ID in picking_order_nums:
            out_stock_per_picking_ID[picking_ID] = 0
            order_lines_per_picking_ID[picking_ID] = 0
            for idx, picking_id_by_row in enumerate(self.data.loc[:, '拣货单号']):
                if picking_id_by_row == picking_ID:
                    out_stock_per_picking_ID[picking_ID] += self.data.loc[idx, '数量']
                    order_lines_per_picking_ID[picking_ID] += 1

        outbound_items_num_list = [num[1] for num in out_stock_per_picking_ID.items()]  # 出库量

        sorted_idx = np.argsort(-np.array(outbound_items_num_list)) # 由高到低
        # 排序后的出库量
        sorted_outbound_items_num_list = sorted(outbound_items_num_list, reverse=True)
        # 排序后的拣货单号
        picking_order_nums = [picking_order_nums[sorted_idx[i]] for i in range(len(sorted_idx))]
        # 序号
        serial_num_list = FullYear._self_sort(sorted_outbound_items_num_list)

        # 订单行数, 即拣货订单对应FLC表格中行数
        order_lines_list = [num[1] for num in order_lines_per_picking_ID.items()]
        # 排序后的订单行数
        sorted_order_lines_list = [order_lines_list[sorted_idx[i]] for i in range(len(sorted_idx))]

        # 订单总数
        # total_order_num = np.sum(picking_order_nums, dtype=np.float64)
        total_order_num = len(picking_order_nums)
        # 订单占比
        order_ratio_list = np.ones(
            shape=(len(picking_order_nums)), dtype=np.float64) / total_order_num * 100

        # 订单累计占比
        cum_order_ratio_list = np.cumsum(order_ratio_list)
        # 出库总量
        total_outbound_num = np.sum(sorted_outbound_items_num_list, dtype=np.float64)
        # 出库量占比
        outbound_ratio_list = np.array(
            sorted_outbound_items_num_list) / total_outbound_num * 100
        # 出库量累计占比
        cum_outbound_ratio_list = np.cumsum(outbound_ratio_list)
        # 订单行总数，
        total_order_line_num = np.sum(sorted_order_lines_list)
        # 平均订单行数，
        average_order_line_num = total_order_line_num / total_order_num
        # 平均行件数，
        average_row_item_num = total_outbound_num / total_order_line_num
        # 平均订单出库量
        average_order_outbound_num = total_outbound_num / total_order_num

        # save data
        EQ = {'序号': serial_num_list, '拣货单号': picking_order_nums,
              '出库量': sorted_outbound_items_num_list,
              '订单行数': sorted_order_lines_list, '订单占比/%': np.round(order_ratio_list, decimals=2),
              '订单累计占比/%': np.round(cum_order_ratio_list, decimals=2),
              '出库量占比/%': np.round(outbound_ratio_list, decimals=2),
              '出库量累计占比/%': np.round(cum_outbound_ratio_list, decimals=2)}

        EQ_dataframe = pd.DataFrame(EQ)
        EQ_dataframe['订单总数'] = total_order_num
        EQ_dataframe['出库总量'] = total_outbound_num
        EQ_dataframe['订单行总数'] = total_order_line_num
        EQ_dataframe['平均订单行数'] = np.round(average_order_line_num, decimals=1)
        EQ_dataframe['平均行件数'] = np.round(average_row_item_num, decimals=1)
        EQ_dataframe['平均订单出库量'] = np.round(average_order_outbound_num, decimals=1)

        self._save_to_excel(EQ_dataframe, sheet_name='FLC冷库EQ')

    def top_twenty_order(self):
        EQ_statistics_dataframe = self._extract_from_excel_to_dataframe(
            sheet_name='FLC冷库EQ').iloc[:, :8]
        idx = EQ_statistics_dataframe[
            EQ_statistics_dataframe['订单累计占比/%'] >= 20].index.tolist()[0]
        EQ_statistics_dataframe = EQ_statistics_dataframe.iloc[:idx+1, :8]

        max_order_line = np.max(
            EQ_statistics_dataframe['订单行数'])  # 最大订单行, e.g. 62

        # calculate the split margin
        if max_order_line / self.split_num <15:
            split_margin = 10
        else:
            split_margin = 15

        # range list
        range_list = []

        for i in np.arange(1, self.split_num + 1):
            if i == 1:
                range_list.append(str(i) + '~' + str(split_margin * i))
            elif i == self.split_num:
                range_list.append(
                    str(1 + int(range_list[-1].split('~')[1])) + '~' + str(max_order_line))
            else:
                range_list.append(str(1 + int(range_list[-1].split('~')[1])) + '~' + str(split_margin * i))

        order_line_range_list = []

        for order_line in EQ_statistics_dataframe['订单行数']:
            for i, range in enumerate(range_list):
                if order_line <= int(range.split('~')[1]):
                    order_line_range_list.append(range_list[i])
                    break

        EQ_statistics_dataframe['订单行区间'] = order_line_range_list

        self._save_to_excel(EQ_statistics_dataframe, sheet_name='前20订单')

    def original_touches_data(self):
        r"""动碰即改货品在改日期内被产生订单的次数。"""
        goods_bill = self._get_goods_bill()
        date_list = self.unique_datetimes  # 日期
        date_str_list = [x.strftime('%Y/%m/%d') for x in date_list]
        data = np.zeros(shape=(len(goods_bill), len(date_list)))
        print('----- Calculating touches, please wait! -----')

        for i, goods_id in enumerate(goods_bill):
            for j, date in enumerate(date_list):
                mask = self.data['货品编号'] == goods_id
                count = np.sum(self.data.loc[mask, :]['日期'] == date)
                data[i, j] = count
        print('------------ Data updated--------------')
        touches_dataframe = pd.DataFrame(data, index=goods_bill,
                                         columns=date_str_list)
        self._save_to_excel(touches_dataframe, sheet_name='动碰原表')

    def touches_data(self):
        r"""动碰核算

        # 序号， 货品编号， 总动碰， 实际出库天数, 出库天数占比， 出库天区间， 日平均动碰，
        总出库量， 单捡次出货量."""
        # 货品编号
        goods_bill = self._get_goods_bill()
        # 总动碰
        data = self._extract_from_excel_to_dataframe(sheet_name='动碰原表')

        total_touches_per_id_list = []
        # 实际出库天数, 即该货品编号动碰不为零天数之和
        actual_outbound_day_list = []

        for goods_id in goods_bill:
            total_touches_per_id_list.append(
                np.sum(self.data.loc[:, '货品编号'] == goods_id))
            row = data[data.iloc[:, 0] == goods_id].iloc[:, 1:]
            actual_outbound_day_list.append(
                np.sum(row.values.squeeze()))

        # 总出库天数
        total_outbound_day = np.sum(actual_outbound_day_list)
        # 出库天数占比, /%
        outbound_day_ratio_list = np.array(actual_outbound_day_list,
                                           dtype=np.float64) / total_outbound_day * 100
        # 出库天区间
        outbound_day_range_list = ['1~7' for _ in range(len(goods_bill))]
        # 日平均动碰
        daily_touche_list = np.array(total_touches_per_id_list,
                                     dtype=np.float64) / (np.array(
            actual_outbound_day_list))
        # 总出库量, 查找FLC冷库IQ，通过货品编号，获取出库量数据
        total_outbound_items_quantity_list = []
        flc_IQ_data = self._extract_from_excel_to_dataframe(
            sheet_name='FLC冷库IQ').loc[:, ['货品编号', '出库量']]
        for goods_id in goods_bill:
            outbound_quantity = flc_IQ_data[
                                    flc_IQ_data['货品编号'] == goods_id].loc[:,
                                '出库量'].tolist()
            total_outbound_items_quantity_list.extend(outbound_quantity)
        # 单拣次出货量
        single_picking_outbound_quantity_list = np.array(
            total_outbound_items_quantity_list, dtype=np.float64) / np.array(
            total_touches_per_id_list)
        # 序号
        serial_num = FullYear._self_sort(actual_outbound_day_list, reverse_mark=True)

        # save data
        touches_checking = {'序号': serial_num, '货品编号': goods_bill,
                            '总动碰': total_touches_per_id_list,
                            '实际出库天数':actual_outbound_day_list,
                            '出库天数占比': outbound_day_ratio_list,
                            '出库天区间': outbound_day_range_list,
                            '日平均动碰': daily_touche_list,
                            '总出库量': total_outbound_items_quantity_list,
                            '单捡次出货量': single_picking_outbound_quantity_list}
        touches_checking_dataframe = pd.DataFrame(touches_checking)
        self._save_to_excel(touches_checking_dataframe, sheet_name='动碰核算')


    def test(self):
        dict1 = {'a': [i for i in range(100)], 'b': [i for i in range(100, 200)], 'c': [i for i in range(200, 300)]}
        test_data = pd.DataFrame(dict1)
        self._save_to_excel(test_data, sheet_name='动碰核算')
        self._save_to_excel(test_data, sheet_name='报价')
        self._save_to_excel(test_data, sheet_name='FLC')

        data = self._extract_from_excel_to_dataframe(sheet_name='报价')

        self.merge_excel()


if __name__ == '__main__':
    time_start = time.time()
    # single_day = SingleDay('国药FLC冷库出库分析2019.9.18.xlsx', 0)
    # single_day.plot_plato()E:\GZ\FLC Data Analysis\国药FLC冷库出库分析2019.9.18.xlsx
    root = os.path.abspath(os.path.join(os.getcwd(), "..")) #os.getcwd()
    full_year = FullYear(root, '../国药FLC冷库出库分析2019.9.18.xlsx')
    # print("------------ Test -----------")
    # full_year.test()
    print("------------ flc row items statistics-----------")

    full_year.flc_row_items_statistics()
    print("-------------- flc IQ -------------------------")
    full_year.flc_IQ()
    print("----------------- flc EQ ----------------------")
    full_year.flc_EQ()
    print("-------- Top 20 order -----------------------------")
    full_year.top_twenty_order()
    print("----------- Original touches data ---------------")
    full_year.original_touches_data()
    print("------------------ touches data -----------------")
    full_year.touches_data()

    full_year.merge_excel()

    time_end = time.time()
    time_count(time_start, time_end)



