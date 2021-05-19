# -*- coding: UTF-8 -*-
"""
@Project : 国药FLC冷库出库分析2019.9.18.xlsx 
@File    : test.py
@IDE     : PyCharm 
@Author  : Peter
@Date    : 17/05/2021 14:21 
@Brief   : 
"""
import pandas as pd
import os

# 读取两个表格
data1 = pd.read_excel('E:\GZ\FLC Data Analysis\国药FLC冷库出库分析2019.9.18.xlsx', index_col=None, sheet_name='FLC行件').iloc[1:10, 3:10]
data2 = pd.read_excel('E:\GZ\FLC Data Analysis\国药FLC冷库出库分析2019.9.18.xlsx', sheet_name='FLC').iloc[1:10, 3:10]

# 将两个表格输出到一个excel文件里面
# writer = pd.ExcelWriter('./results/test_excel.xlsx')
# save_file(data1, sheet_name='sheet1')
# save_file(data2, sheet_name='sheet2')
# data1.to_excel(writer, sheet_name='sheet1')
# data2.to_excel(writer, sheet_name='sheet2')

# 必须运行writer.save()，不然不能输出到本地

# writer.save()
import xlwings

#读取df
# df = pd.read_excel(file_path_df)
# wb = xlwings.Book(r'./results/test_excel.xlsx')
# #在wb中新建一张新的sheet.可以指定位置
#
# sht = wb.sheets.add(name='FLC',before=None,after=None)
# sht.range('A1').value = data1
# #df.values 不然会插入df的索引
# # sht.range("A1").value = df.values
# wb.save()
# wb.close()

# wb = xlwings.Book() # 在wb中新建一张新的sheet.可以指定位置

def save_data(sheet_name, data):
    wb = xlwings.Book(r'E:\GZ\FLC Data Analysis\results\EIQ.xlsx')  # 在wb中新建一张新的sheet.可以指定位置
    sht = wb.sheets.add(name=sheet_name, before=None, after=None)
    sht.range('A1').options(index=False).value = data
    # df.values 不然会插入df的索引
    # sht.range("A1").value = df.values
    wb.save()
    wb.close()

save_data('FLCROw', data1)
save_data('FLC', data2)
# wb.save()
# wb.close()

# os.mknod('')