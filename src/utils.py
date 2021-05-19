# -*- coding: UTF-8 -*-
"""
@Project : 国药FLC冷库出库分析2019.9.18.xlsx 
@File    : utils.py.py
@IDE     : PyCharm 
@Author  : Peter
@Date    : 17/05/2021 16:47 
@Brief   : 
"""

def time_count(start, end):
    r"""

    :params
        @start (time.time(), float)
        @end (time.time(), float)
    """
    elapse = end - start
    minutes =  elapse // 60
    seconds = round(elapse % 60, ndigits=2)
    print("Time used: {:>3} mins, {:>3} s.".format(minutes, seconds))




