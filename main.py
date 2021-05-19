# -*- coding: UTF-8 -*-
"""
@Project : FLC Data Analysis
@File    : main.py
@IDE     : PyCharm 
@Author  : Peter
@Date    : 17/05/2021 14:07 
@Brief   : In this main file, we need to achieve the interaction between GUI
           and the inter code.
"""
import os
import configparser
from datetime import datetime

from src.gui import create_gui
from src.full_year import FullYear

if __name__ == "__main__":
    # configuration
    current_time = datetime.now()

    # ============================== 1. GUI ================================== #
    input_file_path, output_folder_path = create_gui()

    # =========================== 2. Full Year =============================== #
    # root = os.getcwd()
    # The root path is the path we are going to store the cache.
    full_year = FullYear(ROOT=output_folder_path, data_path=input_file_path, current_time=current_time)







