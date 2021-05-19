# -*- coding: UTF-8 -*-
"""
@Project : 国药FLC冷库出库分析2019.9.18.xlsx 
@File    : gui.py.py
@IDE     : PyCharm 
@Author  : Peter
@Date    : 17/05/2021 11:10 
@Brief   : 
"""
import PySimpleGUI as sg


def create_gui():
    # Green & tan color scheme
    sg.ChangeLookAndFeel('GreenTan')

    sg.SetOptions(text_justification='right')

    layout = [[sg.Text('EIQ Analysis for Warehouse Logistics', size=(35, 1),
                       justification='center', font=("Helvetica", 25),
                       relief=sg.RELIEF_RIDGE)],
              [sg.Text('Input Excel File', size=(15, 1)),
               sg.Input(key='-Input-'), sg.FileBrowse()],
              [sg.Text('Target Folder', size=(15, 1)), sg.Input(key='-Output-'),
               sg.FolderBrowse()],
              [sg.Submit(), sg.Cancel()]]

    window = sg.Window('EIQ Analysis Front End', layout, font=("Helvetica", 12))

    event, values = window.read()

    # input_file_path = values['-Input-']
    # target_folder_path = values['-Output-']
    return values['-Input-'], values['-Output-']
