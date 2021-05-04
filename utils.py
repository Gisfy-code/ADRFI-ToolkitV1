# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np


def func1(x, b, f_zero):
    return (f_zero - 0.05) * np.exp((-b) * x) + 0.05


def func2(x, S20, S100):
    return (0.01 * (x - S20) / (S100-S20)) + 0.05 * (S100 - x) / (S100-S20)


def func3(x, S100, S200):
    return (0.005 * (x - S100) / (S200 - S100)) + 0.01 * (S200-x)/(S200-S100)


def func4(x, r, S200):
    return 0.005 * np.exp(-(x - S200) / r)


n2c = {
    1: 'A',
    2: 'B',
    3: 'C',
    4: 'D',
    5: 'E',
    6: 'F',
    7: 'G',
    8: 'H',
    9: 'I',
    10: 'J',
    11: 'K',
    12: 'L',
    13: 'M',
    14: 'N',
    15: 'O',
    16: 'P',
    17: 'Q',
    18: 'R',
    19: 'S',
    20: 'T',
    21: 'U',
    22: 'V',
    23: 'W',
    24: 'X',
    25: 'Y',
    26: 'Z',
}
