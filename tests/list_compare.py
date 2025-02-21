# coding: utf-8

import numpy as np


def float_compare(value1, value2, accuracy):
    if np.fabs(value1 - value2) > accuracy:
        return False
    return True


def list_compare(list1, list2, accuracy):
    if len(list1) != len(list2):
        return False
    for i in range(len(list1)):
        if np.fabs(list1[i] - list2[i]) > accuracy:
            return False
    return True


def list_of_lists_compare(list1, list2, accuracy):
    if len(list1) != len(list2):
        return False
    for i in range(len(list1)):
        if len(list1[i]) != len(list2[i]):
            return False
        for j in range(len(list1[i])):
            if np.fabs(list1[i][j] - list2[i][j]) > accuracy:
                return False
    return True