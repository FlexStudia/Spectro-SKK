# coding: utf-8

import numpy as np


def units_conversion(value_to_convert, unit_from, unit_to):  # possible units: cm-1, micron, nm, A
    try:
        if unit_from == unit_to:
            return value_to_convert
        if value_to_convert == 0:
            return 0
        if (unit_from == "cm-1" and unit_to == "micron") or (unit_from == "micron" and unit_to == "cm-1"):
            return 10000 / value_to_convert
        if (unit_from == "cm-1" and unit_to == "nm") or (unit_from == "nm" and unit_to == "cm-1"):
            return 10000000 / value_to_convert
        if (unit_from == "cm-1" and unit_to == "A") or (unit_from == "A" and unit_to == "cm-1"):
            return 100000000 / value_to_convert
        if unit_from == "micron" and unit_to == "nm":
            return 1000 * value_to_convert
        if unit_from == "nm" and unit_to == "micron":
            return value_to_convert / 1000
        if unit_from == "micron" and unit_to == "A":
            return 10000 * value_to_convert
        if unit_from == "A" and unit_to == "micron":
            return value_to_convert / 10000
        if unit_from == "nm" and unit_to == "A":
            return 10 * value_to_convert
        if unit_from == "A" and unit_to == "nm":
            return value_to_convert / 10
        print("Unit can be only cm-1, micron, nm or A!")
        return np.NAN
    except Exception as e:
        print(f"Error in units_conversion: {e}")
        return np.NAN


def demo():
    print(f"100 cm-1 in microns will be {units_conversion(100, 'cm-1', 'micron')}")
    print(f"But 100 A in microns is {units_conversion(100, 'A', 'micron')}")


#demo()