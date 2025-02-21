# coding: utf-8

# PAGANES
import copy
import numpy as np
import numexpr as ne  # more performant than usual python mathematical operations, especially with arrays
import matplotlib.pyplot as plt

from tools.accessories import convert_units, data_extend
# MODULES
from tools.data_reading import DataPars as DataPars
from tools.unit_conversion import units_conversion


# SKK model class
class SKKModel:
    def __init__(self, lambda_current, lambda_reference, n_reference, wavelength, k, side_effect_smoothing=True):
        self.lambda_current = lambda_current
        self.lambda_reference = lambda_reference
        self.n_reference = n_reference
        self.wavelength = wavelength
        self.k = k
        self.side_effect_smoothing = side_effect_smoothing

    def wavelength_and_k_extend(self):
        try:
            new_wavelength = copy.deepcopy(self.wavelength)
            new_k = copy.deepcopy(self.k)
            # before
            step = self.wavelength[1] - self.wavelength[0]
            k = (self.k[1] - self.k[0]) / step  # k in k * x + b = y
            b = self.k[1] - k * self.wavelength[1]
            for j in range(1, 6):
                new_wavelength = np.insert(new_wavelength, 0, self.wavelength[0] - j * step)
                new_k = np.insert(new_k, 0, k * new_wavelength[0] + b)
            # after
            step = self.wavelength[-1] - self.wavelength[-2]
            k = (self.k[-1] - self.k[-2]) / step
            b = self.k[-2] - k * self.wavelength[-2]
            for j in range(1, 6):
                new_wavelength = np.insert(new_wavelength, len(new_wavelength), self.wavelength[-1] + j * step)
                new_k = np.insert(new_k, len(new_k), k * new_wavelength[-1] + b)
            return new_wavelength, new_k
        except Exception as e:
            raise Exception(f"Critical error in SKKModel:wavelength_and_k_extend: {str(e)}") from e

    def n_calc(self):
        try:
            lambda_reference = self.lambda_reference
            lambda_current = self.lambda_current
            if lambda_reference - lambda_current != 0 and lambda_reference + lambda_current != 0:
                if self.side_effect_smoothing:
                    self.wavelength, self.k = self.wavelength_and_k_extend()
                integral = 0
                w_before = np.delete(self.wavelength, len(self.wavelength) - 1)
                w_after = np.delete(self.wavelength, 0)
                k_before = np.delete(self.k, len(self.k) - 1)
                k_after = np.delete(self.k, 0)
                # part 1
                sum_array = ne.evaluate("where(w_before - lambda_reference == 0, 0, -((-w_after + lambda_reference) * k_before + k_after * (w_before - lambda_reference)) * log(abs(w_before - lambda_reference)) / (2 * (w_before - w_after) * (lambda_reference + lambda_current) * (lambda_reference - lambda_current)))")
                integral = integral + np.sum(sum_array)
                # part 2
                sum_array = ne.evaluate("where(w_before - lambda_current == 0, 0, -((w_after - lambda_current) * k_before - k_after * (w_before - lambda_current)) *log(abs(w_before - lambda_current)) / (2 * (w_before - w_after) * (lambda_reference + lambda_current) * (lambda_reference - lambda_current)))")
                integral = integral + np.sum(sum_array)
                # part 3
                sum_array = ne.evaluate("where(w_before + lambda_reference == 0, 0, -((-w_after - lambda_reference) * k_before + k_after * (w_before + lambda_reference)) * log(abs(w_before + lambda_reference)) / (2 * (w_before - w_after) * (lambda_reference + lambda_current) * (lambda_reference - lambda_current)))")
                integral = integral + np.sum(sum_array)
                # part 4
                sum_array = ne.evaluate("where(w_before + lambda_current == 0, 0, -((w_after + lambda_current) * k_before - k_after * (w_before + lambda_current)) * log(abs(w_before + lambda_current)) / (2 * (w_before - w_after) * (lambda_reference + lambda_current) * (lambda_reference - lambda_current)))")
                integral = integral + np.sum(sum_array)
                # part 5
                sum_array = ne.evaluate("where(w_after - lambda_reference == 0, 0, ((-w_after + lambda_reference) * k_before + k_after * (w_before - lambda_reference)) * log(abs(w_after - lambda_reference)) / (2 * (w_before - w_after) * (lambda_reference + lambda_current) * (lambda_reference - lambda_current)))")
                integral = integral + np.sum(sum_array)
                # part 6
                sum_array = ne.evaluate("where(w_after - lambda_current == 0, 0, ((w_after - lambda_current) * k_before - k_after * (w_before - lambda_current)) * log(abs(w_after - lambda_current)) / (2 * (w_before - w_after) * (lambda_reference + lambda_current) * (lambda_reference - lambda_current)))")
                integral = integral + np.sum(sum_array)
                # part 7
                sum_array = ne.evaluate("where(w_after + lambda_reference == 0, 0, ((-w_after - lambda_reference) * k_before + k_after * (w_before + lambda_reference)) * log(abs(w_after + lambda_reference)) / (2 * (w_before - w_after) * (lambda_reference + lambda_current) * (lambda_reference - lambda_current)))")
                integral = integral + np.sum(sum_array)
                # part 8
                sum_array = ne.evaluate("where(w_after + lambda_current == 0, 0, ((w_after + lambda_current) * k_before - k_after * (w_before + lambda_current)) * log(abs(w_after + lambda_current)) / (2 * (w_before - w_after) * (lambda_reference + lambda_current) * (lambda_reference - lambda_current)))")
                integral = integral + np.sum(sum_array)
                return self.n_reference + (2 * (lambda_reference ** 2 - lambda_current ** 2) / np.pi) * integral
            else:
                return self.n_reference
        except Exception as e:
            raise Exception(f"Critical error in SKKModel:n_calc: {str(e)}") from e


def demo():
    test_file_path = "examples/Quartz_Epara_300K_DOCCD.data_cm-1.tsv"
    test_file_unit = "cm-1"
    test_file_wavelength_column = 0
    test_file_n_column = 1
    test_file_k_column = 2
    lambda_reference = 16000
    reference_unit = "cm-1"
    n_reference = 1.545
    # data read
    test_file_read = DataPars(test_file_path, None)
    test_file_read.file_pars_f()
    test_data = test_file_read.file_body
    test_data_wavelength = test_data[:, test_file_wavelength_column]
    test_data_n = test_data[:, test_file_n_column]
    test_data_k = test_data[:, test_file_k_column]
    # UNITS
    array_dict = dict()
    array_dict["test_data_n"] = test_data_n
    array_dict["test_data_k"] = test_data_k
    test_data_wavelength, array_dict, is_it_reversed = convert_units(test_data_wavelength, array_dict, test_file_unit, "micron")
    test_data_n = array_dict["test_data_n"]
    test_data_k = array_dict["test_data_k"]
    del array_dict
    lambda_reference = units_conversion(lambda_reference, reference_unit, "micron")
    # DATA extend
    test_data_wavelength_before = copy.deepcopy(test_data_wavelength)
    test_data_wavelength, test_data_k, test_data = data_extend(test_data_wavelength, test_data_k, [], lambda_reference)
    # calc
    n_array = np.zeros(len(test_data_wavelength))
    for i, v in enumerate(test_data_wavelength):
        my_SKK = SKKModel(v, lambda_reference, n_reference, test_data_wavelength, test_data_k, side_effect_smoothing=True)
        n_array[i] = my_SKK.n_calc()
    # plot
    fig, ax = plt.subplots()
    ax.plot(test_data_wavelength_before, test_data_n, label='data n')
    ax.plot(test_data_wavelength, n_array, label='calc n')
    ax.legend()
    plt.show()


# demo()
