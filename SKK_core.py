# coding: utf-8

# PACKAGES
import copy
import numpy as np
import numexpr as ne
import matplotlib.pyplot as plt


# MODULES
from tools.data_reading import DataPars as DataPars
from tools.unit_conversion import units_conversion
from tools.accessories import convert_units, data_extend


# SKK model class
class SKKModel:
    def __init__(self, lambda_current, lambda_reference, n_reference, wavelength, k_array, side_effect_smoothing=True):
        self.lambda_current = lambda_current
        self.lambda_reference = lambda_reference
        self.n_reference = n_reference
        self.wavelength = wavelength
        self.k_array = k_array
        self.side_effect_smoothing = side_effect_smoothing

    def wavelength_and_k_extend(self):
        """
            Perform extension of the wavelength and k_array by extrapolating values.

            We apply this function only if we need to smooth edge effects, i.e. when
            self.side_effect_smoothing is True. In this case, we extrapolate k in both directions,
            assuming it is a linear function. The number of extrapolation points is determined
            by the constant NUM_EXTENSIONS. By default, it is 5, but this value has no firm
            theoretical basis: smoothing starts working even at NUM_EXTENSIONS = 3, so we just
            increased it to the first round value which is 5.

            This function extends both the wavelength and k_array arrays by adding extrapolated
            values at the beginning and the end. It first calculates the line parameters (slope and
            intercept) using the adjacent points at both ends of the arrays. Using these values,
            it performs a linear extrapolation to create the desired number of points (NUM_EXTENSIONS)
            to extend on either side of the arrays.

        :return wavelength: np.ndarray
            The extended wavelength array with extrapolated points added.
        :return k_array: np.ndarray
            The extended k_array corresponding to the extrapolated wavelength array.
        """
        NUM_EXTENSIONS = 5  # Number of points to extend on either side

        def calculate_line_params(x1, y1, x2, y2):
            """Calculate the slope and intercept for the line passing through two points."""
            slope = (y2 - y1) / (x2 - x1)
            intercept = y2 - slope * x2
            return slope, intercept

        def extend_wavelength_and_k(wavelength, k_array, start_idx, step, slope, intercept, is_forward=True):
            """Extend wavelength and k_array forward or backward."""
            for index in range(1, NUM_EXTENSIONS + 1):
                new_value = self.wavelength[start_idx] + (index * step if is_forward else -index * step)
                wavelength = np.insert(wavelength, len(wavelength) if is_forward else 0, new_value)
                k_array = np.insert(k_array, len(k_array) if is_forward else 0, slope * new_value + intercept)
            return wavelength, k_array

        try:
            # Step 1: Initialize new wavelength and k arrays
            new_wavelength = copy.deepcopy(self.wavelength)
            new_k = copy.deepcopy(self.k_array)

            # Step 2: Extend backward (prepend)
            backward_step = self.wavelength[1] - self.wavelength[0]
            slope, intercept = calculate_line_params(self.wavelength[0], self.k_array[0], self.wavelength[1], self.k_array[1])
            new_wavelength, new_k = extend_wavelength_and_k(new_wavelength, new_k, 0, backward_step, slope, intercept, is_forward=False)

            # Step 3: Extend forward (append)
            forward_step = self.wavelength[-1] - self.wavelength[-2]
            slope, intercept = calculate_line_params(self.wavelength[-2], self.k_array[-2], self.wavelength[-1], self.k_array[-1])
            new_wavelength, new_k = extend_wavelength_and_k(new_wavelength, new_k, -1, forward_step, slope, intercept, is_forward=True)

            # Step 4: Return the extended arrays
            return new_wavelength, new_k
        except Exception as e:
                raise Exception(f"Critical error in SKKModel:wavelength_and_k_extend: {str(e)}") from e

    def n_calc(self):
        """
        	Computes the refractive index n based on the SKK integral.

        	To work with the SKK integral, we have resorted to a technique standard for such integrals:
        	since the function k is discrete (from the experiment it is just a set of points), k can be
        	represented as a piecewise defined linear function, where each piece is a linear function
        	of the form slope * wavelength + intercept between each known pair of points k. Then
        	the integral of SKK can be calculated. The result will contain a singularity only at the edges
        	(and only mathematically, since in fact this singularity does not appear due to the fact that
        	the edge points are not the edge of the spectrum, the experimental data are simply  cut off
        	there). It is convenient to break the obtained expression into 8 parts (parts 1 through 8 below).

        :return: float
            The computed refractive index n as per the given inputs and formula logic.
        """
        try:
            # if for any reason lambda_reference and lambda_current are equal, we immediately return self.n_reference
            # these conditions are a mathematical restriction on the applicability of further formulas,
            # so we have included the case of self.lambda_reference + self.lambda_current there as well
            if self.lambda_reference - self.lambda_current == 0 or self.lambda_reference + self.lambda_current == 0:
                return self.n_reference

            # handle side effect smoothing
            if self.side_effect_smoothing:
                self.wavelength, self.k_array = self.wavelength_and_k_extend()

            # prepare arrays for computation
            # Here we split the data for wavelength and k into two arrays. Since we represent k as a
            # piecewise-defined function (see doc-string to this function), each piece will have a beginning
            # and an end. All start points are collected into arrays with start suffixes and all end points
            # into arrays with end suffixes.
            # We drop the boundary points here because mathematically the formulas below contain a singularity
            # in them and since the experimental data is cut off here, we have no way to get around this problem
            # any other way.
            wavelength_start = np.delete(self.wavelength, -1)
            wavelength_end = np.delete(self.wavelength, 0)
            k_start = np.delete(self.k_array, -1)
            k_end = np.delete(self.k_array, 0)

            # extract inputs (as ne module cannot handle variables with self)
            lambda_reference = self.lambda_reference
            lambda_current = self.lambda_current

            # calculate integral
            integral = 0
            # part 1
            sum_array = ne.evaluate("where(wavelength_start - lambda_reference == 0, 0, -((-wavelength_end + lambda_reference) * k_start + k_end * (wavelength_start - lambda_reference)) * log(abs(wavelength_start - lambda_reference)) / (2 * (wavelength_start - wavelength_end) * (lambda_reference + lambda_current) * (lambda_reference - lambda_current)))")
            integral += np.sum(sum_array)
            # part 2
            sum_array = ne.evaluate("where(wavelength_start - lambda_current == 0, 0, -((wavelength_end - lambda_current) * k_start - k_end * (wavelength_start - lambda_current)) *log(abs(wavelength_start - lambda_current)) / (2 * (wavelength_start - wavelength_end) * (lambda_reference + lambda_current) * (lambda_reference - lambda_current)))")
            integral += np.sum(sum_array)
            # part 3
            sum_array = ne.evaluate("where(wavelength_start + lambda_reference == 0, 0, -((-wavelength_end - lambda_reference) * k_start + k_end * (wavelength_start + lambda_reference)) * log(abs(wavelength_start + lambda_reference)) / (2 * (wavelength_start - wavelength_end) * (lambda_reference + lambda_current) * (lambda_reference - lambda_current)))")
            integral += np.sum(sum_array)
            # part 4
            sum_array = ne.evaluate("where(wavelength_start + lambda_current == 0, 0, -((wavelength_end + lambda_current) * k_start - k_end * (wavelength_start + lambda_current)) * log(abs(wavelength_start + lambda_current)) / (2 * (wavelength_start - wavelength_end) * (lambda_reference + lambda_current) * (lambda_reference - lambda_current)))")
            integral += np.sum(sum_array)
            # part 5
            sum_array = ne.evaluate("where(wavelength_end - lambda_reference == 0, 0, ((-wavelength_end + lambda_reference) * k_start + k_end * (wavelength_start - lambda_reference)) * log(abs(wavelength_end - lambda_reference)) / (2 * (wavelength_start - wavelength_end) * (lambda_reference + lambda_current) * (lambda_reference - lambda_current)))")
            integral += np.sum(sum_array)
            # part 6
            sum_array = ne.evaluate("where(wavelength_end - lambda_current == 0, 0, ((wavelength_end - lambda_current) * k_start - k_end * (wavelength_start - lambda_current)) * log(abs(wavelength_end - lambda_current)) / (2 * (wavelength_start - wavelength_end) * (lambda_reference + lambda_current) * (lambda_reference - lambda_current)))")
            integral += np.sum(sum_array)
            # part 7
            sum_array = ne.evaluate("where(wavelength_end + lambda_reference == 0, 0, ((-wavelength_end - lambda_reference) * k_start + k_end * (wavelength_start + lambda_reference)) * log(abs(wavelength_end + lambda_reference)) / (2 * (wavelength_start - wavelength_end) * (lambda_reference + lambda_current) * (lambda_reference - lambda_current)))")
            integral += np.sum(sum_array)
            # part 8
            sum_array = ne.evaluate("where(wavelength_end + lambda_current == 0, 0, ((wavelength_end + lambda_current) * k_start - k_end * (wavelength_start + lambda_current)) * log(abs(wavelength_end + lambda_current)) / (2 * (wavelength_start - wavelength_end) * (lambda_reference + lambda_current) * (lambda_reference - lambda_current)))")
            integral += np.sum(sum_array)

            # compute n
            return self.n_reference + (2 * (lambda_reference ** 2 - lambda_current ** 2) / np.pi) * integral
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
    test_file_read = DataPars(test_file_path)
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
