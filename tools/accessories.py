# coding: utf-8

# PACKAGES
import numpy as np

# MODULES
from tools.unit_conversion import units_conversion


def convert_units(array_to_convert, associated_arrays, units_from, units_to):
    """
        Converts a list of numerical values and associated data from one unit to another.

        This function calculates the converted values of an input list from one unit `units_from`
        to another unit `units_to`, and applies the same reordering adjustment to any associated
        data arrays if the conversion results in a reversed order.

        :param array_to_convert: list[float] or np.ndarray[float]
            The numerical array to be converted from one unit to another.
        :param associated_arrays: dict[str, list]
            A dictionary containing other data arrays associated with `array_to_convert`. The
            associated arrays will be reversed in case the converted `array_to_convert`
            undergoes a reversal.
        :param units_from: str
            The unit of the input `array_to_convert`; maybe cm-1, nm, A and micron.
        :param units_to: str
            The target unit to which the `array_to_convert` should be converted; maybe cm-1, nm, A and micron.
        :return: tuple[list[float]  or np.ndarray[float], dict[str, list], bool]
            A tuple containing:
            - The converted array of numerical values.
            - The adjusting associated arrays dictionary (reversed if the conversion order changes).
            - A boolean indicating whether the data (array and associations) has been reversed.
    """
    try:
        if units_from != units_to:
            new_array = [units_conversion(v, units_from, units_to) for v in array_to_convert]
            if new_array[0] > new_array[1]:
                new_array = new_array[::-1]
                for key in associated_arrays.keys():
                    associated_arrays[key] = associated_arrays[key][::-1]
                return new_array, associated_arrays, True
            return new_array, associated_arrays, False
        return array_to_convert, associated_arrays, False
    except Exception as e:
        raise Exception(f"Critical error in convert_units: {str(e)}") from e


def data_extend(data_wavelength, data_k, data_albedo, lambda_reference):
    """
        Adjusts and extends input data arrays based on a reference wavelength.

        This function takes input arrays of wavelengths, associated parameters
        (data_k), and albedo values, and extends them if the reference wavelength
        falls outside the range of the input wavelength array. Extension values
        are interpolated for the wavelengths and zero-filled for the data_k and albedo.

        :param data_wavelength: numpy.ndarray
            Array containing wavelength data in ascending order.
        :param data_k: numpy.ndarray
            Array representing associated parameter values, matching the size of
            `data_wavelength`.
        :param data_albedo: numpy.ndarray
            Array representing albedo values, matching the size of `data_wavelength`.
        :param lambda_reference: float
            Reference wavelength used as a basis for extending the input arrays.

        :return: tuple
            A tuple containing extended arrays: `(data_wavelength, data_k, data_albedo)`,
            where all arrays are of the same length and include values to support the
            reference wavelength if out-of-bounds.
    """
    try:
        if lambda_reference < data_wavelength[0]:
            wavelength_extend = np.linspace(lambda_reference, data_wavelength[0], int((data_wavelength[0] - lambda_reference) / (data_wavelength[1] - data_wavelength[0])), endpoint=False)
            k_extend = np.zeros(len(wavelength_extend))
            data_wavelength = np.insert(data_wavelength, 0, wavelength_extend)
            data_k = np.insert(data_k, 0, k_extend)
            data_albedo = np.insert(data_albedo, 0, k_extend)
        elif lambda_reference > data_wavelength[-1]:
            wavelength_extend = np.linspace(data_wavelength[-1] + data_wavelength[-1] - data_wavelength[-2], lambda_reference + data_wavelength[-1] - data_wavelength[-2], 1 + int((lambda_reference - data_wavelength[-1] - data_wavelength[-1] + data_wavelength[-2]) / (data_wavelength[-1] - data_wavelength[-2])), endpoint=False)
            k_extend = np.zeros(len(wavelength_extend))
            data_wavelength = np.insert(data_wavelength, len(data_wavelength), wavelength_extend)
            data_k = np.insert(data_k, len(data_k), k_extend)
            data_albedo = np.insert(data_albedo, len(data_albedo), k_extend)
        return data_wavelength, data_k, data_albedo
    except Exception as e:
        raise Exception(f"Critical error in data_extend: {str(e)}") from e
