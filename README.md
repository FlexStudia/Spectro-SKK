# About

This code is a public part of the Short Kramers-Kronig (SKK) and Shkuratov iterative model and is dedicated to the SKK part.
(The iterative model itself is still under development.)


# Content

The main code for calculating n using the SKK model is in SKK_core.py. 
Here you can also find an example of applying this module to Quartz_Epara_300K_DOCCD.data_cm-1.tsv (in the "examples" folder).

In the "tests" folder there is a file test_SKK_core.py, which contains unit tests for the main code.
They can also serve as additional examples.

The "tools" folder contains all auxiliary files (for parsing text files, converting units, etc.).
They are only of interest in relation to our code.

The "Maple" folder contains the calculation sheets that the Maple scientific calculations program can open.
See the next section for an explanation of the details.


# Mathematical part

To calculate n we apply to the SKK integral the standard approach for such integrals. 
Since k is a set of discrete points, we represent k as a piecewise defined function linear between each adjacent pair of points. 
The SKK integral can then be computed into a symbolic form. 
In the resulting expressions we substitute the data of k to compute n.

## Maple

A detailed description of the math calculations can be found in the "Maple" folder. 

It includes three files: SKK_int_calc.mw, test_SKK_int_calc_quartz.mw, and test_SKK_int_calc_simple.mw. 

SKK_int_calc.mw contains a symbolic calculation of the SKK integral (and a demonstration that singularity points do not contribute).

test_SKK_int_calc_quartz.mw contains a calculation using Maple methods for Quartz_Epara data.

test_SKK_int_calc_simple.mw contains a calculation for a fictitious simple example. 

The last two files were used for comparison with unit tests of our code.


# Python code

This code is written entirely in Python and can be used by standard methods 
(via virtual environment installation, etc.). All dependencies are in requirements.txt.


# License & Co
This code is distributed under a standard MIT license. 
That is, it may be used freely with one condition: 
all direct copies must contain the copyright notice “Copyright (c) 2025 Maria Gorbacheva, flex.studia.dev@gmail.com” 
and a copy of the LICENSE file from this repository.


# Question & Propositions
Questions and suggestions are welcome and make the code better. 
Please email me at flex.studia.dev@gmail.com.
