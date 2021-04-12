#!/bin/bash 

# using feature engineering

# extrapolate distance
python main.py --data=n_body_extrapolate_distance --fe

# extrapolate mass
python main.py --data=n_body_extrapolate_mass --fe

# interpolation
python main.py --data=n_body_interpolate --fe
