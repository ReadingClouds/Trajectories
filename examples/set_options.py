# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 17:57:21 2022

@author: paclk
"""


def set_options(minim, expt):
    if expt == "ref":
        if "fixed_point_iterator" in minim:
            options = {
                "pioptions": {
                    "maxiter": 1000,
                    "miniter": 5,
                    "disp": False,
                    "relax": 1.0,
                    "relax_reduce": 0.95,
                    "tol": 0.000001,
                    "norm": "max_abs_error",
                },
                "minoptions": {
                    "max_outer_loops": 1,
                    "tol": 0.000001,
                    "minimize_options": {
                        "maxiter": 200,
                        "disp": False,
                    },
                },
            }
        else:
            options = {
                "pioptions": {},
                "minoptions": {
                    "max_outer_loops": 1,
                    "tol": 0.000001,
                    "minimize_options": {
                        "maxiter": 200,
                        "disp": False,
                    },
                },
            }

    elif expt == "std":
        if "fixed_point_iterator" in minim:
            options = {
                "pioptions": {
                    "maxiter": 200,
                    "miniter": 5,
                    "disp": False,
                    "relax": 1.0,
                    "relax_reduce": 0.95,
                    "tol": 0.01,
                    "norm": "max_abs_error",
                },
                "minoptions": {
                    "max_outer_loops": 3,
                    "tol": 0.01,
                    "minimize_options": {
                        "maxiter": 20,
                        "disp": False,
                    },
                },
            }
        else:
            options = {
                "pioptions": {},
                "minoptions": {
                    "max_outer_loops": 4,
                    "tol": 0.05,
                    "minimize_options": {
                        "maxiter": 10,
                        "disp": False,
                    },
                },
            }
    return options
