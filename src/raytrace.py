#!/usr/bin/env python3
"""
Raytrace an image using the geodesic solver
"""
import numpy as np
import itertools
import geodesics

def example0():
    r_s = 1
    a = 0.3 # J / (M c)
    metric = geodesics.kerr_metric
    extra_params = [r_s, a]
    return metric, extra_params

def main():
    metric, extra_params = example0()
    w = 32
    h = 24
    aspect_ratio = h/w
    x = np.linspace(-1, 1, w)
    y = np.linspace(aspect_ratio, -aspect_ratio,  h)
    xx, yy = np.meshgrid(x, y)
    dim = 4
    # We have 2*dim because we will store x and xdot at the exit point.
    exits = np.nan * np.ones((h, w, 2*dim))
    # We'll visit pixels in random order.
    pairs = list(itertools.product(range(h), range(w)))
    shuffled_pairs = np.random.permutation(pairs)
    # TODO: finish

    pass

if __name__ == "__main__":
    main()
