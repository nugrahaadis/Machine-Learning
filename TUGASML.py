# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 21:24:55 2018

@author: User
"""

import numpy as np
import math
import matplotlib.pyplot as plotter


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def evaluate(data_set, params):
    h = []
    e = []
    i = 0
    for d in data_set:
        prediction = d[0] * params[0] + d[1] * params[1] + d[2] * params[2] + d[3] * params[3] + params[4]
        prediction = sigmoid(prediction)
        error = (prediction - fact[i]) ** 2
        h.append(prediction)
        e.append(error)

        update_params(prediction, fact[i], d)
        i += 1

    return h, e


def update_params(prediction, fact, data):
    for i in range(0, 3):
        params[i] -= alpha * 2 * (prediction - fact) * (1 - prediction) * prediction * data[i]
        i += 1
    params[i] -= alpha * 2 * (prediction - fact) * (1 - prediction) * prediction


f = open("iris.data1.csv")
features = [[float(x) for x in line.split(",")] for line in f]
f = open("result.data1.csv")
fact = [float(x) for x in f]
f.close()

alpha = 0.1

params = [np.random.rand(), np.random.rand(), np.random.rand(), np.random.rand(), np.random.rand()]

avg_error_per_epoch = [0,501364144]

for i in range(0, 60):
    h, e = evaluate(features, params)
    avg_error = sum(e)/len(e)
    avg_error_per_epoch.append(avg_error)

plotter.plot(avg_error_per_epoch)
plotter.yscale('log')
plotter.show()