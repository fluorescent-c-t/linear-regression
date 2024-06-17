#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 09:34:31 2024

@author: taishikochi
"""

import os
dirname = os.path.dirname(__file__)
os.chdir(dirname)

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Arial' 
plt.rcParams["font.size"] = 15 
plt.rcParams['figure.figsize'] = [6.5, 5.]

plt.rcParams['xtick.direction'] = 'in' 
plt.rcParams['ytick.direction'] = 'in' 
plt.rcParams["xtick.minor.visible"] = True
plt.rcParams["ytick.minor.visible"] = True
plt.rcParams['xtick.top'] = True 
plt.rcParams['ytick.right'] = True


def read_dataset(file_path, skip_empty_lines=True):
    with open(file_path, 'r') as file:
        data = file.read().splitlines()
        if skip_empty_lines:
            data = [line for line in data if line]
        
        x = []
        y = []
        
        for line in data:
            values = line.split()
            if len(values) == 2:  # Ensure there are  2 columns
                x.append(float(values[0]))
                y.append(float(values[1]))
            else:
                print("Dataset is not 1 dimensional")
                print("Check your dataset")
        
        return x, y

def read_threshold(file_path, skip_empty_lines=True):
    with open(file_path, 'r') as file:
        if skip_empty_lines:
            data = [line.strip() for line in file if line.strip()]
        else:
            data = [line.strip() for line in file]

        if len(data) != 1:
            print("Error: Expected exactly one non-empty line in the file.")
            return None
        
        try:
            value = float(data[0])
            return value
        except ValueError:
            print("Error: Could not convert the value to a single float.")
            return None

def split_dataset_with_threshold(x, y, xth):
    assert len(x) == len(y), "Lengths of x and y must be the same"
    
    x1 = []
    y1 = []
    x2 = []
    y2 = []
    
    for i in range(len(x)):
        if x[i] <= xth:
            x1.append(x[i])
            y1.append(y[i])
            
        if xth <= x[i]:
            x2.append(x[i])
            y2.append(y[i])
    
    return np.array(x1), np.array(x2), np.array(y1), np.array(y2)
  

def bylin_reg(x1, y1, x2, y2, xth):
    n1 = len(x1)
    n2 = len(x2)
    
    a1 = (np.dot(x1 - xth, y1) - (x1 - xth).sum() * y1.sum() / n1) / (((x1 - xth) ** 2).sum() - (x1 - xth).sum() ** 2 / n1)
    a2 = (np.dot(x2 - xth, y2) - (x2 - xth).sum() * y2.sum() / n2) / (((x2 - xth) ** 2).sum() - (x2 - xth).sum() ** 2 / n2)
    
    b2 = y1.sum() / n1 - a1 * (x1 - xth).sum() / n1 - a2 * xth
    b1 = y2.sum() / n2 - a2 * (x2 - xth).sum() / n2 - a1 * xth
    
    return a1, b1, a2, b2

def r2(x, y, a1, b1, a2, b2, xth):
    n = len(x)
    fx = np.zeros_like(x)
    for i in range(n):
        if x[i] <= xth:
            fx[i] = a1 * x[i] + b1
        else:
            fx[i] = a2 * x[i] + b2
    
    y_bar = np.mean(y)
    
    r2 = 1 - ((y - fx) ** 2).sum() / ((y - y_bar) ** 2).sum()
    
    return fx, r2

x, y = read_dataset('./dataset.txt')
xth  = read_threshold('./threshold.txt')

x1, x2, y1, y2 = split_dataset_with_threshold(x, y, xth)

a1, b1, a2, b2 = bylin_reg(x1, y1, x2, y2, xth)

fx, r2 = r2(x, y, a1, b1, a2, b2, xth)

plt.xlabel("T [degC]")
plt.ylabel("λα / λ")
plt.plot(x, fx, 'k-', lw=1.5, alpha=0.6, label='regression')
plt.plot(x, y, 'o', color='none', markersize=5, markeredgewidth=2, markeredgecolor='blue', alpha=0.8, label='experiment')

print(r2, fx)