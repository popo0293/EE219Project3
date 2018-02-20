import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# load data
data = pd.read_csv('data/ratings.csv', header=0, usecols=[0, 1, 2])
print(data.head())
R = data.pivot_table(index='userId', columns='movieId', values='rating', fill_value=0).values

print("(number of users, number of rated movies): ", R.shape)