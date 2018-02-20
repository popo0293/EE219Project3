import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# load data
data = pd.read_csv('data/ratings.csv', header=0, usecols=[0, 1, 2])
print(data.head())
R = data.pivot_table(index='userId', columns='movieId', values='rating', fill_value=0).values

print("(number of users, number of rated movies): ", R.shape)

# Question 1
user_count = R.shape[0]
movie_count = R.shape[1]
max_rating_count = user_count*movie_count
rating_count = data.size/3
sparsity = rating_count*1.0/max_rating_count
print("Matrix sparsity = %0.4f" % sparsity)

# Question 2
