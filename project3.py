import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# load data
r_data = pd.read_csv('data/ratings.csv', header=0, usecols=[0, 1, 2])
print(r_data.head())
R = r_data.pivot_table(index='userId', columns='movieId',
                       values='rating', fill_value=0).values
print("(number of users, number of rated movies): ", R.shape)

# Question 1
user_count = R.shape[0]
movie_count = R.shape[1]
max_rating_count = user_count*movie_count
rating_count = len(r_data.rating.tolist())
sparsity = rating_count*1.0/max_rating_count
print("Matrix sparsity = %0.4f" % sparsity)

# Question 2
plt.figure()
ax = plt.subplot(111)
ratings = r_data.rating.tolist()
xrange = np.arange(0, 5.5, 0.5)
ax.hist(ratings, bins=xrange)
ax.set_xticks(xrange)
ax.set_title("Frequency of rating values")
ax.set_xlabel("rating values")
ax.set_ylabel("movie count")
plt.show()

# Question 3
plt.figure()
movie_rating_count = np.count_nonzero(R, axis=0)
sorted_mrc = sorted(movie_rating_count, reverse=True)
ax = plt.subplot(111)
ax.plot(range(len(movie_rating_count)), sorted_mrc, '-')
ax.set_title("Distribution of ratings among movies")
ax.set_xlabel("most reviewed to least reviewed")
ax.set_ylabel("rating count")
plt.show()

# Question 4
plt.figure()
user_rating_count = np.count_nonzero(R, axis=1)
sorted_urc = sorted(user_rating_count, reverse=True)
ax = plt.subplot(111)
ax.plot(range(len(user_rating_count)), sorted_urc, '-')
ax.set_title("Distribution of ratings among users")
ax.set_xlabel("most active to least active")
ax.set_ylabel("rating count")
plt.show()

# Question 5
'''
ToDo: Explain the salient features of the distribution found in question 3 
and their implications for the recommendation process.
'''

# Question 6
plt.figure()
ax = plt.subplot(111)
movie_var = np.var(R, axis=0)
var_range = np.arange(min(movie_var), max(movie_var)+0.5, 0.5)
ax.hist(movie_var, bins=var_range)
ax.set_xticks(xrange)
ax.set_title("Distribution of ratings among users")
ax.set_xlabel("most active to least active")
ax.set_ylabel("rating count")
plt.show()

# Question 7
'''markdown
$I_u$ : Set of item indices for which ratings have been specifed by user $u$  
$I_v$ : Set of item indices for which ratings have been specifed by user $v$  
$\mu_u$ : Mean rating for user $u$ computed using her specifed ratings  
$r_{uk}$ : Rating of user $u$ for item $k$  

$$\mu_u = \frac{\Sigma_{i\in I_{u}} r_{ui}}{\mid I_u \mid}$$
'''

# Question 8
'''markdown
$I_{u} \cap I_{v}$ represents the indices of movies that are rated by both user $u$ 
and user $v$. It's possible that this intersection be the empty set ($\emptyset$), 
given the sparsity of the matrix. It happens when user $u$ has not rated any movie 
that user $v$ has.
'''

# Question 9
from surprise import KNNBasic, KNNWithMeans
from surprise import Dataset
from surprise.model_selection.validation import cross_validate

k_lst = [2, 3, 5, 8, 12, 20, 30, 50, 70, 100]
