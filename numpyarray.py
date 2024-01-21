# check how many dimensions the array have: ndim attribute

# creating an array with 5-D and verify there it has five dimensions
'''import numpy as np
ashma = np. array([1, 3, 4, 5, 6], ndmin=5)
print(ashma)
print(ashma.ndim)
# checking the indexing in 1-D

ashma = np.array([1, 2, 3, 4])
print(ashma[2] + ashma[3])'''
# accessing the 2-D'''
'''import numpy as np
biswas = np.array([[1, 2, 3, 4], [6, 8, 9, 10]])
print("The second element on the first row is :", biswas[1, 3])
# accesing the 3-D is same as 2-D

ashma = np.array([[[1, 2, 3, 4], [5, 6, 7, 8]], [[7, 8, 9, 10], [11, 12, 13, 14]]])
print("The 3-D model is:", ashma[1, 1, 3])'''

# Array slicing: mean taking element from one given index to another index.
'''import numpy as np
ashma = np.array([1, 2, 3, 4, 5, 6, 7])
print(ashma[-3 :-1])'''
# steps: you will use steps value to determine the step of the slicing
# return every other value from index 1 to 5
'''import numpy as np
ashma = np.array([1, 2, 3, 4, 5, 6, 7])
print(ashma[1:5:2])'''
'''import numpy as np
ashma = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
print(ashma[::2])'''
# slicing 2-D array
'''import numpy as np
ashma = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
print(ashma[0:2, 2])'''
# print from both index 1-4
'''import numpy as np
ashma = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
print(ashma[0:2, 1:4])'''

'''
import numpy as np

ashma = np.array([[[2, 3, 4, 5, 6], [1, 8, 3, 5, 9]], [[11, 12, 13, 14, 15], [5, 7, 9, 3, 1]]])
print(ashma[0:2, 1:3, ])'''
'''
# checking the dat atype of numpy array -dtype
import numpy as np
ashma = np.array([1, 2, 3, 4, 5])
print(ashma.dtype)
# checking the data type of numpy array -string
import numpy as np
ashma = np.array(["apple", "banana", "orange"])
print(ashma.dtype)
# creating array with a defined data type
import numpy as np
ashma = np.array([1, 2, 3, 4, 5], dtype='S')
print(ashma)
print(ashma.dtype)
# now we will create a array with data type with 4 byte integer
import numpy as np
ashma = np.array([1, 2, 3, 4, 5], dtype='i4')
print(ashma)
print(ashma.dtype)
# if a type is given in which the element cannot be casted then numpy will raise error. what if the value cannot be converted
import numpy as np
ashma = np.array(['a', '2', '3'], dtype='i')
print(ashma)
print(ashma.dtype)'''
'''import numpy as np
ashma= np.array([1, 0, 2])
ashma = ashma.astype('bool')
print(ashma)
print(ashma.dtype)'''
# difference between numpy array copy and view
'''import numpy as np
ashma=np.array([1, 2, 3, 4, 5])
ashma1=ashma.copy()
ashma1[0]=12
print(ashma)
print(ashma1)
# now we will make the view, change orginal and display both
import numpy as np
ashma=np.array([1, 2, 3, 4, 5])
ashma1= ashma.view()
ashma[0]=42
print(ashma)
print(ashma1)'''
# the shape of an array is the number of elements in each dimensions.
# we will try to get the shape of each array.
'''import numpy as np
ashma = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
print(ashma.shape)'''
'''
# //Now we will create 5-D array using ndim
import numpy as np
ashma=np.array([1, 2, 3, 4], ndmin=5)
print(ashma)
print(ashma.shape)'''
'''
# Reshaping from 1-D to 2-D
import numpy as np
ashma = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
ashma1= ashma.reshape(4, 3)
print(ashma1)
print(ashma)'''
# Reshaping from 1D-3D
'''import numpy as np
ashma = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
ashma1 = ashma.reshape(2, 3, 2)
print(ashma1)'''
# retirn copy or view
'''import numpy as np
ashma = np.array([1, 2, 3, 4, 5, 6, 7, 8])

print(ashma.reshape(2, 4).base)'''
'''import numpy as np
ashma = np.array([1, 2, 3, 4, 5, 6, 7, 8])
ashma1= ashma.reshape(2, 2, -1)
print(ashma1)'''
# flattening the array by converting multidimesional array in 1-D
'''import numpy as np
ashma = np.array([[1, 2, 3], [4, 5, 6]])
ashma1= ashma.reshape(-1)
print(ashma1)'''
# Iterating array: Going elements one by one, like for loop
# Iterate the element of 1D
'''import numpy as np
ashma = np.array([1, 2, 3, 4, 5])
for i in ashma:
    print(i)'''
 # Iterate the element in 2D
'''import numpy as np
ashma = np.array([[1, 2, 3],  [4, 5, 6]])
for i in ashma:
    print(i)
# Iterate on each scalar element of the 2D
import numpy as np
ashma = np.array([[1, 2, 3], [4, 5, 6]])
for i in ashma:
    for a in i:
        print(a)'''
 # iterate 3D
'''import numpy as np
ashma = np.array([[[2, 3, 4],[5, 6, 7]], [[6, 7, 8], [9, 10, 11]]])
for i in ashma:
    for a in i:
       for b in a:
           print(b)'''
# Iterating using nditer() function, now we will Iterate on each scalar element
'''import numpy as np
ashma = np.array([[[1, 2], [3, 4], [5, 6], [7, 8]]])
for i in np.nditer(ashma):
    print(i)
# Now we will iterate with different step size
import numpy as np
ashma = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
for i in np.nditer(ashma[:, ::2]):
    print(i)'''

# joining the numpy array, here we will pass concatenate
'''import numpy as np
ashma = np.array([1, 2, 3])
ashm1 = np.array([4, 5, 6])
ashma2=np.concatenate((ashma,ashm1))
print(ashma2)
# joining of 2-D arrays along with rows(axis = 1)
import numpy as np
ashma = np.array([1, 2, 3])
ashma1 = np.array([5, 6, 7])
ashma2= np.stack((ashma, ashma1), axis=1)
print(ashma2)'''
# Stacking along with rows
'''import numpy as np
ashma = np.array([1, 2, 3])
ashma1 = np.array([5, 6, 7])
ashma2 = np.hstack((ashma, ashma1))
print(ashma2)
# Stacking along with width
import numpy as np
ashma = np.array([1, 2, 3])
ashma1 = np.array([5, 6, 7])
ashma2 = np.dstack((ashma, ashma1))
print(ashma2)'''

# Array splitting in Numpy: It is reverse to joining
# array_split()
# split array into 3 parts
'''import numpy as np
ashma = np.array([1, 2, 3, 4, 5, 6])
ashma1 = np.array_split(ashma, 4)
print(ashma1)
# split into array
import numpy as np
ashma = np.array([1, 2, 3, 4, 5, 6])
ashma1 = np.array_split(ashma, 3)
print(ashma1)'''
# splitting 2D array
'''import numpy as np
ashma = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
ashma1 = np.array_split(ashma, 3)
print(ashma1)
# split the 2D array into three 2D array
import numpy as np
ashma = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]])
ashma1 = np.array_split(ashma, 3)
print(ashma1)'''
# splitting the 2D into three 2D along with rows
'''import numpy as np
ashma = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]])
ashma1 = np.array_split(ashma, 3, axis=1)
print(ashma1)'''
# alter soln is using hsplit(), opposite hstack()
'''import numpy as np
ashma = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]])
ashma1 = np.hsplit(ashma, 3)
print(ashma1)'''
# Numpy Searching array: You can search an array for a certain value and return the indexes that get the match by using where()
'''import numpy as np
ashma= np.array([1, 2, 3, 4, 5, 4, 4])
ashma1 = np.where(ashma==4)
print(ashma1)'''
# now we will find the indexes where the value are even
'''import numpy as np
ashma = np.array([1, 2, 3, 4, 5, 6, 7, 8])
ashma1 = np.where(ashma%2 ==1)
print(ashma1)'''
# Searchsorted(): perform binary search in array in return gives index
# we will find the index where the value 7 should be inserted
'''import numpy as np
ashma = np.array([6, 7, 8, 9])
ashma1 = np.searchsorted(ashma, 9, side="right")
print(ashma1)
# how to search multiple values
ashma = np.array([1, 3, 5, 7])
ashma1 = np.searchsorted(ashma, [2, 4, 6])
print(ashma1)'''
# Numpy Sorting Array:
'''import numpy as np
ashma = np.array([1, 3, 8, 5])
print (ashma)
print(np.sort(ashma))
# Numpy sorting alphabetically
# Numpy Sorting Array:
import numpy as np
ashma = np.array(["mango", "Apple", "Banana", "orange"])
print (ashma)
print(np.sort(ashma))
# Sort the boolean value
# Numpy Sorting Array:
import numpy as np
ashma = np.array(["True", "False", "True"])
print (ashma)
print(np.sort(ashma))'''
# sort 2Darray
# Numpy Sorting Array:
'''import numpy as np
ashma = np.array([[8, 3, 7], [9, 7, 5]])
print (ashma)
print(np.sort(ashma))'''
# Numpy filtering the Array:
# A boolean index list is a list of boolean corresponding to indexes in the array. (True and False)
# Create an array from the element on index 0 to 2:
# Numpy Sorting Array:
'''import numpy as np
ashma = np.array([40, 44, 47, 49])
ashma1= [True, False, True, False]
ashma3 = ashma[ashma1]
print(ashma3)'''
# now will create filter array, that will return values higher than 44
'''import numpy as np
ashma = np.array([40, 44, 47, 49])
ashma1 = []
for element in ashma:
    if element > 44:
        ashma1.append(True)
    else:
        ashma1.append(False)
ashma2 = ashma[ashma1]
print(ashma1)
print(ashma2)
# create a filter array that will return only even elements from the oginal array
import numpy as np
ashma = np.array([40, 44, 47, 49])
ashma1=[]
for i in ashma:
    if i%2 ==0:
        ashma1.append(True)
    else:
        ashma1.append(False)
print(ashma1)'''
# Yes you can create filter directly from array
'''import numpy as np
ashma = np.array([41, 42, 43, 44, 45])
ashma1= ashma > 42
ashma2 = ashma[ashma1]
print(ashma2)

# it can also be implemented to find even value
import numpy as np
ashma = np.array([1, 2, 3, 4, 5, 6, 7, 8])
ashma1 = ashma%2 ==0
ashma2 = ashma[ashma1]
print(ashma2)'''
# Numpy Random Number: Random means the number that cannot be logically predict
# now we are getting to generate random number  from 0 t0 100
'''from numpy import random
ashma = random.randint(100)
print(ashma)'''
# you can also generate random array- 2D array containing 5 random int
'''from numpy import random
ashma = random.randint(100, size=(3, 5))
print(ashma)

# we will generate 1-D array containiong 5 random float:
from numpy import random
ashma = random.rand(3, 5)
print(ashma)'''
# we can also generate random numbers from an array
'''from numpy import random
ashma = random.choice([3, 5, 8, 1, 4, 6])
print(ashma)'''
# we can aslo generate random number from 2D array
'''from numpy import random
ashma = random.choice([3, 5, 8, 1, 4, 6], size=(3, 5))
print(ashma)'''

# Numpy Random Data Distribution: it is the list of all possible value and how often each value occurs, it is used in data science and statistics
# Random Distribution: probability function. Now we will generate 1D array with 100 values where every value has to be 3, 5, 7, 9
# The probability for the value 3 is set to be 0.1
# The probability for the value 5 is set to be 0.3
# The probability for the value 7 is set to be 0.6
# The probability for the value 9 is set to be 0
# The sum of all probability number should be 1
'''from numpy import random
ashma = random.choice([3, 5, 7, 9], p= [0.1, 0.3, 0.6, 0.0], size=(100))
print(ashma)

# now we will return 2D with 3 rows each containing 5 values
from numpy import random
ashma = random.choice([3, 5, 7, 9], p= [0.1, 0.3, 0.6, 0.0], size=(3, 5))
print(ashma)'''

# Permutation: Arrangement of elements like [3,2,1] is permutaion of [1,2,3] and vice-versa
# The numpy random module provides 2 methods: shuffle(), permutation()
# Now we will randomly shuffle elements of the below array
'''from numpy import random
import numpy as np
ashma = np.array([5, 7, 8, 3, 2, 11])
random.shuffle(ashma)
print(ashma)'''

# Now we will generate permutation of elements of the below array
'''from numpy import random
import numpy as np
ashma = np.array([5, 7, 8, 3, 2, 11])
print(random.permutation(ashma))'''

# Matplotlib(pyplot)- Seaborn|| Visualize Distn with Seaborn
# Distplot- Distribution plot(curve plot- histogram)
'''import matplotlib.pyplot as plt
import seaborn as sns
sns.distplot([1, 2, 3, 7, 9,5], hist=False)
plt.show()'''

# Numpy Normal Distribution: Very important, also invented by some gaussian named scientist
# random.normal() method- loc, scale, size
# we are generating a random normal distn of size 2*3
'''from numpy import random
ashma = random.normal(size=(2,3))
print(ashma)

# here we will genrate random normal distn of size 2*3 with loc(mean) at 1 and scale(standard deviation)
from numpy import random
ashma = random.normal(size=(2,3), loc=1, scale=2)
print(ashma)

# Visualization of normal Distribution
from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns
sns.distplot(random.normal(size=1000), hist=False)
plt.show()'''

# Numpy Bionomial Distribution: Descrete Distn- binary output
# three parameters: n= no of trails, p= probability, s= size
# Given 10 trail for a coin which will generatev 10 data points
'''from numpy import random
ashma = random.binomial(n=10, p=0.5, size=10)
print(ashma)'''

'''
# visualization of bionomial distn
from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns
sns.distplot(random.binomial(n=10, p=0.5, size=1000), hist=True, kde=False)
plt.show()'''

# Numpy for Poisson Distribution
# generate a random 1*10 dist for the occurance of 2
'''from numpy import random
ashma = random.poisson(lam=2, size=10)
print(ashma)'''

# Visualization of poisson distn
'''from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns
sns.distplot(random.poisson(lam=2, size=1000), hist=True, kde=False)
plt.show()'''

# Presenting both the plot in same figure i.e normal and poisson
'''from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns
sns.distplot(random.normal(loc=50, size=1000, scale=7), hist=False, label='Normal')
sns.distplot(random.poisson(lam=50, size=1000), hist=False, label='Poisson')
plt.show()'''

# Uniform Distribution:
'''from numpy import random
ashma = random.uniform(size=(2,3))
print(ashma)

# Visualization of uniform distn
import matplotlib.pyplot as plt
from numpy import random
import seaborn as sns
sns.distplot(random.uniform(size=1000), hist=False)
plt.show()'''

# Logistic Distribution
# draw 2*2 sample of logistic where mean at 1, S.D is 2.0
'''from numpy import random
ashma = random.logistic(loc=1, scale=2, size=(2,3))
print(ashma)
# visualization of logistic distn
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import random
sns.distplot(random.logistic(size=1000), hist=False)
plt.show()'''

# Multinominal Distribution: It has three parameter, n= no. of outcomes/possibility, pvals- list of possibilities or outcome
# draw out the sample for dice roll
'''from numpy import random
ashma = random.multinomial(n=6, pvals=[1/6,1/6,1/6,1/6,1/6,1/6])
print(ashma)'''

# Exponential Distribution: It is used for describing time till next event, that is like failure or sucess.
# Here we will draw a sample for exponential distribution with 2.0 scale and 2*3 size
'''from numpy import random
ashma = random.exponential(scale=2.0, size=(2,3))
print(ashma)

# Visualization of exponential distn
from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns
sns.distplot(random.exponential(size=1000), hist=False)
plt.show()'''

# chi-square Distribution: it is basically used as a basis to verify the hypothesis
# it has two param: df(degree of freedom) and size
# sample for chi-square dist with df=2, size=2*3
'''from numpy import random
ashma= random.chisquare(df=2, size=(2,3))
print(ashma)
# Visulization of chi-square
from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns
sns.distplot(random.chisquare(df=1,size=1000), hist=False)
plt.show()'''

# Numpy Rayleigh Distn: It is basically used for signal processing in ML, it has two param scale(S.D, how the flat the distribution is)-default is 1.0 and the size
# Sample for Rl with scale 2.0 with size 2*3
'''from numpy import random
ashma = random.rayleigh(scale=2.0,size=(2,3))
print(ashma)
# visualization for Rayleign
from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns
sns.distplot(random.rayleigh(scale=2.0, size=1000), hist=False)
plt.show()'''

# Pareto Distribution: It is 80:20 rule, where 20% factors cause 80% outcomes
# sample for pareto Distn using shape=2, size= 2*3
'''from numpy import random
ashma = random.pareto(a=2, size=(2,3))
print(ashma)

# Visualization of Pareto Distn
from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns
sns.distplot(random.pareto(a=2, size=1000), kde=False)
plt.show()'''

# Zipf Distn: its defn is like.. common words in english has occurs 1/5 times as of the most nepali words
# there are two param(a=distn param and size)
# sample for zipf distn having a as 2 and size as 2*3
'''from numpy import random
ashma = random.zipf(a=2, size=(2,3))
print(ashma)
# To visualize zipf distn
from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns
ashma = random.zipf(a=2, size=1000)
sns.distplot(ashma[ashma<10], kde=False)
plt.show()'''

# Numpy Universal function i.e ufuncs: they are actually numpy function that operates on ndarray objects.
# ufuncs also takes additional arguments like where, dtype, out
# vectorization: converting the iterative statements into vector based statements
# Examples without ufunc, here we will use python in build zip()
'''x =[1,2,3,4]
y=[4,5,6,7]
z=[]
for i,j in zip(x,y):
    z.append(i+j)
print(z)
# Using through ufuncs(), we will use add function
import numpy as np
x = [1,2,3,4]
y = [4,5,6,7]
z = np.add(x,y)
print(z)'''

# To create your own ufuncs, you have to define a function like you do in normal function in python, and you add it to the numpy function with frompyfunc() method.
# arguments frompyfuncs(): functions, inputs, outputs
# Create your own ufuncs for addition
'''import numpy as np
def myadd(x,y):
    return x+y

myadd= np.frompyfunc(myadd,2,1)
print(myadd([1,2,3,4], [5,6,7,8]))'''

# arithmetic ufuncs
# by using ufuncs additional also takes additional arguments like where, dtype, out
# here now we will use add()
'''import numpy as np
ashma1=np.array([10,11,12,13,14,15])
ashma2=np.array([20,21,22,23,24,25])
ashma3=np.add(ashma1,ashma2)
print(ashma3)'''

# Numpy Rounding Decimals||trunc(),fix(),around(), floor(), ceil()
# here we are truncating the below array
'''import numpy as np
ashma= np.trunc([-3.666789, 4.234567])
print(ashma)
# Fix()
import numpy as np
ashma= np.fix([-3.666789, 4.234567])
print(ashma)'''

# around()
'''import numpy as np
ashma = np.around(0.89765,2)
print(ashma)
# floor()
import numpy as np
ashma = np.floor([3.7678, -4.3768])
print(ashma)
# Ceil()
import numpy as np
ashma = np.ceil([3.7678, -4.3768])
print(ashma)'''

# Summations: difference: addition betn 2 arguments whereas summation happens over an elements
# adding the 2 array
'''import numpy as np
ashma1 = np.array([2,3,4])
ashma2= np.array([4,5,6])
ashma3=np.add(ashma2,ashma1)
print(ashma3)

# sum the values in 2-array
import numpy as np
ashma1 = np.array([2,3,4])
ashma2= np.array([4,5,6])
ashma3=np.sum([ashma2,ashma1])
print(ashma3)
# summation over an axis: if you specify axis=1, numpy will sum the number in each array
import numpy as np
ashma1 = np.array([2,3,4])
ashma2= np.array([4,5,6])
ashma3=np.sum([ashma2,ashma1], axis=1)
print(ashma3)

# cummulation sum
import numpy as np
ashma1 = np.array([2,3,4])
ashma2 = np.array([4,5,6])
ashma3 = np.cumsum([ashma1,ashma2])
print(ashma3)'''
# product and cummulative product
'''import numpy as np
ashma1 = np.array([2,3,4])
ashma2 = np.array([4,5,6])
ashma3 = np.cumprod([ashma1,ashma2])'''
# product with axis=1
'''import numpy as np
ashma1 = np.array([2,3,4])
ashma2 = np.array([4,5,6])
ashma3 = np.prod([ashma1,ashma2], axis=1)
print(ashma3)
print(ashma3)
# product using prod()
import numpy as np
ashma1 = np.array([2,3,4])
ashma2 = np.array([4,5,6])
ashma3 = np.prod([ashma1,ashma2])
print(ashma3)
# Difference
import numpy as np
ashma1 = np.array([9,4,18,24])
ashma3 = np.diff([ashma1])
print(ashma3)'''

# LCM(Lowest Common Multiple):
# here we will find the LCM of the 2 number
'''import numpy as np
ashma = 8
ashma1 = 12
ashma3=np.lcm(ashma,ashma1)
print(ashma3)
# The lcm above is 24 because the lcm of both number(8 and 12)
# finding the LCM in array
import numpy as np
ashma1 = np.array([2,3,4])
ashma3 = np.lcm.reduce(ashma1)
print(ashma3)
# We will find the lcm of all of an array where the array contains all integers from 1 to 10
import numpy as np
ashma = np.arange(1,11)
ashma1= np.lcm.reduce(ashma)
print(ashma1)'''

# GCD/HCF
'''import numpy as np
ashma = 8
ashma1 = 12
ashma3=np.gcd(ashma,ashma1)
print(ashma3)
# GCD in the below array
import numpy as np
ashma1 = np.array([8,14,24])
ashma3 = np.gcd.reduce(ashma1)
print(ashma3)'''

# Trigonometric Function: sin(), cos(), tan()
# Here now we will find the value of pi/2
'''import numpy as np
ashma = np.sin(np.pi/4)
print(ashma)
# We will now find the sin values of an array
import numpy as np
ashma = np.array([np.pi/2, np.pi/3, np.pi/4, np.pi/5])
ashma1 = np.sin(ashma)
print(ashma1)

# convert degree into radians: by default all of the trigonometric function takes radians as parameters.
# radians values are pi/180* degree value
# here we will convert all the array values to radians:

import numpy as np
ashma = np.array([90,180,270,360])
ashma1 = np.deg2rad(ashma)
print(ashma1)
# here we will convert radians into degree
import numpy as np
ashma = np.array([np.pi/2,np.pi,1.5*np.pi, 2*np.pi])
ashma1 = np.rad2deg(ashma)
print(ashma1)
# here we can also find angles: arcsin(), arccos(), arctan() that takes radians and produce the corresponding sin, cos, tan values
# we will now find the angle of 1.0
import numpy as np
ashma = np.array(1.0)
ashma1 = np.arcsin(ashma)
print(ashma1)
# we will find the angles of each values in an array
import numpy as np
ashma = np.array([-1,1,0.1])
ashma1 = np.arcsin(ashma)
print(ashma1)
# here we can also find the hypotenous using pythagoras theorem:
# hypot() function that takes radians and produce the corresponding sin, cos, tan values
# here we will find the hypot for  3 base and 4 perpendicular
import numpy as np
base=3
perp=4
ashma=np.hypot(base,perp)
print(ashma)'''
# Hyperbolic Function: numpy provides the ufuncs line sinh,cosh,tanh() that takes the value in radian and produce the corresponding sin, cos, and tan values.
# It is a actually inverse function
# Here we will find the value of sinh of pi/2
'''import numpy as np
ashma = np.sinh(np.pi/2)
print(ashma)
# we will now find values in array
import numpy as np
ashma1= np.array([np.pi/2, np.pi/3, np.pi/4, np.pi/5])
ashma = np.cosh(ashma1)
print(ashma)'''
# finding angles in arcsinh, arccosh, arctanh(): it is same as trigonometry angle
'''import numpy as np
ashma = np.arcsinh(1.0)
print(ashma)
# finding angle in array
import numpy as np
ashma1= np.array([0.1,0.2,0.3])
ashma = np.arctanh(ashma1)
print(ashma)'''

# Set operation: collection of unique elements
# to find the unique elements from any array:
# here will convert the array with repeated elements to a set
import numpy as np
ashma = np.array([1,1,1,3,4,4,5,5,7,8,6])
ashma1= np.unique(ashma)
print(ashma1)
# to find the unique value of 2 1D array, we will use union() method:
import numpy as np
ashma = np.array([1,4,5,7])
ashma1= np.array([2,5,6,7])
ashma2= np.union1d(ashma, ashma1)
print(ashma2)
# To find the intersection
import numpy as np
ashma = np.array([1,4,5,7])
ashma1= np.array([2,5,6,7])
ashma2= np.intersect1d(ashma, ashma1, assume_unique=True)
print(ashma2)
# To find set diffrence using setdif1d();
import numpy as np
ashma = np.array([1,4,5,7])
ashma1= np.array([2,5,6,7])
ashma2= np.setdiff1d(ashma, ashma1)
print(ashma2)
# To find the only values that are not present in both the sets, use setxor1d() method:
import numpy as np
ashma = np.array([1,4,5,7])
ashma1= np.array([2,5,6,7])
ashma2= np.setxor1d(ashma, ashma1)
print(ashma2)




























