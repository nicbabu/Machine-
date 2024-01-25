# Machine Learning: It is making the computer to learn from studying data and statistics
# This the program analyzes the data, based on that data it learns to predict the outcomes.
# where to start: Here for calculation important numbers based on the data sets we use mathematics statistics and using the various python module
# data sets: [99,86,87,88,111,86,103,87,94,78,77,85,86]
# data type: There are three types of main category of data type:
# numerical: Discrete Data( limited to integers, eg no of car passing in road), Continious data: Number that are of infinite value like price of an item.
# categorical Data: Data which cannot count or measure nor value but it has value yes?No(i.e boolean), eg color
# Ordinal Data: It's like categorical data but it can be measured or countable like School grades where A is better Than B

# Mean, Median , Mode
# Mean: The average value
# Median : the mid-value
# Mode: the most common value
# Thirteen days car speed: [99,86,87,88,111,86,103,87,94,78,77,85,86]
# Here in this example we have speed of 13 cars
'''import numpy as np
speed = [99,86,87,88,111,86,103,87,94,78,77,85,86]
mean = np.mean(speed)
print(mean)

# To calculate mode: We use scipy
from scipy import stats
speed = [99,86,87,88,111,86,103,87,94,78,77,85,86]
mean = stats.mode(speed)
print(mean)

# Standard Deviation:(low-std)
import numpy as np
speed = [1,2,3,4,6,4,3]
# there are two type of SD, low-SD(Distance is small, whereas the high-SD whose distances are far)
sd = np.std(speed)
print(sd)
# for high-SD
import numpy as np
speed =[32,116,24,45,56]
sd1 = np.std(speed)
print(sd1)

# Variance: How the values are spread, it is in square-root. sd^2 = variance
import numpy as np
speed =[32,116,24,45,56]
sd1 = np.var(speed)
print(sd1)

# Percentile:
import numpy as np
ages = [45,23,13,14,54,33,37,26,54,23,41,80]
# the model has method to finding the specified percentiles:
percent = np.percentile(ages, 90)
print(percent)

# Data Distribution: How to we get big data set?
# here now we will create a array containing 250 random floats between 0 to 5
import numpy as np
import matplotlib.pyplot as plt
ashma = np.random.uniform(0.0,5.0,100000)
print(ashma)
plt.hist(ashma,100)
plt.show()

# Normal Data Distribution: In bumpy probability theory this kind o data distribution is known as normal data distribution or Gaussian Data Distn, where the value are concentrated around a given value.
# Example of normal data distn.
import numpy as np
import matplotlib.pyplot as plt
ashma = np.random.normal(5.0,1.0,100000)
plt.hist(ashma,100, color="red")
plt.show()

# Scatter Plot: A scatter Plot is a diagram where each value in the data set represented by the dots.
# It basically needs 2-Array of the same length: one for X-axis and another for Y-axis
# Car life year and its speed: Here x array represents the age of each car and the y array represent the speed of each car
import matplotlib.pyplot as plt
age = [2,4,7,4,9,7,10,12,13,15]
speed = [65,78,110,65,112,134,152,149,178,190]
plt.scatter(age, speed, color='red')
plt.xlabel("Car Age")
plt.ylabel("Speed")
plt.show()

# Now we will do the above via random data distn. 1000 random number from normal data distn, the 1st array will have the mean set 5.0 with the standard deviation 1.0 and the 2nd array will have the mean set to 10.0 with the standard deviation of 2.0
import numpy as np
import matplotlib.pyplot as plt
x = np.random.normal(5.0,1.0,1000)
y = np.random.normal(10.0,2.0,1000)
plt.scatter(x,y,100, color="Green")
plt.colorbar
plt.show()

# Regression: It is used to find out the relationship between the variables. In machine learning and in statistical modeling that relationship is used to predict the outcomes of the future event.
# linear Regression: It is basically uses the relationship between the data points to draw a straight lines through all of them. This line can be used to predict the future.
# How does it works:
# now we will import scipy(scientific python) and draw the line of linear regression
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
age = [2,4,7,4,9,7,10,12,13,15]
speed = [65,78,110,65,112,134,152,149,178,190]
slope,intercept,r,p,std_err = stats.linregress(age,speed)
def myfunc(age):
    return slope * age + intercept
mymodel = list(map(myfunc,age))
plt.scatter(age,speed)
plt.plot(age,mymodel)
plt.show()

# defining r for relationship
# It is important to knoew the relationship between the value of x-axis and y-axis, if there is no relation then the linear regression cannot be used to predict anything. Then this reln is called R or also called coefficient of correlation.
# The value ranges from -1 to 1. if the value is 0, it means no relation whereas -1,1 mean 100% relation.
# now we will know how well does the data fit in a linear regression
from scipy import stats
age = [2,4,7,4,9,7,10,12,13,15]
speed = [65,78,110,65,112,134,152,149,178,190]
slope,intercept,r,p,std_err = stats.linregress(age,speed)
print(r)
import matplotlib.pyplot as plt
# predict future values: lets try to predict the speed of 10 years old cars
from scipy import stats
age = [2,4,7,4,9,7,10,12,13,15]
speed = [65,78,110,65,112,134,152,149,178,190]
slope,intercept,r,p,std_err = stats.linregress(age,speed)
def myfunc(age):
    return slope * age + intercept
speeds_of_car = myfunc(10)
print(speeds_of_car)

# now if we have the bad fit: lets create an example where linear regression would not be the best method for predicting the future
import matplotlib.pyplot as plt
from scipy import stats
x = [89,112,45,67,70,32,10,5,39,34,29,26,72,40,78]
y = [34,90,21,15,30,66,87,56,95,53,72,58,27,38,2]
slope,intercept,r,p,std_err = stats.linregress(x,y)
def myfunc(x):
    return slope * x + intercept
mymodel = list(map(myfunc,x))
print(r) # this is for cofficient correlation, # now we will check the relationship between R
plt.scatter(x,y)
plt.plot(x,mymodel)
plt.show()

# Polynomial Regression: your data points clearly will not fit a linear regression(straight line) then it might be ideal for a polynomial regresssion.
# It also used the relationship between the x and y to find the best way to draw a line through data points.
# How does it works? Example: Here we will register some(18) cars as they were passing through certain toolboth.
import matplotlib.pyplot as plt
x = [1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22]
y = [100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]
plt.scatter(x,y)
plt.show()
import numpy
# now here we will import numpy and matplotlib and we will draw the line of polynomial regression
import numpy as np
import matplotlib.pyplot as plt
x = [1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22]
y = [100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]
mymodel = np.poly1d(np.polyfit(x,y, 3))
myline = np.linspace(1,22,100)
plt.scatter(x,y)
plt.plot(myline,mymodel(myline))
plt.show()

# R-Squared: If there is no relationship between the x and y then polynomial regression cannot predict anything.
# Here the relationship is measured with the value called R-squared. It's ranges from 0 to 1, where 0 is no relation 1 is 100 related.
# Python and the sklearn module will compute this value for you
import numpy as np
from sklearn.metrics import r2_score
x = [1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22]
y = [100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]
mymodel = np.poly1d(np.polyfit(x,y, 3))
print(r2_score(y,mymodel(x))) # the value of R-squared is in the range i.e 0.96 near by the 1
# Now to predict the future value. Here we will predict the speed of a car passing at 17 PM(as we don't have 5 PM data so)
import numpy as np
from sklearn.metrics import r2_score
x = [1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22]
y = [100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]
mymodel = np.poly1d(np.polyfit(x,y, 3))
speed = mymodel(17)
print(speed)

# What about the bad fit?
# let's create an example where polynomial regression would not be the best method to predict the future values.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
x = [56,78,24,14,17,10,32,31,45,49,62,76,15,38,44,25,29,15]
y = [46,95,53,3,35,72,10,26,34,90,33,20,56,5,2,9,19,7]
mymodel = np.poly1d(np.polyfit(x,y, 3))
myline = np.linspace(10,95,100)
plt.scatter(x,y)
plt.plot(myline,mymodel(myline))
plt.show()

# its r-squared value
x = [56,78,24,14,17,10,32,31,45,49,62,76,15,38,44,25,29,15]
y = [46,95,53,3,35,72,10,26,34,90,33,20,56,5,2,9,19,7]
mymodel = np.poly1d(np.polyfit(x,y, 3))
print(r2_score(y,mymodel(x)))'''
import numpy.random

# Multiple Regression: It is like linear regression but with more than one independent values. It means that we try to predict the values based on two or more variables.
# How does the whole process works:
# for this we will start from importing the pandas modules. Example of kaggle car module datasets
# we import data using pandas
# we will make a list of independent variables and call this variable x, put the dependent value in a variable called y.
# car_independent = df[['Weight','Volume']]
# car_dependent = df['Co2']
# now we will use some methods from sklearn module, so we will have to import the module as well.
# from sklearn import linear_model
# from sklearn module we will use the LinearRegression() method to create a linear regression object
# this objects has a method called fit() that takes the independent and dependent values as a parameter and fills the regression objects with the data that describes the relationship.
# regr = linear_model.LinearRegression()
# regr.fit(x,y)
# now we will predict the co2 of a car where the weight is 23okg and the volume is 1300
# predictedCO2 = regr.predict([[2300,1300]])
'''import pandas as pd
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
df = pd.read_csv("DATA.csv")
car_independent = df[['Weight','Volume']]
car_dependent = df['CO2']
regr = linear_model.LinearRegression()
regr.fit(car_independent,car_dependent)
predicted_score = regr.predict([[2300,1300]])
print(predicted_score)

# coefficient : factors that describes the relation between unknown variable.
# example: if x is a variable then 2x is x two times. x is unknown variable and number 2 is coefficient.
# Here will print the coefficient value of the regression object
import pandas as pd
from sklearn import linear_model
df = pd.read_csv("DATA.csv")
car_independent = df[['Weight','Volume']]
car_dependent = df['CO2']
regr = linear_model.LinearRegression()
regr.fit(car_independent,car_dependent)
print(regr.coef_)
# Explaning the output above: [0.00755095 0.00780526] i.e [weight, volume]
# it tells if the weights is increased by 1kg then the co2 emission increases by 0.00755095, and if the volume of the engine increases by 1 cm then the co2 coefficient increased by 0.00780526

import pandas as pd
from sklearn import linear_model
df = pd.read_csv("DATA.csv")
car_independent = df[['Weight','Volume']]
car_dependent = df['CO2']
regr = linear_model.LinearRegression()
regr.fit(car_independent,car_dependent)
# now here we will predict CO2 of a car where the weight is 2300kg and the volume is 1300
predictedCO2 = regr.predict([[3300,1300]])
print(predictedCO2)
# for the caln we will show that the coefficient of 0.00755095 and the answerwill be below:
# 107.2087328 + (1000*0.00755095)= 114.75968007

# Scale Feature| Standarization: When your data has different values and even diffrent measurements unit, then itcan be difficult to compare there, so the answer to this probelm is scaling and scale function.
# There are different method for scaling the data. Here we will use the method called standarization.
# Formula for standarization : z = (x-u)/s where z is the new value, x is the orginal value, u is mean and s is standard deviation.
import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
df = pd.read_csv('DATA_1.csv')
x = df[['Weight','Volume']]
scaleedx= scale.fit_transform(x)
print(scaleedx)


# now with above result we will predict the co2 emission. Here we will print the co2 from a 1.3 lt car that weight is 2300kg.
import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
df = pd.read_csv('DATA_1.csv')
x = df[['Weight','Volume']]
y = df['CO2']
scaleedx= scale.fit_transform(x)
regr = linear_model.LinearRegression()
regr.fit(scaleedx,y)
scaled = scale.transform([[2300,1.3]])
predictedco2 = regr.predict([scaled[0]])
print(predictedco2)

# Training and Testing the data
# In machine learning we create a model to predict the outcome, to measure of the model is good enough or need to rectify, than we use a method called train/test.
# what is train/test : 80% training and 20% testing
# understand with a proper dataset.
# here for understanding we use a dataset that shows the 100 customers in ashop and their shopping habits.
import numpy as np
import matplotlib.pyplot as plt
np.random.seed()
ashma1 = numpy.random.normal(3,1,100)
ashma2 = numpy.random.normal(150,40,100)/ashma1
plt.scatter(ashma1,ashma2)
plt.show()
# results: x-axis represents the number of minutes before making the purse and y-axis represents the amount of the money spend on the purchase.
# Next is splitting into train/test
# train_x = x[:80]
# train_y = y[:80]
# test_x = x[80:]
# test_y = y[80:]
# how to display the training set: plt.scatter(train_x, train_y) and show the plt
import numpy as np
import matplotlib.pyplot as plt
np.random.seed()
x = numpy.random.normal(3,1,100)
y= numpy.random.normal(150,40,100)/x
train_x = x[:80]
train_y = y[:80]
test_x = x[80:]
test_y = y[80:]
plt.scatter(train_x,train_y)
plt.show()
# now we will display the testing set:
import numpy as np
import matplotlib.pyplot as plt
np.random.seed()
x = numpy.random.normal(3,1,100)
y= numpy.random.normal(150,40,100)/x
train_x = x[:80]
train_y = y[:80]
test_x = x[80:]
test_y = y[80:]
plt.scatter(test_x,test_y)
plt.show()

# what about fit the data set.
# here we will draw a polynomial regression line through the data point.
import numpy as np
import matplotlib.pyplot as plt
np.random.seed()
x = numpy.random.normal(3,1,100)
y= numpy.random.normal(150,40,100)/x
train_x = x[:80]
train_y = y[:80]
test_x = x[80:]
test_y = y[80:]
mymodel = np.poly1d(np.polyfit(train_x,train_y,4)) 
myline = np.linspace(0,6,100)
plt.scatter(train_x,train_y)
plt.plot(myline,mymodel(myline))
plt.show()
# here we will use R-suared(r2)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
np.random.seed()
x = np.random.normal(3,1,100)
y= np.random.normal(150,40,100)/x
train_x = x[:80]
train_y = y[:80]
test_x = x[80:]
test_y = y[80:]
mymodel = np.poly1d(np.polyfit(train_x,train_y,4))
r2 = r2_score(train_y,mymodel(train_x))
print(r2)
# now we will use this for testing set
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
np.random.seed()
x = np.random.normal(3,1,100)
y= np.random.normal(150,40,100)/x
train_x = x[:80]
train_y = y[:80]
test_x = x[80:]
test_y = y[80:]
mymodel = np.poly1d(np.polyfit(train_x,train_y,4))
r2 = r2_score(test_y,mymodel(test_x))
print(r2)
# now we will predict the values:
# In the below example we will see how much money will a customer spent if he or she in the shop stay for 5min
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
np.random.seed()
x = np.random.normal(3,1,100)
y= np.random.normal(150,40,100)/x
train_x = x[:80]
train_y = y[:80]
test_x = x[80:]
test_y = y[80:]
mymodel = np.poly1d(np.polyfit(train_x,train_y,4))
print(mymodel(5))

# Confusion Matrix: it is a table that is used in classification problems to acess where eroors in the model where made.
# Here the rows represents the actual classes rhe outcomes should have been, while the columns represents the prediction.
# Now we will create a confusion matrix
# It is always made with logistic regression

import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

actual_data = np.random.binomial(1, .9, 1000)
predicted = np.random.binomial(1,.9,1000)
confusion_matrix = metrics.confusion_matrix(actual_data,predicted)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix,display_labels= [False, True])
cm_display.plot()
plt.show()
# True means the values are accurately predicted or the false means there was an error or wrongly predicted.

# Created Matrics:
# accuracy: It measures how often the model is correct.
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
actual_data = np.random.binomial(1, .9, 1000)
predicted = np.random.binomial(1,.9,1000)
accuracy = metrics.accuracy_score(actual_data, predicted)
print(accuracy)

# Precision:
# of the positive predicted, what is the percentage of truly positive?
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
actual_data = np.random.binomial(1, .9, 1000)
predicted = np.random.binomial(1,.9,1000)
precision = metrics.precision_score(actual_data, predicted)
print(precision)

# Sensetivity (recalling): of all the positive cases, what is the percentage of the positive prediction.
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
actual_data = np.random.binomial(1, .9, 1000)
predicted = np.random.binomial(1,.9,1000)
recall = metrics.recall_score(actual_data, predicted)
print(recall)

# Specificity: How well the model is at predicting the negative result.
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
actual_data = np.random.binomial(1, .9, 1000)
predicted = np.random.binomial(1,.9,1000)
specificity = metrics.recall_score(actual_data, predicted, pos_label=0)
print(specificity)

# f-score:  it is the harmonic mean of the precision and sensitivity. It consider both false positive and false negative cases.
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
actual_data = np.random.binomial(1, .9, 1000)
predicted = np.random.binomial(1,.9,1000)
recall = metrics.recall_score(actual_data, predicted)
f1score= metrics.f1_score(actual_data, predicted)
print(f1score)
# How to use all of them in a single code?
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
actual_data = np.random.binomial(1, .9, 1000)
predicted = np.random.binomial(1,.9,1000)
accuracy = metrics.accuracy_score(actual_data, predicted)
precision = metrics.precision_score(actual_data, predicted)
recall = metrics.recall_score(actual_data, predicted)
specificity = metrics.recall_score(actual_data, predicted, pos_label=0)
f1score= metrics.f1_score(actual_data, predicted)
specificity = metrics.recall_score(actual_data, predicted, pos_label=0)
print({"Accuracy":accuracy,"Precision":precision,"Recall":recall,"Specificity":specificity,"F1Score":f1score})'''

# Hierarchy Clustering: It is an unsupervised learning method for clustering data points. The algorithm builds clusters by measuring the data dissimilarities between the data.
# Example for the same:
# three lines to make our compiler able to draw.
# now we will compute the ward linkage method using euclidean distance and visualize this through the dendogram.
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram,linkage
x = [4,5,10,4,2,1,6,10,12,15]
y = [21,19,24,17,16,25,24,22,21,21]
data = list(zip(x,y))
linkage_data = linkage(data,method='ward', metric='euclidean')
dendrogram(linkage_data)
plt.show()
# Yes we can do, the same thing with the python scikit-learn and visualize on a 2D plot
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering

x = [4, 5, 10, 4, 3, 11, 14, 6, 10, 12]
y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]
data = list(zip(x, y)) # turn the data into a set of points.
hierarchical_cluster = AgglomerativeClustering(n_clusters=2,affinity='euclidean', linkage='ward')
labels = hierarchical_cluster.fit_predict(data) # this fit_predict method can be call on the data to compute the cluster using t0 defined parameters.
plt.scatter(x, y, c=labels)
plt.show()

# Grid Search: The method is to try out diiferent values and then pick the value gives the best score. This technique is known as grid search.
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
iris = datasets.load_iris()
x = iris['data']
y = iris['target']
logit = LogisticRegression(max_iter=1000)
print(logit.fit(x,y))
print(logit.score(x,y))
# Now here we will implement a grid search
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
iris = datasets.load_iris()
x = iris['data']
y = iris['target']
logit = LogisticRegression(max_iter=1000)
c = [0.25,0.5,0.75,1,1.25,1.5,1.75,2]
scores=[]
for choice in c:
     logit.set_params(C=choice)
     logit.fit(x,y)
     scores.append(logit.score(x,y))
print(scores)'''
# what about best practices

# Decison Tree:
from sklearn import tree
# Each tuple represents ( outlook, Temperature, Humidity, PlayTennis)
data = [("Sunny","Hot","High", "No"),("Sunny","Hot","Normal","No"),("Overcast","Hot","High","Yes"),("Rainy","Mild","High","Yes"),
        ("Rainy","Cool","Normal","No")]
# Convert categorical data to numerical data
outlook_mapping = {"Sunny":0,"Overcast":1,"Rainy":2}
temperature_mapping = {"Hot":0,"Mild":1,"Cool":2}
Humidity_mapping = {"High":0,"Normal":1}
play_tennis_mapping = {"No":0,"Yes":1 }
# iterating over the orginal dataset
data_numeric = [(outlook_mapping[outlook],temperature_mapping[temp],Humidity_mapping[hum],play_tennis_mapping[play])]
for outlook,temp,hum,play in data:
    



