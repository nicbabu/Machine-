# pyplot submodule
# now we will draw a line from a certain position
'''import matplotlib.pyplot as plt
import numpy as np
xpoint= np.array([0,6])
ypoint= np.array([0,250])
plt.plot(xpoint,ypoint)
plt.show( )

import matplotlib.pyplot as plt # now the package can be refered as plt
import numpy as np
# plotting x and y
# plot() function is used to draw points or markers in a diagram
# There are two parameters for specifying points in the diagram x-axis, y-axis
xpoint = np.array([1,8])
ypoint =np.array([3,10])
plt.plot(xpoint, ypoint)
plt.show()
# plotting without line
import matplotlib.pyplot as plt
import numpy as np
xpoint = np.array([1,8])
ypoint =np.array([3,10])
plt.plot(xpoint, ypoint,'o')
plt.show()
import matplotlib.pyplot as plt
import numpy as np
xpoint = np.array([1,8,6,7,9,14])
ypoint =np.array([3,10,7,4,9,11])
plt.plot(xpoint, ypoint)
plt.show()
# Default x-point
import matplotlib.pyplot as plt
import numpy as np
ypoint=np.array([2,4,5,7,9,12])
plt.plot(ypoint)
plt.show()
# Matplotlib Marker: you can use argument marker to emphasize each point with specified marker
import matplotlib.pyplot as plt
import numpy as np
ypoint = np.array([3,8,1,10])
plt.plot(ypoint, marker='o')
plt.show()
# format strings("fmt"): marker|line|color(string notation parameter):
import matplotlib.pyplot as plt
import numpy as np
ypoint = np.array([3,8,1,10])
plt.plot(ypoint,'o-b')
plt.show()
# color reference
# r(red),g=green, b=blue, m=magenta, c=cyan, k=black, y=yellow
# Marker size
import matplotlib.pyplot as plt
import numpy as np
ypoint = np.array([3,8,1,10])
plt.plot(ypoint,marker='o', ms=20)  # ms= marker size
plt.show()
# marker Edge color: Marker each color= mec
import matplotlib.pyplot as plt
import numpy as np
ypoint = np.array([3,8,1,10])
plt.plot(ypoint,marker='o', ms=20, mec='c')  # ms= marker size
plt.show()
# marker face color: mfc
import matplotlib.pyplot as plt
import numpy as np
ypoint = np.array([3,8,1,10])
plt.plot(ypoint,marker='o', ms=20, mec='#6fff33', mfc='#ff5533')  # ms= marker size
plt.show()
# Line Style or ls is used to change the style of the plotted line
import matplotlib.pyplot as plt
import numpy as np
ypoint= np.array([3,8,1,10])
plt.plot(ypoint, ls= 'solid', color='#fbff33')
plt.show()
# with two array
import matplotlib.pyplot as plt
import numpy as np
xpoint=np.array([9,13,14,15])
ypoint= np.array([3,8,1,10])
plt.plot(xpoint,ypoint, lw='25.5', color='#fbff33')
plt.show()
# plot x-point and y-point seperately
import matplotlib.pyplot as plt
import numpy as np
xpoint=np.array([9,13,14,15])
ypoint= np.array([3,8,1,10])
plt.plot(xpoint, marker='o',lw='10', ls='dotted', c='orange')
plt.plot(ypoint)
plt.show()

# create label for the plot
import matplotlib.pyplot as plt
import numpy as np
x = np.array([80,85,90,95,100,105,110,115,120,125])
y = np.array([250,260,270,280,290,300,310,320,330,340])
plt.plot(x,y)
plt.xlabel('No. of house')
plt.ylabel('Price')
plt.show()
# creating title for the plot
import matplotlib.pyplot as plt
import numpy as np
x = np.array([80,85,90,95,100,105,110,115,120,125])
y = np.array([250,260,270,280,290,300,310,320,330,340])
plt.plot(x,y)
plt.title('The house selling prediction')
plt.xlabel('No. of house')
plt.ylabel('Price')
plt.show()
# Now we will set font properties for title and label via fontdict()
import matplotlib.pyplot as plt
import numpy as np
x = np.array([80,85,90,95,100,105,110,115,120,125])
y = np.array([250,260,270,280,290,300,310,320,330,340])
font1 = {'family':'arial', 'color':'blue', 'size':20}
font2 = {'family':'serif', 'color':'red', 'size':15}
plt.plot(x,y)
plt.title('Health Monitor',fontdict=font1)
plt.xlabel('Average oxygen', fontdict=font2)
plt.ylabel('our calorie')
plt.show()

# you also chnage the location of title via "loc"
import matplotlib.pyplot as plt
import numpy as np
x = np.array([80,85,90,95,100,105,110,115,120,125])
y = np.array([250,260,270,280,290,300,310,320,330,340])
font1 = {'family':'arial', 'color':'blue', 'size':20}
font2 = {'family':'serif', 'color':'red', 'size':15}
plt.plot(x,y)
plt.title('Health Monitor',fontdict=font1, loc='left')
plt.xlabel('Average oxygen', fontdict=font2)
plt.ylabel('our calorie')
plt.show()

# Adding the grid lines to the plot via "grid()"
import matplotlib.pyplot as plt
import numpy as np
x = np.array([80,85,90,95,100,105,110,115,120,125])
y = np.array([250,260,270,280,290,300,310,320,330,340])
font1 = {'family':'arial', 'color':'blue', 'size':20}
font2 = {'family':'serif', 'color':'red', 'size':15}
plt.title('Health Monitor',fontdict=font1)
plt.xlabel('Average oxygen', fontdict=font2)
plt.ylabel('our calorie',fontdict=font2)
plt.plot(x,y)
plt.grid()
plt.show()
# now we will specify which grid lines to display via x axis or y-axis.(legal values are x and y and both default value is both)
import matplotlib.pyplot as plt
import numpy as np
x = np.array([80,85,90,95,100,105,110,115,120,125])
y = np.array([250,260,270,280,290,300,310,320,330,340])
font1 = {'family':'arial', 'color':'blue', 'size':20}
font2 = {'family':'serif', 'color':'red', 'size':15}
plt.title('Health Monitor',fontdict=font1)
plt.xlabel('Average oxygen', fontdict=font2)
plt.ylabel('our calorie',fontdict=font2)
plt.plot(x,y)
plt.grid(color='green',ls='--', lw='0.7')
plt.show()

# Display the multiple plots- with subplot() we can draw multiple plot
import matplotlib.pyplot as plt
import numpy as np

# plot1
x = np.array([0,1,2,3])
y= np.array([3,8,1,12])
plt.subplot(1,2,1) # the figure has 1 row, 2 columns and this plot is the first plot
plt.plot(x,y)

# plot2
x = np.array([0,1,2,3])
y= np.array([3,8,10,30])
plt.subplot(1,2,2)
plt.plot(x,y)
plt.show()
# now will draw 2 plot on top of each other
import matplotlib.pyplot as plt
import numpy as np

# plot1
x = np.array([0,1,2,3])
y= np.array([3,8,1,12])
plt.subplot(2,1,1) # the figure has 1 row, 2 columns and this plot is the first plot
plt.plot(x,y)

# plot2
x = np.array([0,1,2,3])
y= np.array([3,8,10,30])
plt.subplot(2,1,2)
plt.plot(x,y)
plt.show()
# now we will draw challenges of 6 plot, we play with both rows and column
import matplotlib.pyplot as plt
import numpy as np

# plot1
x = np.array([0,1,2,3])
y= np.array([3,8,1,12])
plt.subplot(2,3,1) # the figure has 1 row, 2 columns and this plot is the first plot
plt.plot(x,y)
plt.title('sales')

# plot2
x = np.array([0,1,2,3])
y= np.array([3,8,10,30])
plt.subplot(2,3,2)
plt.plot(x,y)
plt.title('home')

# plot3
x = np.array([0,1,2,3])
y= np.array([3,8,1,12])
plt.subplot(2,3,3) # the figure has 1 row, 2 columns and this plot is the first plot
plt.plot(x,y)
plt.title('home')
# plot4
x = np.array([0,1,2,3])
y= np.array([3,8,10,30])
plt.subplot(2,3,4)
plt.plot(x,y)
plt.title('home')

# plot5
x = np.array([0,1,2,3])
y= np.array([3,8,1,12])
plt.subplot(2,3,5) # the figure has 1 row, 2 columns and this plot is the first plot
plt.plot(x,y)
plt.title('home')
# plot6
x = np.array([0,1,2,3])
y= np.array([3,8,10,30])
plt.subplot(2,3,6)
plt.plot(x,y)
plt.title('home')
plt.show()

# draw plot having 1 rows, 2 column
import matplotlib.pyplot as plt
import numpy as np

# plot1
x = np.array([0,1,2,3])
y= np.array([3,8,1,12])
plt.subplot(1,2,1) # the figure has 1 row, 2 columns and this plot is the first plot
plt.plot(x,y)

# plot2
x = np.array([0,1,2,3])
y= np.array([3,8,10,30])
plt.subplot(1,2,2)
plt.plot(x,y)
plt.show()
# now will draw 2 plot on top of each other
import matplotlib.pyplot as plt
import numpy as np

# plot1
x = np.array([0,1,2,3])
y= np.array([3,8,1,12])
plt.subplot(2,1,1) # the figure has 1 row, 2 columns and this plot is the first plot
plt.plot(x,y)

# plot2
x = np.array([0,1,2,3])
y= np.array([3,8,10,30])
plt.subplot(2,1,2)
plt.plot(x,y)
plt.show()


# now we will draw challenges of 6 plot, we play with both rows and column
import matplotlib.pyplot as plt
import numpy as np

# plot1
x = np.array([0,1,2,3])
y= np.array([3,8,1,12])
plt.subplot(2,3,1) # the figure has 1 row, 2 columns and this plot is the first plot
plt.plot(x,y)


# plot2
x = np.array([0,1,2,3])
y= np.array([3,8,10,30])
plt.subplot(2,3,2)
plt.plot(x,y)


# plot3
x = np.array([0,1,2,3])
y= np.array([3,8,1,12])
plt.subplot(2,3,3) # the figure has 1 row, 2 columns and this plot is the first plot
plt.plot(x,y)

# plot4
x = np.array([0,1,2,3])
y= np.array([3,8,10,30])
plt.subplot(2,3,4)
plt.plot(x,y)


# plot5
x = np.array([0,1,2,3])
y= np.array([3,8,1,12])
plt.subplot(2,3,5) # the figure has 1 row, 2 columns and this plot is the first plot
plt.plot(x,y)

# plot6
x = np.array([0,1,2,3])
y= np.array([3,8,10,30])
plt.subplot(2,3,6)
plt.plot(x,y)
plt.suptitle('sales', c='blue')
plt.show()

# scatter: the scatter() function plots one dot for each observation
import matplotlib.pyplot as plt
import numpy as np
x = np.array([5,7,8,7,2,17,2,9,12,9,6])
y = np.array([99,86,87,90,93,97,112,98,115,95,177])
plt.xlabel('car', c='green')
plt.ylabel('speed', c='blue')
plt.scatter(x,y)
plt.show()

# now we will compare two plots on same figure
import matplotlib.pyplot as plt
import numpy as np
# day 1, the age and speed of 11 cars
x = np.array([5,7,8,7,2,17,2,9,12,9,6])
y = np.array([99,86,87,90,93,97,112,98,115,95,177])
plt.xlabel('car', c='green')
plt.ylabel('speed', c='blue')
plt.scatter(x,y)

# day 2, the age and speed of 11 cars
x = np.array([4,2,6,1,7,8,93,4,10,11,3])
y = np.array([97,112,98,115,120,123,115,140,132,124,142])
plt.xlabel('car', c='green')
plt.ylabel('speed', c='blue')
plt.scatter(x,y)
plt.show()

# scatter plot color property
# now we will set our own color
import matplotlib.pyplot as plt
import numpy as np
# day 1, the age and speed of 11 cars
x = np.array([5,7,8,7,2,17,2,9,12,9,6])
y = np.array([99,86,87,90,93,97,112,98,115,95,177])
plt.xlabel('car', c='green')
plt.ylabel('speed', c='blue')
plt.scatter(x,y, color='red', lw=4.2)

# day 2, the age and speed of 11 cars
x = np.array([4,2,6,1,7,8,93,4,10,11,3])
y = np.array([97,112,98,115,120,123,115,140,132,124,142])
plt.xlabel('car', c='green')
plt.ylabel('speed', c='blue')
plt.scatter(x,y, color='green')
plt.show()

# now will change color of each scatter plot
import matplotlib.pyplot as plt
import numpy as np
# day 1, the age and speed of 11 cars
x = np.array([5,7,8,7,2,17,2,9,12,9,6])
y = np.array([99,86,87,90,93,97,112,98,115,95,177])
color= (["red","green","blue","orange","brown", "cyan", "purple","yellow","black","gray","magenta"])
plt.scatter(x,y, c=color, lw=4.2)
plt.show()

# now we will create color array and specify a color map in scatter plot
import matplotlib.pyplot as plt
import numpy as np
x = np.array([5,7,8,7,2,17,2,9,12,9,6])
y = np.array([99,86,87,90,93,97,112,98,115,95,177])
color= np.array([10,20,30,24,34,32,16,11,15,17,23])
plt.scatter(x,y, c=color, cmap='viridis')
plt.show()
# you can also include color bar in the plot
import matplotlib.pyplot as plt
import numpy as np
x = np.array([5,7,8,7,2,17,2,9,12,9,6])
y = np.array([99,86,87,90,93,97,112,98,115,95,177])
color= np.array([10,20,30,24,34,32,16,11,15,17,23])
plt.scatter(x,y, c=color, cmap='viridis')
plt.colorbar()
plt.show()
# you can also change the size
import matplotlib.pyplot as plt
import numpy as np
x = np.array([5,7,8,7,2,17,2,9,12,9,6])
y = np.array([99,86,87,90,93,97,112,98,115,95,177])
size= np.array([10,20,30,24,34,32,16,11,15,17,23])
plt.scatter(x,y,s=size)
plt.show()

# alpha: you can adjust the transparency of the dots
import matplotlib.pyplot as plt
import numpy as np
x = np.array([5,7,8,7,2,17,2,9,12,9,6])
y = np.array([99,86,87,90,93,97,112,98,115,95,177])
size= np.array([10,20,30,24,34,32,16,11,15,17,23])
plt.scatter(x,y,s=size, alpha=0.5)
plt.show()

# Now we will combine color, size and alpha
import matplotlib.pyplot as plt
import numpy as np

x = np.random.randint(100, size=(100))
y = np.random.randint(100, size=(100))
color = np.random.randint(100, size=(100))
size = 10* np.random.randint(100, size=(100))
plt.scatter(x,y,s=size, c=color, alpha=0.6, cmap='nipy_spectral')
plt.colorbar()
plt.show()

# Bar graph:
import matplotlib.pyplot as plt
import numpy as np
x= np.array(['a', 'b', 'c', 'd'])
y = np.array([3, 7, 9, 10])
plt.bar(x,y)
plt.show()

# now we will create horizontal bar
import matplotlib.pyplot as plt
import numpy as np
x= np.array(['a', 'b', 'c', 'd'])
y = np.array([3, 7, 9, 10])
plt.barh(x,y)
plt.show()

# color of the bar() and barh()
import matplotlib.pyplot as plt
import numpy as np
x= np.array(['a', 'b', 'c', 'd'])
y = np.array([3, 7, 9, 10])
plt.barh(x,y, color="green")
plt.show()

# how we can change the bar width
import matplotlib.pyplot as plt
import numpy as np
x= np.array(['a', 'b', 'c', 'd'])
y = np.array([3, 7, 9, 10])
plt.barh(x,y, width=0.1)
plt.show()

# for horizontal bar you have to use height instead of width
# how we can change the bar width
import matplotlib.pyplot as plt
import numpy as np
x= np.array(['a', 'b', 'c', 'd'])
y = np.array([3, 7, 9, 10])
plt.barh(x,y, height=0.1)
plt.show()

# Histogram: it is defined as the graph showing the frequency distribution
# Example-say you ask for the height of 250 people then how we will make a histogram
# hist() function
import numpy as np
x= np.random.normal(170,10,250)
print(x)

# the hist() will read the array and produce the histogram
import matplotlib.pyplot as plt
import numpy as np
x = np.random.normal(170,10,250)
plt.hist(x)
plt.show()

# pie chart i.e pie() function:
# creating pie chart
import matplotlib.pyplot as plt
import numpy as np
y = np.array([35,48,55])
mylabels= ["Apple","Banana", "orange"]
plt.pie(y,labels= mylabels)
plt.show()

# startangle parameter: the default start angle is at the x-axis but you can change the start by specifying start angle parameter
import matplotlib.pyplot as plt
import numpy as np
y = np.array([35,48,55])
mylabels= ["Apple","Banana", "orange"]
plt.pie(y,labels= mylabels, startangle=90)
plt.show()

# how to explode the pie chart by a value:
import matplotlib.pyplot as plt
import numpy as np
y = np.array([35,48,55])
mylabels= ["Apple","Banana", "orange"]

myexplode= [0.2,0,0]  # where the values is distance(wedges)
plt.pie(y,labels= mylabels,explode=myexplode)
plt.show()

# shadow parameter for creativity
import matplotlib.pyplot as plt
import numpy as np
y = np.array([35,48,55])
mylabels= ["Apple","Banana", "orange"]
mycolor= ["Blue","green", "Red"]
myexplode= [0.2,0,0]  # where the values is distance(wedges)
plt.pie(y,labels= mylabels,explode=myexplode, shadow=True, colors=mycolor)
plt.show()

#  we can also add the labeling or leagend(list of explanation) in piechart
import matplotlib.pyplot as plt
import numpy as np
y = np.array([35,48,55])
mylabels= ["Apple","Banana", "orange"]
mycolor= ["Blue","green", "Red"]
myexplode= [0.2,0,0]  # where the values is distance(wedges)
plt.pie(y,labels= mylabels,explode=myexplode, shadow=True, colors=mycolor)
plt.legend()
plt.show()'''

# now we will use legend with header
import matplotlib.pyplot as plt
import numpy as np
y = np.array([35,48,55])
mylabels= ["Apple","Banana", "orange"]
mycolor= ["Blue","green", "Red"]
myexplode= [0.2,0,0]  # where the values is distance(wedges)
plt.pie(y,labels= mylabels,explode=myexplode, shadow=True, colors=mycolor)
plt.legend(title ="Name of fruits")
plt.show()










