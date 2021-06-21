'''
Matplotlib is an amazing visualization library in Python for 2D plots of arrays. Matplotlib is a multi-platform data visualization library built on NumPy arrays and designed to work with the broader SciPy stack.
One of the greatest benefits of visualization is that it allows us visual access to huge amounts of data in easily digestible visuals. Matplotlib consists of several plots like line, bar, scatter, histogram etc.
'''

        #  Installation :  
python -mpip install -U matplotlib

        # Importing matplotlib :
from matplotlib import pyplot as plt
or
import matplotlib.pyplot as plt 

########################################################################################################################

The use of the following functions, methods, classes and modules is shown in this example:

 - pyplot. figure.
 - axes. Axes. text.
 - axis. Axis. set_minor_formatter.
 - axis. Axis. set_major_locator.
 - axis. Axis. set_minor_locator.
 - patches. Circle.
 - patheffects. withStroke.
 - ticker. FuncFormatter.

#########################################################################################################################

'''
Matplotlib is a library in Python and it is numerical – mathematical extension for NumPy library. 
The figure module provides the top-level Artist, the Figure, which contains all the plot elements. 
This module is used to control the default spacing of the subplots and top level container for all plot elements.
'''

##########################################################################################################################

                # matplotlib axis()
This function is used to set some axis properties to the graph.

        # Syntax: 
matplotlib.pyplot.axis(*args, emit=True, **kwargs)

        # Parameters:
xmin, xmax, ymin, ymax:These parameters can be used to
set the axis limits on the graph
emit:Its a bool value used to notify observers of the axis limit change

##########################################################################################################################

                 # Matplotlib Artist draw() in Python : 
                 
Matplotlib is a library in Python and it is numerical – mathematical extension for NumPy library. The Artist class contains Abstract base class for objects that render into a FigureCanvas. All visible elements in a figure are subclasses of Artist.

The draw() method in the artist module of the matplotlib library is used to draw the Artist using the given renderer.

        # Syntax: 
Artist.draw(self, renderer, \*args, \*\*kwargs)

        # Parameters: 
This method accepts the following parameters.

        # renderer: 
This parameter is the RendererBase subclass.

        # Returns: 
This method does not return any value.

                # Example : 

# Implementation of matplotlib function
from matplotlib.artist import Artist
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
	
fig, ax = plt.subplots()
	
def tellme(s):
	ax.set_title(s, fontsize = 16)
	fig.canvas.draw()
	renderer = fig.canvas.renderer
	Artist.draw(ax, renderer)
	
tellme('matplotlib.artist.Artist.draw() function Example')
ax.grid()

plt.show()

##########################################################################################################################

                # Matplotlib Labels and Title : 
                
With Pyplot, you can use the xlabel() and ylabel() functions to set a label for the x- and y-axis.

                # Example : 
                
    # Add labels to the x- and y-axis:

import numpy as np
import matplotlib.pyplot as plt

x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])

plt.plot(x, y)

plt.xlabel("Average Pulse")
plt.ylabel("Calorie Burnage")

plt.show()

##########################################################################################################################

            # Create a Title for a Plot
With Pyplot, you can use the title() function to set a title for the plot.

                # Example : 
Add a plot title and labels for the x- and y-axis:

import numpy as np
import matplotlib.pyplot as plt

x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])

plt.plot(x, y)

plt.title("Sports Watch Data")
plt.xlabel("Average Pulse")
plt.ylabel("Calorie Burnage")

plt.show()

##########################################################################################################################

                # matplotlib.pyplot.xticks() Function : 
                
The annotate() function in pyplot module of matplotlib library is used to get and set the current tick locations and labels of the x-axis.

        # Syntax:
matplotlib.pyplot.xticks(ticks=None, labels=None, **kwargs)

##########################################################################################################################

                # Matplotlib.pyplot.legend() : 
                
A legend is an area describing the elements of the graph. In the matplotlib library, there’s a function called legend() which is used to Place a legend on the axes.

    Syntax : 
matplotlib.pyplot.legend([“blue”, “green”], bbox_to_anchor=(0.75, 1.15), ncol=2)    

                # Example : 
                
import numpy as np
import matplotlib.pyplot as plt

# X-axis values
x = [1, 2, 3, 4, 5]

# Y-axis values
y = [1, 4, 9, 16, 25]

# Function to plot
plt.plot(x, y)

# Function add a legend
plt.legend(['single element'])

# function to show the plot
plt.show()

##########################################################################################################################

Add Grid Lines to a Plot
With Pyplot, you can use the grid() function to add grid lines to the plot.

Example
Add grid lines to the plot:

import numpy as np
import matplotlib.pyplot as plt

x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])

plt.title("Sports Watch Data")
plt.xlabel("Average Pulse")
plt.ylabel("Calorie Burnage")

plt.plot(x, y)

plt.grid()

plt.show()

##########################################################################################################################

            # Table Of Content : 
            
- Line Graph.
- Bar chart.
- Histograms.
- Scatter Plot.
- Pie Chart.
- 3D Plots.

##########################################################################################################################

                # Line plot :

# importing matplotlib module
from matplotlib import pyplot as plt

# x-axis values
x = [5, 2, 9, 4, 7]

# Y-axis values
y = [10, 5, 8, 4, 2]

# Function to plot
plt.plot(x,y)

# function to show the plot
plt.show()

###########################################################################################################################

                # Bar plot :
                
# importing matplotlib module
from matplotlib import pyplot as plt

# x-axis values
x = [5, 2, 9, 4, 7]

# Y-axis values
y = [10, 5, 8, 4, 2]

# Function to plot the bar
plt.bar(x,y)

# function to show the plot
plt.show()

#############################################################################################################################

                # Histogram :
                
# importing matplotlib module
from matplotlib import pyplot as plt

# Y-axis values
y = [10, 5, 8, 4, 2]

# Function to plot histogram
plt.hist(y)

# Function to show the plot
plt.show()

#############################################################################################################################

                # Scatter Plot :
                
# importing matplotlib module
from matplotlib import pyplot as plt

# x-axis values
x = [5, 2, 9, 4, 7]

# Y-axis values
y = [10, 5, 8, 4, 2]

# Function to plot scatter
plt.scatter(x, y)

# function to show the plot
plt.show()

###############################################################################################################################

            # Pie Chart : 
            
Matplotlib API has pie() function in its pyplot module which create a pie chart representing the data in an array.

    # Syntax: 
matplotlib.pyplot.pie(data, explode=None, labels=None, colors=None, autopct=None, shadow=False)

    # Parameters:
data represents the array of data values to be plotted, the fractional area of each slice is represented by data/sum(data).
If sum(data)<1, then the data values returns the fractional area directly, thus resulting pie will have empty wedge of size 1-sum(data).

                # Example : 

# Import libraries
from matplotlib import pyplot as plt
import numpy as np


# Creating dataset
cars = ['AUDI', 'BMW', 'FORD',
		'TESLA', 'JAGUAR', 'MERCEDES']

data = [23, 17, 35, 29, 12, 41]

# Creating plot
fig = plt.figure(figsize =(10, 7))
plt.pie(data, labels = cars)

# show plot
plt.show()

###############################################################################################################################

                # Creating Box Plot : 
The matplotlib.pyplot module of matplotlib library provides boxplot() function with the help of which we can create box plots.

    # Syntax: 
matplotlib.pyplot.boxplot(data, notch=None, vert=None, patch_artist=None, widths=None)

                    # Example : 

# Import libraries
import matplotlib.pyplot as plt
import numpy as np


# Creating dataset
np.random.seed(10)
data = np.random.normal(100, 20, 200)

fig = plt.figure(figsize =(10, 7))

# Creating plot
plt.boxplot(data)

# show plot
plt.show()

###############################################################################################################################

                # Bubble Plot : 

Bubble plots are scatter plots with bubbles (color filled circles) rather than information focuses. Bubbles have various sizes dependent on another variable in the data.

            # Example : 
            
# import all important libraries
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# load dataset
data= "https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv"

# convert to dataframe
df = pd.read_csv(data)

# display top most rows
df.head()

###############################################################################################################################

                # Stacked Plot : 

# importing package
import matplotlib.pyplot as plt

# create data
x = ['A', 'B', 'C', 'D']
y1 = [10, 20, 10, 30]
y2 = [20, 25, 15, 25]

# plot bars in stack manner
plt.bar(x, y1, color='r')
plt.bar(x, y2, bottom=y1, color='b')
plt.show()

###############################################################################################################################

                # Contour plots : 

# imports
import numpy as np
import matplotlib.pyplot as plt
#
# define a function
def func(x, y):
	return np.sin(x) ** 2 + np.cos(y) **2
# generate 50 values b/w 0 a5
x = np.linspace(0, 5, 50)
y = np.linspace(0, 5, 50)

# Generate combination of grids
X, Y = np.meshgrid(x, y)
Z = func(X, Y)

# Draw rectangular contour plot
plt.contour(X, Y, Z, cmap='gist_rainbow_r');

###############################################################################################################################

                    # Table Chart : 

Matplotlib.pyplot.table() is a subpart of matplotlib library in which a table is generated using the plotted graph for analysis. This method makes analysis easier and more efficient as tables give a precise detail than graphs.

      # Syntax: 
matplotlib.pyplot.table(cellText=None, cellColours=None, cellLoc=’right’, colWidths=None,rowLabels=None, rowColours=None, rowLoc=’left’, colLabels=None, colColours=None, colLoc=’center’, loc=’bottom’, bbox=None, edges=’closed’, **kwargs)

                # Example : 

# importing necesarry packagess
import numpy as np
import matplotlib.pyplot as plt


# input data values
data = [[322862, 876296, 45261, 782372, 32451],
		[58230, 113139, 78045, 99308, 516044],
		[89135, 8552, 15258, 497981, 603535],
		[24415, 73858, 150656, 19323, 69638],
		[139361, 831509, 43164, 7380, 52269]]

# preparing values for graph
columns = ('Soya', 'Rice', 'Wheat', 'Bakri', 'Ragi')
rows = ['%d months' % x for x in (50, 35, 20, 10, 5)]
values = np.arange(0, 2500, 500)
value_increment = 1000

# Adding pastel shades to graph
colors = plt.cm.Oranges(np.linspace(22, 3, 12))
n_rows = len(data)
index = np.arange(len(columns)) + 0.3
bar_width = 0.4

# Initialing vertical-offset for the graph.
y_offset = np.zeros(len(columns))

# Plot bars and create text labels for the table
cell_text = []

for row in range(n_rows):
	plt.plot(index, data[row], bar_width, color=colors[row])
	y_offset = y_offset + data[row]
	cell_text.append(['%1.1f' % (x / 1000.0) for x in y_offset])

# Reverse colors and text labels to display table contents with
# color.
colors = colors[::-1]
cell_text.reverse()

# Add a table at the bottom
the_table = plt.table(cellText=cell_text,
					rowLabels=rows,
					rowColours=colors,
					colLabels=columns,
					loc='bottom')

# make space for the table:
plt.subplots_adjust(left=0.2, bottom=0.2)
plt.ylabel("Price in Rs.{0}'s".format(value_increment))
plt.yticks(values * value_increment, ['%d' % val for val in values])
plt.xticks([])
plt.title('Cost price increase')

# plt.show()-display graph
# Create image. plt.savefig ignores figure edge and face color.
fig = plt.gcf()
plt.savefig('pyplot-table-original.png',
			bbox_inches='tight',
			dpi=150)

###############################################################################################################################

                # Polar chart:
The polar() function in pyplot module of matplotlib library is used to make a polar plot.

    # Syntax: 
matplotlib.pyplot.polar(*args, **kwargs)

    # Parameters: 
This method does not accept any parameters.

    # Returns: 
This method does not returns any value.

                # Example : 

# Implementation of matplotlib function
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np

from matplotlib.transforms import offset_copy

xs = np.arange(-2, 2)
ys = np.cos(xs**2)
plt.polar(xs, ys)

plt.title('matplotlib.pyplot.polar() function Example',
									fontweight ="bold")
plt.show()

###############################################################################################################################

                # Matplotlib – Axes Class : 

'''
Matplotlib is one of the Python packages which is used for data visualization. You can use the NumPy library to convert data into an array and numerical mathematics extension of Python. Matplotlib library is used for making 2D plots from data in arrays.
'''
        # Syntax :

axes([left, bottom, width, height])

        # Example :
        
import matplotlib.pyplot as plt


fig = plt.figure()

#[left, bottom, width, height]
ax = plt.axes([0.1, 0.1, 0.8, 0.8])
