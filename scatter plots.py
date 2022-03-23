import matplotlib.pyplot as plt


def make_scatter_plots(df):
	df.plot.scatter(x='Fare', y ='Age')
	df.plot.scatter(x='Pclass', y ='Age')
	df.plot.scatter(x='Sex', y ='Age')
	plt.show()