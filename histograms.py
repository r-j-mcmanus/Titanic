import matplotlib.pyplot as plt

def make_histograms(df)
	nbins=dict(zip(df.keys()
		,
		[
		10, #PassengerId
		2, #Survived
		3, #Pclass
		10, #Age
		10, # number of siblings or spouses
		10, # number of parents or children
		10, # fare
		2, # Sex_female
		2, # Sex_male
		2, # Embarked_C
		2, # Embarked_Q
		2 # Embarked_S
		]
		))

	for key in df.keys():
		plt.hist(df[df['Survived'] == 1][key].values, density = True, bins = nbins[key])
		plt.hist(df[df['Survived'] == 0][key].values, density = True, bins = nbins[key], alpha = 0.8)
		plt.xlabel(key)
		plt.ylabel('Survived')
		plt.legend(['Survived', 'Died'])
		plt.show()