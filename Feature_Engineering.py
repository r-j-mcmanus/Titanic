import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def Feature_Engineering_Cabin(df):
	df['Cabin']=df['Cabin'].fillna('')
	df['Cabin']=df['Cabin'].apply(lambda cabin_string: cabin_string.split() )
	df['Number_Of_Cabins']= df['Cabin'].apply(lambda cabin_string_list: len(cabin_string_list) )
	df['Cabin']=df['Cabin'].apply(lambda cabin_string_list: [ cabin_string[0] for cabin_string in cabin_string_list] )
	df['Cabin']=df['Cabin'].apply(lambda cabin_string_list:  list(['X']) if cabin_string_list == list([]) else cabin_string_list)

	#print(set().union(*df['Cabin'].values))
	for cabin_letter in set().union(*df['Cabin'].values):
		df['Cabin_'+cabin_letter]=df['Cabin'].apply(lambda cabin_string_list:  1 if cabin_letter in cabin_string_list else 0)

	return df

def Feature_Engineering_Name_To_Titles(df):
	#print(np.array(df['Name'].values))
	titles = ['Mr', 'Mrs', 'Ms', 'Miss', 'Mlle', 'Master', 'Dr', 'Rev', 'Capt', 'Major', 'Col', 'Countess', 'Jonkheer', 'Mme', 'Don']
	for title in titles:
		df['Name']=df['Name'].apply(lambda name_string: title if title in name_string else name_string )
	df['Name']=df['Name'].apply(lambda name_string: 'X' if name_string not in titles else name_string )
	#print(np.array(df['Name'].values))

	return df

def Feature_Engineering_Tickets(df):
	df['Ticket'] = df['Ticket'].fillna('')
	numbers = ['0','1','2','3','4','5','6','7','8','9']
	df['Ticket_numbered'] = df['Ticket'].apply(lambda ticket_string: 1 if  ticket_string[0] in numbers else 0)
	return df

def Feature_Engineering(df):
	#we drop these for now, will come back to include them
	df = df.set_index('PassengerId')
	df = Feature_Engineering_Cabin(df)
	df = Feature_Engineering_Name_To_Titles(df)
	df = Feature_Engineering_Tickets(df)

	#df = df.drop('Ticket', axis = 1)
	df = df.drop('Name', axis = 1)
	df = df.drop('Cabin', axis = 1)

	df = df[[
	'Survived','Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket_numbered', 'Fare', 'Embarked', 'Number_Of_Cabins', 
	'Cabin_X', 'Cabin_A', 'Cabin_B', 'Cabin_C', 'Cabin_D', 'Cabin_E', 'Cabin_F'
	]]

	df = pd.get_dummies(df)
	df = df.drop('Sex_male', axis = 1)

	#print('check is nan')
	#for key in df.keys():
	#	print(key, df[key].isnull().sum())
	#print('')
	#we see that age has 177 nan's
	#fair has 1
	#make an imputer to handel these values
	#might be smarter way using correlations ect but for now use mean

	return df