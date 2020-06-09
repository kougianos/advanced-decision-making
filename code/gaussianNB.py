attributes = ['sex', 'school', 'higher', 'studytime', 'G3', 'absences']

print("This python script applies Gaussian Naive Bayes to the dataset.")
print('''Please select one of the following attributes to test:
sex 			-> 0
school 			-> 1
higher education 	-> 2
study time 		-> 3
G3 			-> 4
absences 		-> 5
'''
)

selection = input()
while(True):

	try:
		if int(selection) not in range(6):
			selection = input("Wrong input. Please select a number from 0 to 5: ")
		else:
			print(f"Your selected attribute is {attributes[int(selection)]}\n")
			selection = attributes[int(selection)]
			break
	except:
		selection = input("Wrong input. Please select a number from 0 to 5: ")

print("Please select how many iterations you would like to execute: ")
iterations = input()
while(True):

	try:
		iterations = int(iterations)
		print(f"Iterations: {iterations}\n")
		break
	except:
		iterations = input("Wrong input. Please select an integer: ")

print("Calculating average score...", end='')

# ################################################ #
# GaussianNB
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import datetime

begin_time = datetime.datetime.now()

df = pd.read_csv("./data/student-merged.csv")

# Data preprocessing - convert all classified attributes to numeric
for index_label, row_series in df.iterrows():

    if row_series['school'] == 'GP':
        df.at[index_label , 'school'] = 0
    elif row_series['school'] == 'MS':
        df.at[index_label , 'school'] = 1

    if row_series['sex'] == 'F':
        df.at[index_label , 'sex'] = 0
    elif row_series['sex'] == 'M':
        df.at[index_label , 'sex'] = 1

    if row_series['address'] == 'U':
        df.at[index_label , 'address'] = 0
    elif row_series['address'] == 'R':
        df.at[index_label , 'address'] = 1

    if row_series['famsize'] == 'GT3':
        df.at[index_label , 'famsize'] = 0
    elif row_series['famsize'] == 'LE3':
        df.at[index_label , 'famsize'] = 1

    if row_series['Pstatus'] == 'A':
        df.at[index_label , 'Pstatus'] = 0
    elif row_series['Pstatus'] == 'T':
        df.at[index_label , 'Pstatus'] = 1

    if row_series['Mjob'] == 'at_home':
        df.at[index_label , 'Mjob'] = 0
    elif row_series['Mjob'] == 'health':
        df.at[index_label , 'Mjob'] = 1
    elif row_series['Mjob'] == 'services':
        df.at[index_label , 'Mjob'] = 2
    elif row_series['Mjob'] == 'teacher':
        df.at[index_label , 'Mjob'] = 3
    elif row_series['Mjob'] == 'other':
        df.at[index_label , 'Mjob'] = 4

    if row_series['Fjob'] == 'at_home':
        df.at[index_label , 'Fjob'] = 0
    elif row_series['Fjob'] == 'health':
        df.at[index_label , 'Fjob'] = 1
    elif row_series['Fjob'] == 'services':
        df.at[index_label , 'Fjob'] = 2
    elif row_series['Fjob'] == 'teacher':
        df.at[index_label , 'Fjob'] = 3
    elif row_series['Fjob'] == 'other':
        df.at[index_label , 'Fjob'] = 4

    if row_series['reason'] == 'home':
        df.at[index_label , 'reason'] = 0
    elif row_series['reason'] == 'reputation':
        df.at[index_label , 'reason'] = 1
    elif row_series['reason'] == 'course':
        df.at[index_label , 'reason'] = 2
    elif row_series['reason'] == 'other':
        df.at[index_label , 'reason'] = 3

    if row_series['guardian'] == 'mother':
        df.at[index_label , 'guardian'] = 0
    elif row_series['guardian'] == 'father':
        df.at[index_label , 'guardian'] = 1
    elif row_series['guardian'] == 'other':
        df.at[index_label , 'guardian'] = 2

    if row_series['schoolsup'] == 'no':
        df.at[index_label , 'schoolsup'] = 0
    elif row_series['schoolsup'] == 'yes':
        df.at[index_label , 'schoolsup'] = 1

    if row_series['famsup'] == 'no':
        df.at[index_label , 'famsup'] = 0
    elif row_series['famsup'] == 'yes':
        df.at[index_label , 'famsup'] = 1

    if row_series['paid'] == 'no':
        df.at[index_label , 'paid'] = 0
    elif row_series['paid'] == 'yes':
        df.at[index_label , 'paid'] = 1

    if row_series['activities'] == 'no':
        df.at[index_label , 'activities'] = 0
    elif row_series['activities'] == 'yes':
        df.at[index_label , 'activities'] = 1

    if row_series['nursery'] == 'no':
        df.at[index_label , 'nursery'] = 0
    elif row_series['nursery'] == 'yes':
        df.at[index_label , 'nursery'] = 1

    if row_series['higher'] == 'no':
        df.at[index_label , 'higher'] = 0
    elif row_series['higher'] == 'yes':
        df.at[index_label , 'higher'] = 1

    if row_series['internet'] == 'no':
        df.at[index_label , 'internet'] = 0
    elif row_series['internet'] == 'yes':
        df.at[index_label , 'internet'] = 1

    if row_series['romantic'] == 'no':
        df.at[index_label , 'romantic'] = 0
    elif row_series['romantic'] == 'yes':
        df.at[index_label , 'romantic'] = 1

# Convert dataframe to numeric in order to be used with GaussianNB
df = df.apply(pd.to_numeric)

score = count = 0
for i in range(iterations):

	# Create inputs
	inputs = df.drop(selection, axis='columns')

	# Create target
	target = df[selection]

	X_train, X_test, y_train, y_test = train_test_split(inputs,target,test_size=0.2)

	model = GaussianNB()

	model.fit(X_train, y_train)

	count += 1
	score += model.score(X_test,y_test)
	
	if count%100==0:
		print(".", end='') 


print(f"\nThe average score for the attribute {selection} after {iterations} iterations is {score/count}")
print(f"Script completed successfully, time: {datetime.datetime.now() - begin_time}")
