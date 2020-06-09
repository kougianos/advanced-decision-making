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

	if(iterations==0):
		iterations = input("Wrong input. Please select a positive integer: ")

	try:
		iterations = int(iterations)
		print(f"Iterations: {iterations}\n")
		break
	except:
		iterations = input("Wrong input. Please select an integer: ")

print("Calculating average score...", end='')

# ################################################ #
# LinearRegression implementation
import pandas as pd  
import numpy as np  
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
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

df = df.apply(pd.to_numeric)

# Create inputs
X = df.drop(selection, axis='columns')

# Create target
y = df[selection]

for i in range(iterations):

	#Split Data 80-20
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

	#train the algorithm
	lnr = LinearRegression()  
	lnr.fit(X_train, y_train) 

	y_pred = lnr.predict(X_test)

	# First round the values of predictions, the typecast them as ints
	y_pred = y_pred.round()
	y_pred = y_pred.astype(int)

	#Compare the actual output values with the predicted values
	df = pd.DataFrame({'Actual': y_test.to_numpy().flatten(), 'Predicted': y_pred.flatten()})

# No need to calculate average from each iteration as the Linear Regression algorithm makes the same predictions every time
correct_predictions = total_counts = 0
for index_label, row_series in df.iterrows():
	if(row_series['Actual']==row_series['Predicted']):
		correct_predictions += 1
	total_counts += 1

print(f"\nThe average score for the attribute {selection} after {iterations} iterations is {correct_predictions/total_counts}")
print(f"Script completed successfully, time: {datetime.datetime.now() - begin_time}")