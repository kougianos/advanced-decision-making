import pandas as pd
import datetime

begin_time = datetime.datetime.now()

df=pd.read_csv('../data/student-merged.csv')

# Calculate means and create new columns
df['Pedu'] = df.apply(lambda row: int(round((row.Medu + row.Fedu)/2)), axis = 1)
df['freetime'] = df.apply(lambda row: int(round((row.freetime + row.goout)/2)), axis = 1)
df['alc'] = df.apply(lambda row: int(round((row.Dalc + row.Walc)/2)), axis = 1)

# Drop unneeded columns
df.drop(['Medu','Fedu','goout','Dalc','Walc'],axis='columns',inplace=True)

# Export dataframe to csv
df.to_csv("./data/student-merged.csv", index=False)

# Print script execution time
print(f"Script completed successfully, time: {datetime.datetime.now() - begin_time}")
