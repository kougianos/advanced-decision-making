import pandas as pd
import datetime

begin_time = datetime.datetime.now()

# Load datasets
df1=pd.read_csv("../data/student-mat.csv", sep=";")
df2=pd.read_csv("../data/student-por.csv", sep=";")

# Merge datasets
merged=pd.concat([df1, df2])

# Drop duplicate students
result = merged.drop_duplicates(subset=["school","sex","age","address","famsize","Pstatus","Medu","Fedu","Mjob","Fjob","reason","nursery","internet","romantic","health","famrel","freetime","goout","traveltime"])

# Export dataframe to csv
result.to_csv("../data/student-merged.csv", index=False)

# Print script execution time
print(f"The new merged dataset has {len(result)} rows and is saved at ./data/student-merged.csv")
print(f"Script completed successfully, time: {datetime.datetime.now() - begin_time}")