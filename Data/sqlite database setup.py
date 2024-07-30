import pandas as pd
import sqlite3

# Load CSV into DataFrame
csv_file_path = "dementia_patients_health_data.csv"
df = pd.read_csv(csv_file_path)

# Create SQLite database connection
conn = sqlite3.connect('dementia_data.db')

# Upload DataFrame to SQLite Database
df.to_sql('dementia_data', conn, if_exists='replace', index=False)

# Close connection
conn.close()
