import pandas as pd
import mysql.connector
from sqlalchemy import create_engine

# 1. DATABASE CONNECTION
db_config = {
    "host": "localhost",
    "user": "root",
    "password": "*********", 
    "database": "consumer360"
}

try:
    conn = mysql.connector.connect(**db_config)
    print("Successfully connected to MySQL!")
except Exception as e:
    print(f"Error: {e}")

# 2. PULL DATA
query = "SELECT * FROM transactions_raw"
df = pd.read_sql(query, conn)

# 3. CLEANING & PREPROCESSING
df['InvoiceDateTime'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
df['TotalRevenue'] = df['Quantity'] * df['UnitPrice']
df = df.dropna(subset=['CustomerID', 'InvoiceDateTime'])

# 4. CALCULATE RFM METRICS
snapshot_date = df['InvoiceDateTime'].max() + pd.Timedelta(days=1)

rfm_data = df.groupby('CustomerID').agg({
    'InvoiceDateTime': lambda x: (snapshot_date - x.max()).days,
    'InvoiceNo': 'count',
    'TotalRevenue': 'sum'
})

rfm_data.rename(columns={
    'InvoiceDateTime': 'Recency',
    'InvoiceNo': 'Frequency',
    'TotalRevenue': 'Monetary'
}, inplace=True)

# 5. SCORING (Fixes the "Bin Edges" Error)
# We use .rank() to ensure every customer gets a unique position before splitting into 5 groups
rfm_data['R_Score'] = pd.qcut(rfm_data['Recency'].rank(method='first'), 5, labels=[5, 4, 3, 2, 1])
rfm_data['F_Score'] = pd.qcut(rfm_data['Frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
rfm_data['M_Score'] = pd.qcut(rfm_data['Monetary'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])

# 6. SEGMENTATION LOGIC
def define_segments(row):
    r = int(row['R_Score'])
    f = int(row['F_Score'])
    
    if r >= 4 and f >= 4:
        return 'Champions'
    elif r >= 3 and f >= 4:
        return 'Loyal Customers'
    elif r >= 4 and f >= 2:
        return 'Potential Loyalist'
    elif r == 5:
        return 'Recent Users'
    elif r == 4:
        return 'Promising'
    elif r == 3 and f == 3:
        return 'Needs Attention'
    elif r == 3:
        return 'About To Sleep'
    elif r <= 2 and f >= 4:
        return "Can't Lose Them"
    elif r <= 2 and f >= 2:
        return 'Hibernating'
    else:
        return 'Lost'

rfm_data['Segment'] = rfm_data.apply(define_segments, axis=1)

# 7. EXPORT RESULTS
rfm_data.to_csv("rfm_results_final.csv")

# Create SQL Engine for saving
engine = create_engine(f"mysql+mysqlconnector://{db_config['user']}:{db_config['password']}@{db_config['host']}/{db_config['database']}")
rfm_data.to_sql('customer_segments', con=engine, if_exists='replace')

print("--- ANALYSIS COMPLETE ---")
print(f"Processed {len(rfm_data)} unique customers.")
print("Results saved to 'rfm_results_final.csv' and MySQL table 'customer_segments'.")
print(rfm_data.head())
