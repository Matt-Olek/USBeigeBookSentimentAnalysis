import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import yfinance as yf

# Step 1: Load the dataset
file_path = "beige_book_results_LLM.csv"
data = pd.read_csv(file_path)

# Step 2: Download S&P 500 historical data
sp500_data = yf.download("^GSPC", start="2004-01-01", end="2024-01-01")
sp500_data["Return"] = sp500_data["Adj Close"].pct_change().shift(-1)
sp500_data = sp500_data.reset_index()
sp500_data = sp500_data[["Date", "Return"]]  # Keep only relevant columns
sp500_data["date"] = pd.to_datetime(sp500_data["Date"]).dt.to_period("M")
sp500_data = sp500_data.groupby("date").last().reset_index()

# Step 3: Merge datasets
# Convert 'date' column in the Beige Book data to datetime and then to monthly periods
data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.to_period("M")

# Perform the merge
merged_data = data.droplevel(0).join(sp500_data, on="date", how="inner")

# Drop unnecessary columns
merged_data = merged_data.drop(columns=["url", "summary", "date", "Date"])

# Step 4: Create a target variable
merged_data["Target"] = (merged_data["Return"] > 0).astype(
    int
)  # 1 if positive return, else 0
merged_data = merged_data.drop(columns=["Return"])  # Drop the return column

# Step 5: Split into features and target
X = merged_data[
    ["hiring market", "consumer spending", "economic growth", "material prices"]
].fillna(0)
y = merged_data["Target"]

# Step 6: Train-test-validation split
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

# Step 7: Train a Random Forest classifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)

# Step 8: Evaluate the model
y_val_pred = rf_clf.predict(X_val)
print("Validation Set Performance:")
print(classification_report(y_val, y_val_pred))

y_test_pred = rf_clf.predict(X_test)
print("Test Set Performance:")
print(classification_report(y_test, y_test_pred))
