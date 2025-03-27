import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.model_selection import GridSearchCV

# Load dataset
df = pd.read_csv("customer_complaints.csv")

# Check how many unique categories exist before processing
print("Unique Categories in Dataset Before Encoding:", df['Category'].unique())
print("Total Unique Categories Before Encoding:", df['Category'].nunique())

# Ensure only 12 valid categories are kept
valid_categories = df['Category'].value_counts().index[:12]
df = df[df['Category'].isin(valid_categories)]

# Select necessary columns and remove empty rows
df = df[['Complaint', 'Category']].dropna()

# Remove categories with less than 2 complaints to ensure stable train-test split
category_counts = df['Category'].value_counts()
df = df[df['Category'].isin(category_counts[category_counts >= 2].index)]

# Load a more generalizable SBERT model
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

# Convert complaints into embeddings
X = sbert_model.encode(df['Complaint'].tolist(), convert_to_tensor=False)

# Encode categories as numerical labels
label_encoder = LabelEncoder()
df['Encoded_Category'] = label_encoder.fit_transform(df['Category'])

# Ensure labels start from 0 and are continuous
df['Encoded_Category'] = df['Encoded_Category'].astype('category').cat.codes
df['Encoded_Category'] = df['Encoded_Category'].astype(int)

# Extract final labels
y = df['Encoded_Category'].tolist()

# Verify the final label sequence
print("Final Mapped Labels:", sorted(set(y)))  # This should now print [0,1,2,3,4,5,6,7,8,9,10,11]

print("Updated Categories in Dataset:")
print(df[['Category', 'Encoded_Category']].drop_duplicates())

# Check dataset balance
print("Category distribution:")
print(df['Category'].value_counts())

# Ensure test size is at least the number of unique categories
min_test_size = max(0.1, len(df['Category'].unique()) / len(df))

# Split into train and test sets using stratified sampling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=min_test_size, stratify=y, random_state=42)

# Get category distribution in y_train
train_category_counts = Counter(y_train)

# Apply SMOTE only if all classes have at least 6 samples
if all(count >= 6 for count in train_category_counts.values()):
    smote = SMOTE(sampling_strategy='not minority', random_state=42, k_neighbors=5)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    print("SMOTE applied to balance dataset.")
else:
    print("Skipping SMOTE because some categories have fewer than 6 samples.")

# Hyperparameter tuning for XGBoost
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.05, 0.1, 0.2]
}

grid_search = GridSearchCV(XGBClassifier(random_state=42), param_grid, scoring='accuracy', cv=3)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print("Best XGBoost Parameters:", best_params)

# Train optimized XGBoost classifier
xgb_classifier = XGBClassifier(**best_params, random_state=42)
xgb_classifier.fit(X_train, y_train)

# Predict on test data
y_pred = xgb_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2%}")

# Save the trained classifier, SBERT model, and label encoder
joblib.dump(xgb_classifier, "model.pkl")
joblib.dump(sbert_model, "vectorizer.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

print("Training complete! Files saved: `model.pkl`, `vectorizer.pkl`, `label_encoder.pkl`")
