import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
try:
    df = pd.read_csv("Iris.csv", delimiter="\t")  # Adjust delimiter if needed
    print(" Dataset loaded successfully!\n")
except FileNotFoundError:
    print(" Error: The file 'Iris.csv' was not found.")
    exit()

# Clean column names
df.columns = df.columns.str.strip()

# Drop 'Id' column if present
df.drop(columns=['Id'], inplace=True, errors='ignore')

# Check for missing values
if df.isnull().sum().any():
    print(" Missing values found. Filling with column means.")
    df.fillna(df.mean(), inplace=True)

# Encode species labels
if 'species' not in df.columns:
    print(" Error: 'species' column not found.")
    exit()

label_encoder = LabelEncoder()
df['species'] = label_encoder.fit_transform(df['species'])

# Split data
X = df.drop(columns=['species'])
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f" Model Accuracy: {accuracy:.2%}")
print("\n Classification Report:\n", classification_report(y_test, y_pred))

# Feature Importance
importances = model.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)
print("\n Feature Importances:")
print(importance_df.to_string(index=False))

# Plot Feature Importances
plt.figure(figsize=(6, 4))
sns.barplot(x="Importance", y="Feature", data=importance_df, palette="viridis")
plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# Confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap='Blues', fmt='d',
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# Pie chart: Species distribution
species_counts = df['species'].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(species_counts, labels=label_encoder.inverse_transform(species_counts.index),
        autopct='%1.1f%%', colors=['skyblue', 'lightgreen', 'coral'])
plt.title("Distribution of Iris Species")
plt.show()

# Contour plot
plt.figure(figsize=(8, 6))
sns.kdeplot(data=df, x="sepal_length", y="sepal_width", hue="species", fill=True, alpha=0.5)
plt.title("Contour Plot of Sepal Dimensions")
plt.tight_layout()
plt.show()

# 3D scatter plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(df['sepal_length'], df['sepal_width'], df['petal_length'],
                     c=df['species'], cmap='viridis')
ax.set_xlabel('Sepal Length')
ax.set_ylabel('Sepal Width')
ax.set_zlabel('Petal Length')
plt.title("3D Scatter Plot of Iris Dataset")
plt.colorbar(scatter, ax=ax, label='Species')
plt.show()

# Strip plot
plt.figure(figsize=(8, 6))
sns.stripplot(data=df, x="species", y="petal_length", jitter=True, palette="Set2")
plt.title("Strip Plot of Petal Length by Species")
plt.show()

# Violin plot
plt.figure(figsize=(8, 6))
sns.violinplot(data=df, x="species", y="petal_width", palette="pastel")
plt.title("Violin Plot of Petal Width by Species")
plt.show()

# Density plot
plt.figure(figsize=(8, 6))
sns.kdeplot(data=df, x="sepal_length", hue="species", fill=True)
plt.title("Density Plot of Sepal Length")
plt.show()

# Heatmap of feature correlations
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Heatmap of Feature Correlations")
plt.tight_layout()
plt.show()

# Prediction function
def classify_iris(sample):
    sample = np.array(sample).reshape(1, -1)
    prediction = model.predict(sample)
    return label_encoder.inverse_transform(prediction)[0]

# Example prediction
new_sample = [5.1, 3.5, 1.4, 0.2]
predicted_species = classify_iris(new_sample)
print(f"\n Predicted species for sample {new_sample}: {predicted_species}")
