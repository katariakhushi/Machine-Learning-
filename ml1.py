from sklearn.datasets import load_iris
import pandas as pd
# Load the Iris dataset
iris_data = load_iris()
# Convert the dataset into a pandas DataFrame
iris_df = pd.DataFrame(data=iris_data.data, columns=iris_data.feature_names)
# Print the first five rows of the DataFrame
print(iris_df.head())