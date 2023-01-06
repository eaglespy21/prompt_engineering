import numpy as np
from sklearn.model_selection import train_test_split

import pickle

# Load the data from the file
with open('digits.pkl', 'rb') as f:
  data = pickle.load(f)
  X = data['X']
  y = data['y']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
X_train = X_train / 16.0
X_test = X_test / 16.0

# Convert the labels to categorical format
num_classes = 10
y_train = np.eye(num_classes)[y_train]
y_test = np.eye(num_classes)[y_test]

# Save the data to a file
with open('preprocesseddigits.pkl', 'wb') as f:
  data = {'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test}
  pickle.dump(data, f)