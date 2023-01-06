import pickle
# from sklearn.datasets import fetch_openml

# # Load the dataset
# X, y = fetch_openml(name='cifar-10', return_X_y=True)

from sklearn.datasets import load_digits

X, y = load_digits(return_X_y=True)

# Save the data to a file
with open('digits.pkl', 'wb') as f:
  data = {'X': X, 'y': y}
  pickle.dump(data, f)