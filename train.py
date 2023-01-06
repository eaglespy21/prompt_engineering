import pickle
from sklearn.neural_network import MLPClassifier

# Load the data from the file
with open('preprocesseddigits.pkl', 'rb') as f:
  data = pickle.load(f)
  X_train = data['X_train']
  y_train = data['y_train']
  X_test = data['X_test']
  y_test = data['y_test']

# Create the model
model = MLPClassifier(hidden_layer_sizes=(50,), max_iter=10, alpha=1e-4,
                      solver='sgd', verbose=10, tol=1e-4, random_state=1,
                      learning_rate_init=.1)

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
score = model.score(X_test, y_test)
print(f'Test score: {score:.2f}')

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)