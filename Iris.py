# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 20:29:24 2023

@author: NENO
"""

import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt  # Import Matplotlib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load the data
data = pd.read_csv('Iris.csv')

# Convert species labels to numeric values (0, 1, 2)
data['Species'] = data['Species'].map({'Iris-setosa': 0, 'Iris-versicolor': 1,
                                       'Iris-virginica': 2})

# Split the data into features (X) and labels (y)
X = data.drop('Species', axis=1)
y = data['Species']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=42)

# Standardize the feature values between 0 and 1
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)








# Create a simple feedforward neural network model



#The Sequential class in Keras is used to create a sequential neural network 
#model. A sequential model is a linear stack of layers, where you define each 
#layer one after the other, and data flows sequentially through them.
model = Sequential()

#Dense is a fully connected layer, which means that each neuron in this layer 
#is connected to every neuron in the previous layer (input layer in this case).
#activation='relu' indicates that the Rectified Linear Unit (ReLU) activation 
#function will be applied to the output of this layer. ReLU is a common 
#activation function used for hidden layers in neural networks.

#Input layer
model.add(Dense(64, input_dim=5, activation='relu'))

#Hidden layer
model.add(Dense(32, activation='relu'))

# 3 output neurons for 3 classes, softmax for multi-class classification
#The activation function used here is 'softmax'. Softmax is often used in 
#multiclass classification problems as it converts the raw scores into 
#probability distributions over the classes. Each neuron in this layer 
#represents the probability of the corresponding class.

#Output layer
model.add(Dense(3, activation='softmax'))


#Adam (short for Adaptive Moment Estimation) is a popular optimization 
#algorithm commonly used for training neural networks. It combines the benefits
# of two other optimization techniques, AdaGrad and RMSProp, and is known for 
#its efficiency and effectiveness in a wide range of tasks.

#sparse_categorical_crossentropy is typically used for classification problems 
#where the target labels are integers (e.g., class indices). It computes the 
#cross-entropy loss between the predicted class probabilities and the true 
#class labels.

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=1000, batch_size=8, 
                    validation_split=0.1)




# Extract the training and validation loss from the history
train_loss = history.history['loss']
val_loss = history.history['val_loss']

# Create a plot to visualize the training and validation loss
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(train_loss) + 1), train_loss, label='Training Loss')
plt.plot(range(1, len(val_loss) + 1), val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')






    	
