# Iris-Flower-Classification-with-Neural-Networks
This GitHub repository contains Python code for a simple neural network model that classifies Iris flowers into three species (setosa, versicolor, and virginica). The code leverages popular libraries such as Pandas, TensorFlow, and Matplotlib for data handling, model creation, and visualization.

# Code Summary:

**1. Data Loading:** The Iris dataset is loaded from a CSV file and species labels are converted into numeric values (0, 1, 2).

**2. Data Preparation:** The data is split into features (X) and labels (y) and further divided into training and test sets. Feature scaling is performed to standardize values.

**3. Neural Network Model:** A feedforward neural network model is constructed using TensorFlow's Keras API. It consists of an input layer with 64 neurons and ReLU activation, a hidden layer with 32 neurons and ReLU activation, and an output layer with 3 neurons using softmax activation for multi-class classification.

**4. Model Compilation:** The model is compiled using the Adam optimizer and sparse categorical cross-entropy loss, which is appropriate for integer-encoded class labels. Accuracy is chosen as the evaluation metric.

**5. Model Training:** The model is trained with 200 epochs and a batch size of 8. A validation split of 10% is used for monitoring training progress.

**6. Visualization:** Training and validation loss curves are plotted to visualize model performance during training.

**7. Model Evaluation:** Finally, the trained model is evaluated on the test data, and the test accuracy is printed.

This code provides a clear example of building a neural network for multi-class classification tasks, making it a valuable resource for beginners and anyone interested in Iris flower classification using machine learning.

# Results

## Model
![image](https://github.com/nenomg/Iris-Flower-Classification-with-Neural-Networks/assets/105873794/77776e66-31e0-4423-a496-97415736bde8)

## Training
- 5 EPOCHS
  ![image](https://github.com/nenomg/Iris-Flower-Classification-with-Neural-Networks/assets/105873794/f959b2d9-1057-4926-aada-29b7ab534fc9)

- 20 EPOCHS
  ![image](https://github.com/nenomg/Iris-Flower-Classification-with-Neural-Networks/assets/105873794/7d407ae7-809c-4363-abec-80015c27e830)

- 50 EPOCHS
  ![image](https://github.com/nenomg/Iris-Flower-Classification-with-Neural-Networks/assets/105873794/f8302b3c-d83b-48f6-8ff0-d49f062a70ef)

- 100 EPOCHS
  ![image](https://github.com/nenomg/Iris-Flower-Classification-with-Neural-Networks/assets/105873794/d75e75ba-43eb-4c30-ada2-3eedfa479454)

- 500 EPOCHS
  ![image](https://github.com/nenomg/Iris-Flower-Classification-with-Neural-Networks/assets/105873794/7ad6fb2b-ad46-439a-93dd-b73594858192)

- 1000 EPOCHS
  ![image](https://github.com/nenomg/Iris-Flower-Classification-with-Neural-Networks/assets/105873794/7029127b-de4b-468f-96c8-6af61809aa0b)

## Coclusions
The more epochs for training the model, the better it gets predicting the test dataset. With only 2 epochs I get a 96% over the test data, the rest of them gets a 100% success.
In conclusion, the model is able to guess the 3 species of the Iris Flowers.
