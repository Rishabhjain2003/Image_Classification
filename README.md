# Image_Classification
## Image Classification using Logistic Regression (PyTorch)

This repository contains code for a basic image classification model using Logistic Regression implemented with PyTorch's `nn.Linear` module. The model is trained and evaluated on the MNIST handwritten digit dataset.

**Requirements:**

* Python 3.x
* PyTorch
* torchvision
* matplotlib

**Code Structure:**

* `image_classification_pytorch.ipynb`: Jupyter Notebook containing the complete implementation.

**Functionality:**

1. **Data Loading:**
    * Loads the MNIST dataset using `torchvision.datasets.MNIST`.
    * Splits the dataset into training and testing sets.
    * Preprocesses the images by converting them to tensors.

2. **Model Definition:**
    * Defines a simple Logistic Regression model using `nn.Linear`.
    * The model takes a flattened image (784 pixels) as input and outputs probabilities for each of the 10 digits (0-9).

3. **Training:**
    * Iterates through the training data in epochs.
    * For each image-label pair:
        * Reshapes the image to a vector.
        * Calculates the model's output (logits).
        * Calculates the cross-entropy loss using `F.cross_entropy`.
        * Calculates accuracy using a custom `accuracy` function.
    * Backpropagates the loss and updates model weights using SGD optimizer.
    * Prints the training loss and accuracy for each epoch.

4. **Evaluation:**
    * Evaluates the model on the test set using the same steps as training.
    * Prints the test loss and accuracy.

5. **Prediction:**
    * Defines a `predict_image` function that takes an image and the model as input.
    * Preprocesses the image and feeds it to the model.
    * Returns the predicted digit class based on the highest probability.

**Further Improvements:**

* Explore using more sophisticated activation functions (e.g., ReLU) instead of the default identity function in Logistic Regression.
* Implement data augmentation techniques to improve model generalization.
* Experiment with different hyperparameters (learning rate, epochs) to optimize performance.
* Consider using more advanced neural network architectures like CNNs for better accuracy on image classification tasks.

This code provides a basic example of image classification with Logistic Regression in PyTorch. It serves as a starting point for further exploration and experimentation with deep learning models.
