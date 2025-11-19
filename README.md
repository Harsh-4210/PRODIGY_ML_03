Here is the README for your `PRODIGY_ML_03` repository.

-----

# PRODIGY\_ML\_03: Cat vs. Dog Image Classification

A machine learning project that implements a Support Vector Machine (SVM) to classify images as either "Cat" or "Dog." This project demonstrates how traditional machine learning algorithms can be applied to computer vision tasks through effective image preprocessing and feature extraction.

## 📌 Project Overview

Image classification is a core task in computer vision. While deep learning (CNNs) is often the go-to for this, this project explores the effectiveness of **Support Vector Machines (SVM)** for binary classification of images.

**Key Goals:**

  * Preprocess raw images (resizing, flattening, normalization).
  * Extract features suitable for a linear or non-linear classifier.
  * Train an SVM model to distinguish between cats and dogs.
  * Evaluate the model's accuracy and visualize predictions.

## 📂 Dataset

The project uses a subset of the famous **Kaggle Cats and Dogs Dataset**.

  * **Structure:** The data is typically organized into `train` and `test` directories, with separate subfolders for `cats` and `dogs`.
  * **Content:** Thousands of images of cats and dogs in various poses and environments.
  * **Note:** A processed subset might be included in the `Dataset` folder of this repo for easier replication.

## 🛠️ Technologies Used

  * **Python**: Primary programming language.
  * **OpenCV (cv2)**: For image loading, resizing, and processing.
  * **Scikit-learn**: For implementing the SVM model and evaluation metrics.
  * **NumPy**: For numerical array manipulation (flattening images).
  * **Matplotlib**: For visualizing sample images and prediction results.

## 🚀 How to Run

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Harsh-4210/PRODIGY_ML_03.git
    ```
2.  **Navigate to the project directory:**
    ```bash
    cd PRODIGY_ML_03
    ```
3.  **Install dependencies:**
    ```bash
    pip install numpy matplotlib scikit-learn opencv-python jupyter
    ```
4.  **Launch the Notebook:**
    Open the Jupyter Notebook to view the implementation.
    ```bash
    jupyter notebook "Task 03 — Image Classification Using SVM.ipynb"
    ```

## 📊 Methodology

1.  **Data Loading & Preprocessing:**

      * Images are loaded from the `Dataset` folder.
      * Each image is resized to a fixed dimension (e.g., 64x64 or 128x128) to ensure consistency.
      * Images are flattened into 1D arrays to serve as input features for the SVM.
      * Pixel values are normalized (scaled between 0 and 1) to improve model convergence.

2.  **Model Training:**

      * The data is split into training and testing sets (e.g., 80% train, 20% test).
      * An SVM classifier is initialized (often with an RBF or Linear kernel).
      * The model is trained on the flattened image vectors.

3.  **Evaluation:**

      * The model predicts labels for the test set.
      * Accuracy score and a classification report (Precision, Recall, F1-Score) are generated to assess performance.

## 📈 Results

  * The SVM model successfully learns to distinguish basic features of cats and dogs.
  * While not as powerful as a Convolutional Neural Network (CNN), it provides a solid baseline for understanding image classification fundamentals.
  * Example Output:
      * *Input Image*:     \* *Prediction*: **Cat**

## 🤝 Contributing

Contributions are welcome\! If you'd like to experiment with different kernels (Poly, Sigmoid), add PCA for dimensionality reduction, or compare this with a Logistic Regression model, feel free to fork the repo and submit a pull request.

## 📜 License

This project is open-source and available for educational purposes.
