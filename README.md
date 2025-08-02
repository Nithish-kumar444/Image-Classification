# Image-Classification
# using
#🧠 CIFAR-10 using CNN (TensorFlow/Keras)

This project demonstrates how to build, train, evaluate, and visualize a Convolutional Neural Network (CNN) for image classification using the CIFAR-10 dataset. The CNN is built using TensorFlow and Keras.

---

## 📁 Dataset

The project uses the **CIFAR-10** dataset, which contains 60,000 32x32 color images across 10 classes, with 6,000 images per class.

**Class Labels:**
- `airplane`
- `automobile`
- `bird`
- `cat`
- `deer`
- `dog`
- `frog`
- `horse`
- `ship`
- `truck`

---

## 🚀 Features

- Load and normalize the CIFAR-10 dataset
- Visualize sample training images
- Build a CNN with 3 convolutional layers
- Train the model with validation
- Plot training vs. validation accuracy and loss
- Predict and visualize model outputs
- Show confidence levels for predictions

---

## 🧰 Technologies Used

- Python
- TensorFlow / Keras
- Matplotlib
- Jupyter Notebook (for step-by-step execution)

---

## 🧪 Model Architecture

Input: 32x32 RGB Image
↓
Conv2D (32 filters, 3x3) + ReLU
↓
MaxPooling2D (2x2)
↓
Conv2D (64 filters, 3x3) + ReLU
↓
MaxPooling2D (2x2)
↓
Conv2D (64 filters, 3x3) + ReLU
↓
MaxPooling2D (2x2)
↓
Flatten
↓
Dense (64 neurons, ReLU)
↓
Dense (10 neurons for classification)


---

## 📈 Training Performance

- **Optimizer**: Adam
- **Loss Function**: Sparse Categorical Crossentropy (from logits)
- **Epochs**: 10
- **Metrics**: Accuracy (Training + Validation)
- **Final Test Accuracy**: ~[depends on run, usually ~70%+]

---

## 📊 Visualizations

- Sample training images with labels
- Accuracy and Loss graphs
- Image prediction results with:
  - Actual label
  - Predicted label
  - Confidence score
  - Correct predictions in **blue**
  - Incorrect predictions in **red**

---

## 🔧 How to Run

1. Clone the repository or copy the `.ipynb` notebook.
2. Make sure you have the following installed:
   ```bash
   pip install tensorflow matplotlib

3. Open the notebook in Jupyter Lab or Google Colab.
4. Run each cell in order.
