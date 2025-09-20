# fruit-classifier

dataset used:https://www.kaggle.com/datasets/jeckyhindrawan/fruit-classification

Here is a README file for your fruit classification project using MobileNetV2 and Keras. This template is based on standard deep learning project structures and tailored to your code and dataset setup.

Fruit Classification Using MobileNetV2
Overview
This project builds and trains a deep learning model for classifying images of fruits using transfer learning with MobileNetV2 in TensorFlow/Keras. The model is capable of identifying 10 different fruit classes from image data, and provides prediction output for new/unlabeled images.

Dataset
Train/Test Directories:
Images are structured into train, test, and predict folders.

Each class has its own subdirectory in train and test (e.g., Apple, Banana, etc.).

For prediction, images are placed in a subdirectory inside predict (e.g., predict/testing/).

Classes:

Apple

Avocado

Banana

Cherry

Kiwi

Mango

Orange

Pineapple

Strawberries

Watermelon

Model Architecture
Base Model: MobileNetV2 (pre-trained on ImageNet)

Custom Layers:

GlobalAveragePooling2D

Dense (128 units, ReLU)

Dropout (0.3)

Output Dense (10 units, Softmax)

Optimizer: RMSprop with fine-tuned learning rate

Loss Function: Sparse Categorical Crossentropy

Callbacks:

EarlyStopping

ModelCheckpoint

ReduceLROnPlateau

Setup & Usage
1. Install Dependencies
bash
pip install tensorflow pandas numpy matplotlib
2. Prepare Dataset
Place training images in archive (1)/dataset-4/train/[class_name]/

Place testing images in archive (1)/dataset-4/test/[class_name]/

Place prediction images in a subfolder:
archive (1)/dataset-4/predict/testing/

3. Train Model
Run the Python script (e.g., fruit_classifier.py).
The script trains the model and saves output graphs for loss and accuracy.

4. Predict New Images
Place images for prediction in archive (1)/dataset-4/predict/testing/

The script loads these and prints predicted class labels for each image.

5. Save/Load Model
The trained model is saved in HDF5 or native Keras format:

python
final_model.save('archive (1)/dataset-4/fruit_tf_model.h5')
# For native Keras format:
# final_model.save('archive (1)/dataset-4/fruit_tf_model.keras')
Output
Training/Validation Accuracy and Loss plots

Predicted fruit class for each image in predict/testing/

Model checkpoint file (fruit_tf_model.h5)

Sample Prediction Code
python
test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=1,
    class_mode=None,
    shuffle=False
)
preds = final_model.predict(test_gen, steps=len(test_gen))
predicted_indices = np.argmax(preds, axis=1)
predicted_labels = [fruit_arr[idx] for idx in predicted_indices]
for fname, label in zip(test_gen.filenames, predicted_labels):
    print(f"{fname}: {label}")
Applications
Automated sorting for agriculture/food tech

Grocery store fruit identification

Dietary tracking apps

