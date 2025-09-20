#Fruit Classification Using MobileNet
A deep learning project for multi-class fruit image classification using Keras and MobileNet. This solution employs fine-tuning, image augmentation, and custom prediction scripts to achieve robust results with small and medium-sized datasets.

#Features
Image preprocessing and augmentation (rotation, shift, zoom, flip)

MobileNet transfer learning with fine-tuning

Customizable hyperparameters (optimizer, epochs, dropout, dense units)

Standalone scripts for training and batch predictions on new images

Saves best weights and visualization plots for monitoring performance

#Dataset Structure
<pre><code>
archive (1)/dataset-4/
├── train/
│   ├── Apple/
│   ├── Avocado/
│   └── ... (other fruit classes)
├── test/
│   ├── Apple/
│   ├── Avocado/
│   └── ... (other fruit classes)
└── predict/
    └── testing/
        ├── 1.jpeg
        ├── 2.jpeg
        └── ...
</code></pre>

#Installation
\\\
python 
pip install tensorflow pandas numpy matplotlib
\\\

#Training the Model
\\\
python
python fruit_classifier.train.py
\\\
Trains a MobileNet-based model for fruit classification.

Saves the trained model as archive (1)/dataset-4/fruit_tf_model.h5.

Produces and shows training/validation loss and accuracy plots.

#Hyperparameters (can be changed in script):

Learning rate

Number of epochs

Dropout rate

Dense layer units

Number of unfrozen layers for fine-tuning

#Class Alignment
fruit_arr = [k for k, v in sorted(train_gen.class_indices.items(), key=lambda item: item[1])]

#Testing and Inference
\\\ python 
python test.py
\\\
Loads the trained model and prints fruit predictions for images in predict/testing/1.jpeg through predict/testing/30.jpeg

Uses predict_fruit(img_path) to process and predict each input image

#Example output:
\\\
python
Predicted fruit: Apple
Predicted fruit: Mango
\\\

#test.py highlights:
Loads class order dynamically from training

Accepts image path, resizes and normalizes to fit model

Predicts and prints class label for each fil

#Visualization
After training, check your logs/plots for .plot() output showing:

Training vs. validation loss

Training vs. validation accuracy

#Model Saving
Model is saved as HDF5 (fruit_tf_model.h5).
Tip: Keras recommends:

\\\
pythpn
final_model.save('fruit_tf_model.keras')
\\\

#Acknowledgements
[Keras MobileNet Documentation]:https://keras.io/api/applications/mobilenet/
[Fruit Classification Datasets]:https://www.kaggle.com/datasets/jeckyhindrawan/fruit-classification




