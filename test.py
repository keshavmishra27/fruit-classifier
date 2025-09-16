from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from fruit_classifier_train import train_gen

# Load trained model
model = load_model(r'archive (1)\dataset-4\fruit_tf_model.h5')

# Convert train class indices to label list
fruit_arr = [k for k, v in sorted(train_gen.class_indices.items(), key=lambda item: item[1])]

#function to predict images
def predict_fruit(img_path):
    img = image.load_img(img_path, target_size=(224,224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array)
    class_idx = np.argmax(preds)
    return fruit_arr[class_idx]

for _ in range(1,31):
    img_path = rf'archive (1)\dataset-4\predict\testing\{_}.jpeg'
    print(f"Predicted fruit: {predict_fruit(img_path)}")
