# 🌿 Plant Disease Detection using Transfer Learning (ResNet50)

This project implements a plant disease classification model using transfer learning with ResNet50. The model is trained on the **New Plant Diseases Dataset (Augmented)** from Kaggle, containing thousands of labeled leaf images across multiple disease categories.

The model uses a frozen ResNet50 backbone with custom classification layers and is evaluated using accuracy curves and a classification report.

---

## 📂 Dataset

**Source:** [New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)

Dataset structure used in this project:
```
New Plant Diseases Dataset(Augmented)/
├── train/
├── valid/
└── test/
```

- **Total Classes:** 38
- Images were loaded using `ImageDataGenerator`

---

## 🔧 Technologies & Libraries

This project uses:

- Python
- TensorFlow / Keras
- ResNet50 (ImageNet weights)
- NumPy and Pandas
- Matplotlib
- Scikit-learn (classification report)

---

## 🧠 Model Architecture

### ✔ ResNet50 Feature Extractor
```python
pretrained_model = tf.keras.applications.ResNet50(
    include_top=False,
    input_shape=(256, 256, 3),
    pooling='avg',
    classes=4,             # ignored because include_top=False
    weights='imagenet'
)
```

All pretrained layers are frozen:
```python
for layer in pretrained_model.layers:
    layer.trainable = False
```

### ✔ Custom Classification Head
```python
resnet_model = Sequential()
resnet_model.add(pretrained_model)
resnet_model.add(Flatten())
resnet_model.add(Dense(512, activation='relu'))
resnet_model.add(Dense(38, activation='softmax'))
```

### ✔ Compilation
```python
resnet_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

---

## 🏋️ Training Procedure

Images were loaded using `ImageDataGenerator`:
```python
# Training loader
train_set = train_datagen.flow_from_directory(
    train_dir,
    target_size=(256, 256),
    batch_size=32,
    seed=101,
    shuffle=True,
    class_mode='categorical'
)

# Validation loader
validation_set = validation_datagen.flow_from_directory(
    valid_dir,
    target_size=(256, 256),
    batch_size=32,
    shuffle=False,
    class_mode='categorical'
)

# Test loader
test_set = ImageDataGenerator().flow_from_directory(
    test_dir,
    target_size=(256, 256),
    batch_size=32,
    shuffle=False,
    class_mode='categorical'
)

# Model training
history = resnet_model.fit(
    train_set,
    validation_data=validation_set,
    epochs=10
)
```

---

## 📊 Evaluation

### ✔ Training Accuracy Plot
```python
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
# ... additional plotting code
```

### ✔ Training Loss Plot
```python
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
# ... additional plotting code
```

### ✔ Classification Report
```python
Y_pred = resnet_model.predict(test_set)
y_pred = np.argmax(Y_pred, axis=1)

print(classification_report(
    test_set.classes,
    y_pred,
    target_names=list(train_set.class_indices.keys())
))
```

This provides precision, recall, f1-score, and support for each class.

---

## 📌 Results Summary

- **Training Accuracy:** ~ ___
- **Validation Accuracy:** ~ ___
- **Test Performance:** Provided by classification report
- **Model Type:** ResNet50 (Frozen) + Dense layers
- **Input Size:** 256 × 256

---

## ▶ How to Run the Project

### 1. Install dependencies
```bash
pip install tensorflow numpy pandas matplotlib scikit-learn
```

### 2. Place the dataset
```
/kaggle/input/new-plant-diseases-dataset/New Plant Diseases Dataset(Augmented)/
```

### 3. Run the notebook

Execute all cells in `plant-disease-detection.ipynb`.

---

## 🚀 Possible Improvements

- Unfreeze deeper layers of ResNet50 (fine-tuning)
- Replace `Flatten` with `GlobalAveragePooling2D`
- Try EfficientNet or DenseNet
- Add stronger augmentation
- Deploy using Streamlit / Flask

---

## 🏁 Conclusion

This project demonstrates a practical plant disease classification system using transfer learning. ResNet50 provides strong feature extraction, while custom dense layers adapt the network to the dataset's 38 classes. The resulting model offers reliable prediction performance suitable for agricultural applications and early disease detection.

---
