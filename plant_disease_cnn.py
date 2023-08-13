import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import numpy as np
import seaborn as sns


def plot_training_history(history):
    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(true_labels, predictions, classes):
    cm = confusion_matrix(true_labels, predictions)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

# Dataset part
base_directory = '.'

all_image_paths = []
all_labels = []

# Navigate through each directory
for dir_name in os.listdir(base_directory):
    dir_path = os.path.join(base_directory, dir_name)
    
    # Ensure it's a directory
    if os.path.isdir(dir_path):
        # Parse the directory name
        plant_name = dir_name.split('-')[0].strip()
        disease_name = dir_name.split('-')[1].strip()
        
        # Create a label from plant and disease
        label = f"{plant_name}_{disease_name}"
        
        # Add each image path in this directory to the list
        for img_name in os.listdir(dir_path):
            img_path = os.path.join(dir_path, img_name)
            
            # Ensure it's an image file and append to the lists
            if img_name.endswith(('.jpg', '.jpeg', '.png')):
                all_image_paths.append(img_path)
                all_labels.append(label)

# You can now pair them if required using zip function
data = list(zip(all_image_paths, all_labels))
df = pd.DataFrame(data, columns=['image_path', 'label'])

class_counts = df['label'].value_counts()
print(class_counts[class_counts == 1])
classes_to_remove = class_counts[class_counts == 1].index.tolist()
df = df[~df['label'].isin(classes_to_remove)]

reduced_df = df.groupby('label', group_keys=False).apply(lambda x: x.sample(frac=0.1))

# After creating reduced_df
class_counts_after_reduction = reduced_df['label'].value_counts()
single_sample_labels = class_counts_after_reduction[class_counts_after_reduction == 1].index

# Remove single-sample labels from the dataframe
reduced_df = reduced_df[~reduced_df['label'].isin(single_sample_labels)]

# Now, split the reduced dataset into training and validation datasets
train_data, val_data = train_test_split(reduced_df, test_size=0.2, random_state=42, stratify=reduced_df['label'])

# 1. Find unique classes in train and validation
train_classes = train_data['label'].unique()
val_classes = val_data['label'].unique()
# 2. Find classes that are in training set but not in validation set
missing_classes = set(train_classes) - set(val_classes)

# 3. For each missing class, move one image from train to validation
for missing_class in missing_classes:
    sample_to_move = train_data[train_data['label'] == missing_class].sample()
    val_data = pd.concat([val_data, sample_to_move])
    train_data = train_data.drop(sample_to_move.index)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

val_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_data,
    x_col="image_path",
    y_col="label",
    target_size=(224, 224), # VGG16 default size
    batch_size=64,
    class_mode='categorical')


# Save class indices to a json file
import json
with open('class_indices.json', 'w') as f:
    json.dump(train_generator.class_indices, f)

validation_generator = val_datagen.flow_from_dataframe(
    dataframe=val_data,
    x_col="image_path",
    y_col="label",
    target_size=(224, 224),
    batch_size=64,
    class_mode='categorical')

print(len(train_generator.class_indices))
print(len(validation_generator.class_indices))

# Load the VGG16 model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the layers of VGG16
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers
x = base_model.output
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
predictions = Dense(len(train_generator.class_indices), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=70,
    validation_data=validation_generator,
    validation_steps=len(validation_generator))

model_path = "my_model.h5"
model.save(model_path)

# 1. Plot the training and validation accuracy and loss.
plot_training_history(history)

# 2. Predict the classes of the validation dataset.
validation_generator.reset()  # Reset the generator to be sure of the ordering
predictions = model.predict(validation_generator, steps=len(validation_generator))
predicted_classes = np.argmax(predictions, axis=1)

# Getting true classes
true_classes = validation_generator.classes
class_labels = list(validation_generator.class_indices.keys())   

# 3. Create a confusion matrix based on the true labels and predicted labels.
plot_confusion_matrix(true_classes, predicted_classes, class_labels)

# 4. Print classification metrics including the F1 score.
print(classification_report(true_classes, predicted_classes, target_names=class_labels))

f1 = f1_score(true_classes, predicted_classes, average='macro')
print(f"F1 Score: {f1:.2f}")
