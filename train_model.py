from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os

# Parse arguments
argparser = argparse.ArgumentParser()
argparser.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
argparser.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output loss/accuracy plot")
argparser.add_argument("-m", "--model", type=str, default="covid.model",
	help="path to output loss/accuracy plot")
argparser.add_argument("-r", "--learningrate", type=float, default=1e-3,
	help="initial learning rate")
argparser.add_argument("-e", "--epochs", type=int, default=25,
	help="number of epochs to train for")
argparser.add_argument("-b", "--batchsize", type=int, default=16,
	help="batch size")
argparser.add_argument("-a", "--averagepooling", type=int, default=2,
	help="batch size")
argparser.add_argument("-o", "--output", type=str, default="",
	help="batch size")
args = vars(argparser.parse_args())

INIT_LEARNING_RATE = args["learningrate"]	# Initial learning rate
EPOCHS = args["epochs"]						# Epochs to train for
BATCH_SIZE = args["batchsize"]				# Batch size
POOLING = args["averagepooling"]
OUTPUT_FILE = args["output"]


run_info = f'Hyperparameters:\nInitial Learning Rate: {INIT_LEARNING_RATE}\nEpochs: {EPOCHS}\nBatch Size: {BATCH_SIZE}\nAverage Pooling: {POOLING}x{POOLING}\nOutput File: {OUTPUT_FILE}'
print(run_info)
# Learning Rate: 1e-2, 1e-3, 1e-4
# Epochs: 15, 25, 35
# Batch Sizes: 8, 16, 32
# Average Pooling: 4x4, 2x2

print("[LOG] Parsing images...")
images = []	# List to hold images
labels = []	# List to hold labels -> 'normal' or 'covid'

# Get paths to images from dataset directory
image_paths = list(paths.list_images(args["dataset"]))

for img_path in image_paths:
	# Extract image label from image path: subdirectory name
	label = img_path.split(os.path.sep)[-2]
	
	# Read and resize image
	image = cv2.imread(img_path)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = cv2.resize(image, (224, 224))

	images.append(image)
	labels.append(label)

# Scale pixel intensities to [0, 1]
images = np.array(images) / 255.0
labels = np.array(labels)

# One-hot encode labels
label_bin = LabelBinarizer()
labels = label_bin.fit_transform(labels)
labels = to_categorical(labels)

# Split data and labels into 20% testing set, 80% training set
(trainX, testX, trainY, testY) = train_test_split(images, labels, test_size=0.20, stratify=labels, random_state=42)

# Training data augmentor
train_augmentor = ImageDataGenerator(rotation_range=15, fill_mode="nearest")


# Create VGG16 model without top layer
base_model = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

# Construct the head layer which will go on the base model
head_layer = base_model.output
head_layer = AveragePooling2D(pool_size=(POOLING, POOLING))(head_layer)
head_layer = Flatten(name="flatten")(head_layer)
head_layer = Dense(64, activation="relu")(head_layer)
head_layer = Dropout(0.5)(head_layer)
head_layer = Dense(2, activation="softmax")(head_layer)


# Connect the two layers into one model
model = Model(inputs=base_model.input, outputs=head_layer)

# Freeze layers in base model to prevent them from being updated during training
for layer in base_model.layers:
	layer.trainable = False

print("[LOG] Compiling model...")
# Adam optimization (stochastic gradient descent)
adam_opt = Adam(lr=INIT_LEARNING_RATE, decay=INIT_LEARNING_RATE / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=adam_opt, metrics=["accuracy"])

print("[LOG] Training network head...")
Hist = model.fit(
	x=train_augmentor.flow(trainX, trainY, batch_size=BATCH_SIZE),
	steps_per_epoch=len(trainX) // BATCH_SIZE,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BATCH_SIZE,
	epochs=EPOCHS)

print("[LOG] evaluating network...")
predictions = model.predict(testX, batch_size=BATCH_SIZE)

# Select label with highest probability
predictions = np.argmax(predictions, axis=1)

print(classification_report(testY.argmax(axis=1), predictions, target_names=label_bin.classes_))


# Get error matrix to determine accuracy, sensitivity, etc.
error_matrix = confusion_matrix(testY.argmax(axis=1), predictions)
total = sum(sum(error_matrix))
accuracy = (error_matrix[0, 0] + error_matrix[1, 1]) / total
sensitivity = error_matrix[0, 0] / (error_matrix[0, 0] + error_matrix[0, 1])
specificity = error_matrix[1, 1] / (error_matrix[1, 0] + error_matrix[1, 1])

print(run_info)
print(error_matrix)
print("accuracy: {:.4f}".format(accuracy))
print("sensitivity: {:.4f}".format(sensitivity))
print("specificity: {:.4f}".format(specificity))

with open(OUTPUT_FILE, 'w+') as output_file:
	output_file.write(run_info + '\n')
	np.savetxt(output_file, np.column_stack(error_matrix))
	output_file.write("accuracy: {:.4f}".format(accuracy) + '\n')
	output_file.write("sensitivity: {:.4f}".format(sensitivity) + '\n')
	output_file.write("specificity: {:.4f}".format(specificity) + '\n')

N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), Hist.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), Hist.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), Hist.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), Hist.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on COVID-19 Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])

print("[LOG] saving COVID-19 detector model...")
model.save(args["model"], save_format="h5")