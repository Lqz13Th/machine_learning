import ydf
import numpy as np
import pandas as pd

pd.set_option("display.max_rows", 5000)
pd.set_option("expand_frame_repr", False)

path = "https://storage.googleapis.com/download.tensorflow.org/data/palmer_penguins/penguins.csv"
dataset = pd.read_csv(path)
label = "species"

print(dataset.head(3))

# Use the ~20% of the examples as the testing set
# and the remaining ~80% of the examples as the training set.
np.random.seed(1)
is_test = np.random.rand(len(dataset)) < 0.2

train_dataset = dataset[~is_test]
test_dataset = dataset[is_test]

print("Training examples: ", len(train_dataset))
# >> Training examples: 272

print("Testing examples: ", len(test_dataset))
# >> Testing examples: 72

model = ydf.CartLearner(label=label, min_examples=7).train(train_dataset)

model.plot_tree()

train_evaluation = model.evaluate(train_dataset)
print("train accuracy:", train_evaluation.accuracy)
# >> train accuracy:  0.9338

test_evaluation = model.evaluate(test_dataset)
print("test accuracy:", test_evaluation.accuracy)
# >> test accuracy:  0.9167

print(model.evaluate(test_dataset).accuracy)

