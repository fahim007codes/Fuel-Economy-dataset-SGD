import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

# Load dataset
fuel = pd.read_csv('../input/dl-course-data/fuel.csv')

X = fuel.copy()
# Remove target
y = X.pop('FE')

# Preprocessing
preprocessor = make_column_transformer(
    (StandardScaler(), make_column_selector(dtype_include=np.number)),
    (OneHotEncoder(sparse=False), make_column_selector(dtype_include=object)),
)

X = preprocessor.fit_transform(X)
y = np.log(y)  # log transform target instead of standardizing

input_shape = [X.shape[1]]
print("Input shape: {}".format(input_shape))

# Define model
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=input_shape),
    layers.Dense(128, activation='relu'),    
    layers.Dense(64, activation='relu'),
    layers.Dense(1),
])

# Compile model
model.compile(
    optimizer="adam",
    loss="mae",
)

# Train model
history = model.fit(
    X, y,
    batch_size=128,
    epochs=200,
)

# Hyperparameters
learning_rate = 0.05
batch_size = 32
num_examples = 256

# Animate SGD (Function not defined in the provided code)
animate_sgd(
    learning_rate=learning_rate,
    batch_size=batch_size,
    num_examples=num_examples,
    steps=50,  # total training steps (batches seen)
    true_w=3.0,  # the slope of the data
    true_b=2.0,  # the bias of the data
)
