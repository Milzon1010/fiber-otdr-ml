"""
Toy training script for 1D CNN on synthetic windows.
This is a placeholder to show project structure.
"""
import numpy as np, tensorflow as tf
from src.fiber_otdr_ml.model import build_cnn1d

def synthetic_data(n=2048, length=256):
    rng = np.random.default_rng(0)
    X = rng.normal(0,1,(n,length,1)).astype("float32")
    y = (rng.random(n) > 0.7).astype("int32")
    for i in np.where(y==1)[0]:
        j = rng.integers(32, length-32)
        X[i, j:, 0] -= rng.uniform(3,6)
    return X, y

if __name__ == "__main__":
    X, y = synthetic_data()
    model = build_cnn1d(input_len=X.shape[1], classes=2)
    model.fit(X, y, epochs=3, batch_size=64, validation_split=0.1)
    model.save("models/cnn1d_demo.h5")
