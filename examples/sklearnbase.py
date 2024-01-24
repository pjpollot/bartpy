import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from bartpy.sklearnmodel import SklearnModel


def run(alpha, beta, n_trees):
    x = np.random.normal(0, 1, size=60)
    X = pd.DataFrame(x)
    y = np.random.normal(0, 0.1, size=60) + 2 * x + np.sin(x)
    test_x = np.linspace(-3, 3, 200)
    test_X = pd.DataFrame(test_x)
    plt.scatter(x, y, alpha=0.2, label="Data")
    model = SklearnModel(n_samples=200, n_burn=50, n_trees=n_trees, alpha=alpha, beta=beta)
    model.fit(X, y)
    predictions = model.predict(test_X)
    predictions_paths = model.predict(test_X, return_samples=True)[::20,:]
    print(predictions_paths.shape)
    plt.plot(test_x, predictions, "--", color="red", label="BART mean prediction")
    for path in predictions_paths:
        plt.plot(test_x, path, color="black", alpha=0.4)
    plt.legend() 
    plt.show()
    return model, x, y


if __name__ == "__main__":
    plt.style.use('ggplot')
    print("here")
    from datetime import datetime as dt
    print(dt.now())
    model, x, y = run(0.95, 2., 100)
    print(dt.now())
