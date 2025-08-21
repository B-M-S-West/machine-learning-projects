import marimo

__generated_with = "0.14.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    import pickle
    return LinearRegression, np, pd, pickle


@app.cell
def _(np):
    sizes = np.random.randint(500,35000,100)
    return (sizes,)


@app.cell
def _(sizes):
    sizes
    return


@app.cell
def _(np):
    bedrooms = np.random.randint(1,6, 100)
    return (bedrooms,)


@app.cell
def _(bedrooms):
    bedrooms
    return


@app.cell
def _(bedrooms, np, sizes):
    prices = (sizes * 0.3) + (bedrooms * 50) + np.random.randint(-200, 200, 100)
    return (prices,)


@app.cell
def _(prices):
    prices
    return


@app.cell
def _(bedrooms, pd, prices, sizes):
    data = pd.DataFrame({
        "size_sqft": sizes,
        "bedrooms": bedrooms,
        "price": prices
    })
    return (data,)


@app.cell
def _(data):
    data.head()
    return


@app.cell
def _(data):
    X = data[["size_sqft","bedrooms"]]
    return (X,)


@app.cell
def _(data):
    y = data["price"]
    return (y,)


@app.cell
def _(LinearRegression):
    model = LinearRegression()
    return (model,)


@app.cell
def _(X, model, y):
    model.fit(X, y)
    return


@app.cell
def _(np):
    sample_x = np.array([[2000,3], [1500,2]])
    return (sample_x,)


@app.cell
def _(model, sample_x):
    model.predict(sample_x)
    return


@app.cell
def _(model, pickle):
    with open("house_price_model.pkl", "wb") as f:
        pickle.dump(model, f)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
