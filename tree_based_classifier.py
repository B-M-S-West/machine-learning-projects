import marimo

__generated_with = "0.14.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    return np, plt


@app.cell
def _(np):
    pos_fraction= np.linspace(0.00, 1.00, 1000)
    return (pos_fraction,)


@app.cell
def _(pos_fraction):
    # Gini impurity calculated
    gini = 1 - pos_fraction**2 - (1-pos_fraction)**2
    return (gini,)


@app.cell
def _(gini, plt, pos_fraction):
    plt.plot(pos_fraction, gini)
    plt.ylim(0, 1)
    plt.xlabel('Positive fraction')
    plt.ylabel('Gini Impurity')
    plt.show()
    return


@app.cell
def _(np):
    def gini_impurity(labels):
        # Empyt set is also pure
        if len(labels) == 0:
            return 0
        # Count occurences of each label
        counts = np.unique(labels, return_counts=True)[1]
        fractions = counts / float(len(labels))
        return 1 - np.sum(fractions ** 2)
    return (gini_impurity,)


@app.cell
def _(gini_impurity):
    print(f'{gini_impurity([1, 1, 0, 1, 0]):.4f}')
    print(f'{gini_impurity([1, 1, 0, 1, 0, 0]):.4f}')
    print(f'{gini_impurity([1, 1, 1, 1]):.4f}')
    return


@app.cell
def _(np, plt):
    _pos_fraction = np.linspace(0.001, 0.999, 1000)
    ent = - (_pos_fraction * np.log2(_pos_fraction) + (1 - _pos_fraction) * np.log2(1 - _pos_fraction))
    plt.plot(_pos_fraction, ent)
    plt.ylim(0, 1)
    plt.xlabel('Positive fraction')
    plt.ylabel('Entropy')
    plt.show()
    return


@app.cell
def _(np):
    def entropy(labels):
        if len(labels) == 0:
            return 0
        counts = np.unique(labels, return_counts=True)[1]
        fractions = counts / float(len(labels))
        return - np.sum(fractions*np.log2(fractions))
    return (entropy,)


@app.cell
def _(entropy):
    print(f'{entropy([1, 1, 0, 1, 0]):.4f}')
    print(f'{entropy([1, 1, 0, 1, 0, 0]):.4f}')
    print(f'{entropy([1, 1, 1, 1]):.4f}')
    return


app._unparsable_cell(
    r"""
    criterion_function = {'gini': gini_impurity,
                         'entropy': entrop}
    def eighted_impurit
    """,
    name="_"
)


if __name__ == "__main__":
    app.run()
