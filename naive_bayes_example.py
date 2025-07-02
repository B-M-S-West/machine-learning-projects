import marimo

__generated_with = "0.14.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    return mo, np


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Naive Bayes from Scratch
    This first secton follows on from a practical example of calculating movie preferences by hand. This is now using numpy to implement naive bayes from scratch.
    """
    )
    return


@app.cell
def _(np):
    X_train = np.array([
        [0, 1, 1],
        [0, 0, 1],
        [0, 0, 0],
        [1, 1, 0]
    ])

    Y_train = ['Y', 'N', 'Y', 'Y']
    X_test = np.array([[1, 1, 0]])
    return X_test, X_train, Y_train


@app.function
def get_label_indices(labels):
    """
    Group samples based on their labels and return indices
    @param labels: list of labels
    @return: dict, {class1: [indices], class2: [indices]}
    """
    from collections import defaultdict
    label_indices = defaultdict(list)
    for index, label in enumerate(labels):
        label_indices[label].append(index)
    return label_indices


@app.cell
def _(Y_train):
    label_indices = get_label_indices(Y_train)
    print('label_indices:\n', label_indices)
    return (label_indices,)


@app.function
def get_prior(label_indices):
    """
    Compute prior based on training samples
    @param label_indices: grouped sample indices by class
    @return: dictionary, with class label as key, corresponding prior as the value
    """
    prior = {label: len(indices) for label, indices in label_indices.items()}
    total_count = sum(prior.values())
    for label in prior:
        prior[label] /= total_count
    return prior


@app.cell
def _(label_indices):
    prior = get_prior(label_indices)
    print('Prior:', prior)
    return (prior,)


@app.function
def get_likelihood(features, label_indices, smoothing=0):
    """
    Compute likelihood based on training samples
    @param features: matrix of features
    @param label_indices: grouped sample indices by class
    @param smoothing: integer, additive smoothing parameter
    @return: dictionary, with class as key, corresponding conditional probability P(feature|class) vector as value
    """
    likelihood = {}
    for label, indices in label_indices.items():
        likelihood[label] = features[indices, :].sum(axis=0) + smoothing
        total_count = len(indices)
        likelihood[label] = likelihood[label] / (total_count + 2 * smoothing)
    return likelihood


@app.cell
def _(X_train, label_indices):
    smoothing = 1
    likelihood = get_likelihood(X_train, label_indices, smoothing)
    print('Likelihood:\n', likelihood)
    return (likelihood,)


@app.function
def get_posterior(X, prior, likelihood):
    """
    Compute posterior of testing samples, based on prior and likelihood
    @param X: testing samples
    @param prior: dictionary, with class label as key, corresponding prior as the value
    @param likelihood: dictionary, with class label as key, corresponding conditional probability vector as value
    @return: dictionary, with class label as key, corresponding posterior as value
    """
    posteriors = []
    for x in X:
        # posterior is propotional to prior * likelihood
        posterior = prior.copy()
        for label, likelihood_label in likelihood.items():
            for index, bool_value in enumerate(x):
                posterior[label] *= likelihood_label[index] if bool_value else (1 - likelihood_label[index])
            # normalise so that all sums up to 1
        sum_posterior = sum(posterior.values())
        for label in posterior:
            if posterior[label] == float('inf'):
                posterior[label] = 1.0
            else:
                posterior[label] /= sum_posterior
        posteriors.append(posterior.copy())
    return posteriors


@app.cell
def _(X_test, likelihood, prior):
    posterior = get_posterior(X_test, prior, likelihood)
    print('Posterior:\n', posterior)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Implementing Naive Bayes with scikit-learn
    Following on from coding it from scratch to make predictions. This not uses the BernoulliNB modeul from scikit-learn to make the prediction.
    """
    )
    return


@app.cell
def _():
    from sklearn.naive_bayes import BernoulliNB
    return (BernoulliNB,)


@app.cell
def _(BernoulliNB):
    clf = BernoulliNB(alpha=1.0, fit_prior=True)
    return (clf,)


@app.cell
def _(X_train, Y_train, clf):
    clf.fit(X_train, Y_train)
    return


@app.cell
def _(X_test, clf):
    pred_prob = clf.predict_proba(X_test)
    print('[scikit-learn] Predicted probablities:\n', pred_prob)
    return


@app.cell
def _(X_test, clf):
    pred = clf.predict(X_test)
    print('[scikit-learn] Prediction:', pred)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
