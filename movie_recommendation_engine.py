import marimo

__generated_with = "0.14.9"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Movie Recommendation Engine
    The dataset for this task was taken from here

    https://grouplens.org/datasets/movielens/

    I have used the ml-latest-small.zip for this project for speed of process. Download and store the files in a folder called 'movie_data' to access within your project directory.
    """
    )
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    return mo, np, pd


@app.cell
def _(pd):
    # Load the ratings dataset and create the dataframe
    data_path = 'movie_data/ratings.csv'
    df = pd.read_csv(data_path, sep=',', engine='python')
    df.columns = ['user_id', 'movie_id', 'rating', 'timestamp']
    print(df)
    return (df,)


@app.cell
def _(df):
    # Calculate the number of different users and movies in the dataset
    n_users =df['user_id'].nunique()
    n_movies = df['movie_id'].nunique()
    print(f"Number of users: {n_users}")
    print(f"Number of movies: {n_movies}")
    return n_movies, n_users


@app.cell
def _(np):
    # Function to create a matrix that is equal to the number of users by the number of movies
    def load_user_rating_data(df, n_users, n_movies):
        data = np.zeros([n_users, n_movies], dtype=np.intc)
        movie_id_mapping = {}
        for user_id, movie_id, rating in zip(df['user_id'], df['movie_id'], df['rating']):
            user_id = int(user_id) - 1
            if movie_id not in movie_id_mapping:
                movie_id_mapping[movie_id] = len(movie_id_mapping)
            data[user_id, movie_id_mapping[movie_id]] = rating
        return data, movie_id_mapping

    return (load_user_rating_data,)


@app.cell
def _(df, load_user_rating_data, n_movies, n_users):
    # Constructs the matrix from the function above
    data, movie_id_mapping = load_user_rating_data(df, n_users, n_movies)
    return data, movie_id_mapping


@app.cell
def _(data, np):
    # Analyse the data distribution in order to see if there is a class imbalance issue
    values, counts = np.unique(data, return_counts=True)
    for value, count in zip(values, counts):
        print(f"Number of rating {value}: {count}")
    return


@app.cell
def _(df):
    # For this data set we see most are unknown ratings, so we take the movie with the most know ratings as target movie
    print(df['movie_id'].value_counts())
    return


@app.cell
def _(data, df, movie_id_mapping, np):
    # Set the target movie if to that with the highest amount of known ratings. Treat ratings of other movies as features. Only use rows with ratings available for the target movie so we can validate predictions.
    target_movie_id = df['movie_id'].value_counts().idxmax()
    X_raw = np.delete(data, movie_id_mapping[target_movie_id], axis=1)
    Y_raw = data[:, movie_id_mapping[target_movie_id]]
    X = X_raw[Y_raw > 0]
    Y = Y_raw[Y_raw > 0]
    print('Shape of X:', X.shape)
    print('Shape of Y:', Y.shape)
    return X, Y


@app.cell
def _(Y):
    # Movies with a rating greater than 3 we will classify as being like or recommended
    recommend = 3
    Y[Y <= recommend] = 0
    Y[Y > recommend] = 1
    n_pos = (Y == 1).sum()
    n_neg = (Y == 0).sum()
    print(f"{n_pos} positive samples and {n_neg} negative samples")
    return


@app.cell
def _(mo):
    mo.md(r"""The previous section allows us to analyse the label distribution and see how balanced the dataset is and if possible to train on it. 249 postives to 79 negatives means we have enough of each to go on and evaluate our classifier's performance. We will need to randomly split the dataset into two sets, the training and testing sets, which simulate learning data and prediction data. It's a small dataset, so 30% will be appropriate.""")
    return


@app.cell
def _():
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report, roc_auc_score
    return (
        MultinomialNB,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
        train_test_split,
    )


@app.cell
def _(X, Y, train_test_split):
    # Split the data set and check the size
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
    print(len(Y_train), len(Y_test))
    return X_test, X_train, Y_test, Y_train


@app.cell
def _(MultinomialNB, X_test, X_train, Y_train):
    # Use of Multinomial as ratings are from 1-5. Smoothing factor of 1 and prior from the training set. This trains the model and we get the predicted probabilities
    clf = MultinomialNB(alpha=1.0, fit_prior=True)
    clf.fit(X_train, Y_train)
    prediction_prob = clf.predict_proba(X_test)
    print(prediction_prob[0:10])
    return clf, prediction_prob


@app.cell
def _(X_test, clf):
    # This gives us the predicted class for the test set
    prediction = clf.predict(X_test)
    print(prediction[:10])
    return (prediction,)


@app.cell
def _(X_test, Y_test, clf):
    # Evaluate the models performance here
    accuracy = clf.score(X_test, Y_test)
    print(f'The accuracy is: {accuracy*100:.1f}%')
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Evaluating classification performance

    Other than accuracy of the model the following metrics will be look at in this section:

    - Confusion Matric

    - Recall

    - F1 Score

    - Area under the curve
    """
    )
    return


@app.cell
def _(Y_test, confusion_matrix, prediction):
    # Confusion matrix
    print(confusion_matrix(Y_test, prediction, labels=[0,1]))
    return


@app.cell
def _(Y_test, f1_score, precision_score, prediction, recall_score):
    # Precision score
    precision = precision_score(Y_test, prediction, pos_label=1)
    # Recall score
    recall = recall_score(Y_test, prediction, pos_label=1)
    # F1 Score
    f1 = f1_score(Y_test, prediction, pos_label=1)
    print(f'Precision Score is: {precision}\nRecall Score is: {recall}\nF1 Score is: {f1}')
    return


@app.cell
def _(Y_test, classification_report, prediction):
    # Generate the classification report
    report = classification_report(Y_test, prediction)
    print(report)
    return


@app.cell
def _(Y_test, np, prediction_prob):
    # Now focused on area under the curve. First need to create the Reciver Operating Characteristic curve
    pos_prob = prediction_prob[:, 1]
    thresholds = np.arange(0.0, 1.1, 0.05)
    true_pos, false_pos = [0]*len(thresholds), [0]*len(thresholds)
    for pred, y in zip(pos_prob, Y_test):
        for i, threshold in enumerate(thresholds):
            if pred >= threshold:
                # if truth and prediction are both 1
                if y == 1:
                    true_pos[i] += 1
                # if truth is 0 while prediction is 1
                else:
                    false_pos[i] += 1
            else:
                break
    return false_pos, pos_prob, true_pos


@app.cell
def _(Y_test, false_pos, true_pos):
    # Calculate the true and false positive rates for all threshold settings
    n_pos_test = (Y_test == 1).sum()
    n_neg_test = (Y_test == 0).sum()
    true_pos_rate = [tp / n_pos_test for tp in true_pos]
    false_pos_rate = [fp / n_neg_test for fp in false_pos]
    return false_pos_rate, true_pos_rate


@app.cell
def _():
    # Use matplotlib to plot the graph
    import matplotlib.pyplot as plt
    return (plt,)


@app.cell
def _(false_pos_rate, plt, true_pos_rate):
    plt.figure()
    lw = 2
    plt.plot(false_pos_rate, true_pos_rate, color='darkorange', lw=lw, label='ROC Curve')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Reciever Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

    return


@app.cell
def _(Y_test, pos_prob, roc_auc_score):
    # Calculate the ROC AUC Score
    roc = roc_auc_score(Y_test, pos_prob)
    print(f'ROC AUC Score is: {roc}')
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Tuning models with cross-valdation
    Using k-fold cross-validation to assess how a model wil generally perform in practice.
    This randomly divides the original data into k equal sized subsets, with class proportion often preserved.
    """
    )
    return


@app.cell
def _():
    from sklearn.model_selection import StratifiedKFold
    return (StratifiedKFold,)


@app.cell
def _(StratifiedKFold):
    # This divides the data into chunks. As a small dataset, using 5. Large k is more computationally expensive.
    k = 5
    k_fold = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    return (k_fold,)


@app.cell
def _():
    # Explor values for alpha, the smoothing factor, and fit_prior, whether to use prior tailored to the training data
    smoothing_factor_option = [1, 2, 3, 4, 5, 6]
    fit_prior_option = [True, False]
    auc_record = {}
    return auc_record, fit_prior_option, smoothing_factor_option


@app.cell
def _(
    MultinomialNB,
    X,
    Y,
    auc_record,
    fit_prior_option,
    k_fold,
    roc_auc_score,
    smoothing_factor_option,
):
    # Initialise a dictionary to store lists of AUC scores
    for alpha in smoothing_factor_option:
        auc_record[alpha] = {}
        for _fit_prior in fit_prior_option:
            auc_record[alpha][_fit_prior] = []

    # Collect AUC scores for each fold
    for train_indices, test_indices in k_fold.split(X, Y):
        X_train_k, X_test_k = X[train_indices], X[test_indices]
        Y_train_k, Y_test_k = Y[train_indices], Y[test_indices]
    
        for alpha in smoothing_factor_option:
            for _fit_prior in fit_prior_option:
                _clf = MultinomialNB(alpha=alpha, fit_prior=_fit_prior)
                _clf.fit(X_train_k, Y_train_k)
                _prediction_prob = _clf.predict_proba(X_test_k)
                _pos_prob = _prediction_prob[:, 1]
                _auc = roc_auc_score(Y_test_k, _pos_prob)
                auc_record[alpha][_fit_prior].append(_auc)
    return


@app.cell
def _(auc_record):
    # Print results with proper averaging
    print(f"{'Smoothing':<10} {'Fit Prior':<10} {'AUC':<10}")
    print("-" * 30)

    max_auc = 0
    best_params = None

    for smoothing, smoothing_record in auc_record.items():
        for fit_prior, auc_list in smoothing_record.items():
            current_auc = sum(auc_list) / len(auc_list)  # Average the AUC scores
            print(f'{smoothing:<10} {fit_prior:<10} {current_auc:<10.5f}')
        
            if current_auc > max_auc:
                max_auc = current_auc
                best_params = (smoothing, fit_prior)

    print(f"\nBest: Smoothing={best_params[0]}, Fit Prior={best_params[1]}, AUC={max_auc:.5f}")
    return


@app.cell
def _(MultinomialNB, X_test, X_train, Y_test, Y_train, roc_auc_score):
    # Now retrain the model with the best averaged AUC
    _clf = MultinomialNB(alpha=2, fit_prior=True)
    _clf.fit(X_train, Y_train)
    _pos_prob = _clf.predict_proba(X_test)[:, 1]
    print('AUC with the best model:', roc_auc_score(Y_test, _pos_prob))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
