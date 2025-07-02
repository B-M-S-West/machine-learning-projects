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
def _():
    return


@app.cell
def _(mo):
    mo.md(r"""The previous section allows us to analyse the label distribution and see how balanced the dataset is and if possible to train on it. 249 postives to 79 negatives means we have enough of each to go on and evaluate our classifier's performance. We will need to randomly split the dataset into two sets, the training and testing sets, which simulate learning data and prediction data. It's a small dataset, so 30% will be appropriate.""")
    return


@app.cell
def _():
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import MultinomialNB
    return MultinomialNB, train_test_split


@app.cell
def _(X, Y, train_test_split):
    # Split the data set and check the size
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
    print(len(Y_train), len(Y_test))
    return X_test, X_train, Y_test, Y_train


@app.cell
def _(MultinomialNB, X_test, X_train, Y_train):
    clf = MultinomialNB(alpha=1.0, fit_prior=True)
    clf.fit(X_train, Y_train)
    prediction_prob = clf.predict_proba(X_test)
    print(prediction_prob[0:10])
    return (clf,)


@app.cell
def _(X_test, clf):
    prediction = clf.predict(X_test)
    print(prediction[:10])
    return


@app.cell
def _(X_test, Y_test, clf):
    accuracy = clf.score(X_test, Y_test)
    print(f'The accuracy is: {accuracy*100:.1f}%')
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
