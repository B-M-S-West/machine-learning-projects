import marimo

__generated_with = "0.14.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import logging
    import pickle

    from datetime import datetime
    from decimal import Decimal

    import duckdb
    import sqlglot
    import numpy as np
    import orjson
    import plotly.express as px
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    return (
        Decimal,
        RandomForestClassifier,
        duckdb,
        pickle,
        px,
        train_test_split,
    )


@app.cell
def _(duckdb):
    # Read the csv data from an external location and exclude records with null values and alter column type
    def process_palmerpenguins_data(duckdb_conn):
        duckdb_conn.read_csv(
            "http://blobs.duckdb.org/data/penguins.csv"
        ).filter(
            "columns(*)::text != 'NA'"
        ).filter(
            "columns(*) is not null"
        ).select(
            "*, row_number() over () as observation_id"
        ).to_table(
            "penguins_data"
        )

        duckdb_conn.sql(
            "alter table penguins_data alter bill_length_mm set data type decimal(5, 2)"
        )
        duckdb_conn.sql(
            "alter table penguins_data alter bill_depth_mm set data type decimal(5, 2)"
        )
        duckdb_conn.sql(
            "alter table penguins_data alter body_mass_g set data type integer"
        )
        duckdb_conn.sql(
            "alter table penguins_data alter flipper_length_mm set data type integer"
        )

    duckdb_conn = duckdb.connect()

    process_palmerpenguins_data(duckdb_conn=duckdb_conn)
    return (duckdb_conn,)


@app.cell
def _(duckdb_conn, px):
    # plot species and island
    px.bar(
        duckdb_conn.table("penguins_data").aggregate(
            "species, island, count(*) as number_of_observations"
        ).order(
            "island, species"
        ).df(),
        x="island",
        y="number_of_observations", 
        color="species", 
        title="Palmer Penguins Observations", 
        barmode="group", 
        labels={
            "number_of_observations": "Number of Observations",
            "island": "Island"
        }
    )
    return


@app.cell
def _(duckdb_conn, px):
    # plot features per species
    px.scatter(
        duckdb_conn.table("penguins_data").df(), 
        x="bill_length_mm", 
        y="bill_depth_mm", 
        size="body_mass_g", 
        color="species", 
        title="Penguins Observations, bill length and depth, per species", 
        labels={
            "bill_length_mm": "Bill Length in mm", 
            "bill_depth_mm": "Bill Depth in mm"
        }
    )
    return


@app.cell
def _(duckdb_conn):
    # analyze the data
    duckdb_conn.table("penguins_data").describe().df()
    return


@app.cell
def _(duckdb_conn):
    # instead of label encoding, we create reference tables
    def process_reference_data(duckdb_conn):
        for feature in ["species", "island"]:
            duckdb_conn.sql(f"drop table if exists {feature}_ref")
            (
                duckdb_conn.table("penguins_data")
                .select(feature)
                .unique(feature)
                .row_number(
                    window_spec=f"over (order by {feature})", projected_columns=feature
                )
                .select(f"{feature}, #2 - 1 as {feature}_id")
                .to_table(f"{feature}_ref")
            )
            duckdb_conn.table(f"{feature}_ref").show()

    process_reference_data(duckdb_conn)
    return


@app.cell
def _(train_test_split):
    def train_split_data(selection_query):
        X_df = selection_query.select("""
            bill_length_mm,
            bill_depth_mm,
            flipper_length_mm,
            body_mass_g,
            island_id,
            observation_id,
            species_id
        """).order("observation_id").df()
        y_df = [
            x[0]
            for x in selection_query.order("observation_id").select("species_id").fetchall()
        ]

        num_test = 0.30
        return train_test_split(X_df, y_df, test_size=num_test)

    return (train_split_data,)


@app.cell
def _(RandomForestClassifier, pickle, train_split_data):
    def get_model(selection_query):
        X_train, X_test, y_train, y_test = train_split_data(selection_query)

        model = RandomForestClassifier(n_estimators=1, max_depth=2, random_state=5)

        model.fit(X_train.drop(["observation_id", "species_id"], axis=1).values, y_train)

        pickle.dump(model, open("duckdb_ml/model/penguin_model.sav", "wb"))

        print(f" Accuracy score is: {model.score( 
            X_test.drop(["observation_id", "species_id"], axis=1).values, y_test
        )}")
    return (get_model,)


@app.cell
def _(duckdb_conn, get_model, pickle):
    selection_query = (
        duckdb_conn.table("penguins_data")
        .join(duckdb_conn.table("island_ref"), condition="island")
        .join(duckdb_conn.table("species_ref"), condition="species")
    )
    get_model(selection_query)

    model = pickle.load(open("duckdb_ml/model/penguin_model.sav", "rb"))
    return model, selection_query


@app.cell
def _(duckdb_conn, model, selection_query):
    # get predictions with pandas and duckb in python

    predicted_df = selection_query.select(
        "bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, island_id, observation_id, species_id"
    ).df()

    predicted_df["predicted_species_id"] = model.predict(
        predicted_df.drop(["observation_id", "species_id"], axis=1).values
    )

    (
        duckdb_conn.table("predicted_df")
        .select("observation_id", "species_id", "predicted_species_id")
        .filter("species_id != predicted_species_id")
    )
    return (predicted_df,)


@app.cell
def _(mo, predicted_df):
    _df = mo.sql(
        f"""
        -- directly with SQL
        SELECT observation_id, species_id, predicted_species_id
        FROM predicted_df
        WHERE species_id != predicted_species_id
        """
    )
    return


@app.cell
def _(Decimal, duckdb_conn, pickle, selection_query):
    # get predictions with duckdb udf, row by row
    def get_prediction_per_row(
        bill_length_mm: Decimal, bill_depth_mm: Decimal, flipper_length_mm: int, body_mass_g: int, island_id: int
    ) -> int:
        model = pickle.load(open("duckdb_ml/model/penguin_model.sav", "rb"))
        return int(
            model.predict(
                [
                    [
                        bill_length_mm, 
                        bill_depth_mm, 
                        flipper_length_mm, 
                        body_mass_g, 
                        island_id, 
                    ]
                ]
            )[0]
        )

    try:
        duckdb_conn.remove_function("predict_species_per_row")
    except Exception:
        pass
    finally:
        duckdb_conn.create_function(
            "predict_species_per_row", get_prediction_per_row, return_type=int
        )

    selection_query.select(
        """
        observation_id,
        species_id,
        predict_species_per_row(
            bill_length_mm,
            bill_depth_mm, 
            flipper_length_mm, 
            body_mass_g, 
            island_id
        ) as predicted_species_id
        """
    ).filter("species_id != predicted_species_id")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
