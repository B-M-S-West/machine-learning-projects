import marimo

__generated_with = "0.14.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return


@app.cell
def _():
    import logging
    import pickle

    from datetime import datetime
    from decimal import Decimal

    import duckdb
    import numpy as np
    import orjson
    import plotly.express as px
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    return (duckdb,)


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
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
