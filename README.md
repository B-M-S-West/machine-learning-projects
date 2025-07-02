## Initial Project Notes
Starting off these machine learning projects in one directory to keep them together. Instead of running as Jupyter Notebooks I'm using marimo as a alternative interactive notebook.

I am also using uv to manage the dependencies.

Once dependencies have been added using:
```
uv add "named_dependency"
```
You can run the different marimo notebooks using the following:
```
uv run marimo edit main.py
```