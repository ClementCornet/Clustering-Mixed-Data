### To Add a new Algorithm to the App

Write a Python file, with a function taking a pandas DataFrame as a parameter, and returning this dataframe, with a 'cluster' column.
Write a Markdown File explaining how this new algorithm works.
Import your python file in `template.py`.
In `template.go_to_page` function, add an if case:

```
if algo_name == '<your_algo>':
        algo_page(algo_name, <your_py_file>.process, '<path_to_your_markdown>')

```

Then go to `app.py`, and add '<your_algo>' to 'choice' selectbox on line 21.

### To Add a CSV dataset

Add it to `data` folder. Then, in `utilities/helpers.py` add the name of your file to 'select' selectbox line 33.