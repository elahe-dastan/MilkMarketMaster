# Milk Market Master (M3)

> Vesper Coding Challenge

## Introduction

Based on the given data (in the CSV format), we are going to predict the Skim Milk Powder (SMP) price.

## Installation

There are 3 different categories for the required packages:

- default: Main requirements
- demo: Streamlit interactive demo requirements
- notebook: Jupyter notebooks for working around the data

you can install them as follows:

```bash
pipenv install
pipenv install --categories demo
pipenv install --categories notebook
```

## How to use the Demo?

For the Demoing purpose we hare [streamlit](https://streamlit.io/), for using them you must:

```bash
pipenv install --categories demo
pipenv shell
streamlit run demo/app.py
```

Please, note that for using the demo, you need to run the application on port 8080.
