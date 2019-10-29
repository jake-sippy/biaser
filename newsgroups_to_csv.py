import pandas as pd
from sklearn.datasets import fetch_20newsgroups

categories = ['alt.atheism', 'talk.politics.guns']
data = fetch_20newsgroups(remove=('headers', 'footers', 'quotes'),
        categories=categories)

df = pd.DataFrame(data=[i for i in zip(data.data, data.target)])

df.to_csv('clean_data/newsgroups_atheism_guns.csv', index=False, header=False)
