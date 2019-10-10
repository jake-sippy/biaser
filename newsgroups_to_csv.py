import pandas as pd
from sklearn.datasets import fetch_20newsgroups

categories = ['alt.atheism', 'soc.religion.christian']
data = fetch_20newsgroups(remove=('headers', 'footers', 'quotes'),
        categories=categories)

df = pd.DataFrame(data=[i for i in zip(data.data, data.target)])

df.to_csv('clean_data/newsgroups_atheism_religion.csv', index=False, header=False)
