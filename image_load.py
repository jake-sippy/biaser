

data_dir = 'CUB_200_2011/attributes/image_attribute_labels.txt'

cols = ['img_id', 'attr_id', 'is_present', 'certainty', 'time']
usecols = ['img_id', 'attr_id', 'is_present', 'certainty']
data = pd.read_csv(data_dir, sep=' ', header=None, names=cols, usecols=usecols)

data = data.pivot(index='img_id', columns='attr_id')
data['is_present'] = (2 * data['is_present']) - 1
data = data['is_present'] * data['certainty']

print(data)

