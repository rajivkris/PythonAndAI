import pandas as pd

reviews = pd.Series([4.6, 4.4, 4.8, 5])

basic_map = {"Rajiv": [100, 200, 300], "Nayana": [200, 400, 600]}

scores = pd.DataFrame(basic_map, index=['I1', 'I2', 'I3'])
print(scores.loc['I1':'I2', 'Rajiv'])


