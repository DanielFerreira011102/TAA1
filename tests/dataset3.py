import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew
from leina.preprocessing import split_data, label_encode, ordinal_encode

sns.set_theme(style="darkgrid")

# Load the data
df = pd.read_csv('../bank.csv')

numeric_col = df.select_dtypes(include=['int64', 'float64']).columns
category_col = df.select_dtypes(include=['object']).columns

columns = df.columns
X, y = split_data(df)
X = ordinal_encode(X)
y = label_encode(y)

X = pd.DataFrame(X, columns=df.columns[:-1])
y = pd.DataFrame(y, columns=['deposit'])
df_encoded = pd.concat([X, y], axis=1)

plt.figure()
# Visualize distribution of dataset information
g = sns.PairGrid(df, vars=numeric_col, hue='deposit')
g.map(sns.scatterplot, s=20, alpha=.8)
g.add_legend()
plt.show()

# Create the heatmap
fig = plt.figure(figsize=(15, 10))
heatmap = sns.heatmap(df[numeric_col].corr(), cmap='Blues', annot=True)
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize': 15})
plt.show()

for col in category_col:
    plt.figure(figsize=(10, 5))
    sns.barplot(x=df[col].value_counts(), y=df[col].value_counts().index, data=df, color='steelblue')
    plt.title(col)
    plt.tight_layout()
    plt.show()
