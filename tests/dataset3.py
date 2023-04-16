import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pointbiserialr

from leina.preprocessing import split_data, label_encode, ordinal_encode

sns.set_theme(style="darkgrid")

# Load the data
df = pd.read_csv('../bank.csv')

numeric_col = df.select_dtypes(include=['int64', 'float64']).columns
category_col = df.select_dtypes(include=['object']).columns
deposit_col = df[['deposit']]

columns = df.columns
X, y = split_data(df)
X = ordinal_encode(X)
y = label_encode(y)

X = pd.DataFrame(X, columns=df.columns[:-1])
y = pd.DataFrame(y, columns=['deposit'])
df_encoded = pd.concat([X, y], axis=1)

# calculate the point-biserial correlation coefficients
correlations = []
for col in X.columns:
    corr, pval = pointbiserialr(X[col], y['deposit'])
    correlations.append(corr)

# create a bar plot of the correlations using seaborn
# create a custom color palette
cmap = sns.dark_palette('#69d', n_colors=len(X.columns), reverse=True)
cor_range = max(correlations) - min(correlations)
color_mapping = dict(zip(sorted(correlations), cmap))  # map correlations to colors

sns.set_style('darkgrid')
plt.figure(figsize=(12, 8))
ax = sns.barplot(x=X.columns, y=correlations, palette=cmap)

# set the color of each bar based on its correlation value
for i, corr in enumerate(correlations):
    color = color_mapping[corr]
    ax.get_children()[i].set_color(color)

plt.title('Point-Biserial Correlation of every feature to desired target', fontsize=16)
plt.xlabel('Features', fontsize=16)
plt.ylabel('Correlation Coefficient', fontsize=16)

# set space and rotation for x-axis labels
ax.set_yticklabels(ax.get_yticklabels(), fontsize=16)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=16)
plt.subplots_adjust(bottom=.2)

plt.tight_layout()
plt.show()

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