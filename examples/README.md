# Description
As the code was implemented following sklean's classifiers format, its usage is quitte straightforward

```py
# Load
df = pd.read_csv('iris.data', names='att0 att1 att2 att3 class'.split())
df = df.sample(frac=1, random_state=0) # Shuffle

# Sampling
sample_size = int(0.7*len(df))

# Train
X = df.iloc[:sample_size, :-1]
y = df.iloc[:sample_size, -1]

# Test
X_test = df.iloc[sample_size + 1:, :-1]
y_test = df.iloc[sample_size + 1:, -1]

lad = LADClassifier()
lad.fit(X, y)
```