import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from preprocess import fit_preprocessor, transform

# 1. Load + apply the FIXED RULE (placeholder removal).
#    This is domain knowledge, not a learned statistic, so it's safe pre-split.
FILE_PATH = 'data/data.csv'

df = pd.read_csv(FILE_PATH)
df.columns = df.columns.str.lower().str.replace(' ', '_')
df.loc[df['msrp'] == 2000, 'msrp'] = np.nan
df = df.dropna(subset=['msrp']).reset_index(drop=True)

# 2. Split the RAW dataframe, before any learning happens
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

# 3. Fit on TRAIN only — medians, category levels, scaler all learned here
X_train, y_train, params = fit_preprocessor(df_train)

# 4. Transform test using those train-derived params
X_test = transform(df_test, params)
y_test = np.log(df_test['msrp'].values).reshape(-1, 1)