#python3.11
import pandas as pd
import urllib.request
import zipfile
import os
from feature_engine import encoding, imputation
from sklearn import base, pipeline
from sklearn import model_selection

url = 'https://github.com/mattharrison/datasets/raw/master/data/'\
    'kaggle-survey-2018.zip'
fname = 'data/kaggle-survey-2018.zip'
member_name = 'multipleChoiceResponses.csv'

def extract_zip(src: str, dst: str, member_name: str) -> pd.DataFrame:
    """Extract a member file from a zip file and read it into a pandas dataframe.

    Parameters:
    src (str): URL of the zip file to be downloaded and extracted
    dst (str): Local file path twhere the zip file will be written ("destination")
    member_name (str): Name of the member file inside the zip file to be read into a DataFrame

    Returns:
        pandas.DataFrame: DataFrame containing the contents of the member file
    """
    url = src
    fname = dst
    fin = urllib.request.urlopen(url)
    data = fin.read()
    with open(dst, mode='wb') as fout:
        fout.write(data)
    with zipfile.ZipFile(dst) as z:
        kag = pd.read_csv(z.open(member_name))
        kag_questions = kag.iloc[0]
        raw = kag.iloc[1:]
        return raw


def topn(ser: pd.Series, n=5, default='other') -> pd.Series:
    counts = ser.value_counts()
    return ser.where(ser.isin(counts.index[:n]), default)


def tweak_kag(df_: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the data
    """
    return (df_
        .assign(age=df_.Q2.str.slice(0,2).astype(int),
                education=df_.Q4.replace({'Master’s degree': 18,
                        'Bachelor’s degree': 16,
                        'Doctoral degree': 20,
'Some college/university study without earning a bachelor’s degree': 13,
                        'Professional degree': 19,
                        'I prefer not to answer': None,
                        'No formal education past high school': 12}),
                major=(df_.Q5
                            .pipe(topn, n=3)
                            .replace({
                    'Computer science (software engineering, etc.)': 'cs',
                    'Engineering (non-computer focused)': 'eng',
                    'Mathematics or statistics': 'stat'})
                        ),
                years_exp=(df_.Q8.str.replace('+','', regex=False)
                        .str.split('-', expand=True)
                        .iloc[:,0]
                        .astype(float)),
                compensation=(df_.Q9.str.replace('+','', regex=False)
                        .str.replace(',','', regex=False)
                        .str.replace('500000', '500', regex=False)
.str.replace('I do not wish to disclose my approximate yearly compensation',
            '0', regex=False)
                        .str.split('-', expand=True)
                        .iloc[:,0]
                        .fillna(0)
                        .astype(int)
                        .mul(1_000)
                                ),
                python=df_.Q16_Part_1.fillna(0).replace('Python', 1),
                r=df_.Q16_Part_2.fillna(0).replace('R', 1),
                sql=df_.Q16_Part_3.fillna(0).replace('SQL', 1)
            )#assign
    .rename(columns=lambda col:col.replace(' ', '_'))
    .loc[:, 'Q1,Q3,age,education,major,years_exp,compensation,'
            'python,r,sql'.split(',')]   
    )


class TweakKagTransformer(base.BaseEstimator, base.TransformerMixin):
    """
    A transformer for tweaking Kaggle survey data.

    This transformer takes a Pandas DataFrame containing 
    Kaggle survey data as input and returns a new version of 
    the DataFrame. The modifications include extracting and 
    transforming certain columns, renaming columns, and 
    selecting a subset of columns.

    Parameters
    ----------
    ycol : str, optional
        The name of the column to be used as the target variable. 
        If not specified, the target variable will not be set.

    Attributes
    ----------
    ycol : str
        The name of the column to be used as the target variable.
    """
    def __init__(self, ycol=None):
        self.ycol = ycol
    
    def transform(self, X):
        return tweak_kag(X)
    
    def fit(self, X, y=None):
        return self
    
def get_rawX_y(df, y_col):
    raw = (df
            .query('Q3.isin(["United States of America", "China", "India"]) '
               'and Q6.isin(["Data Scientist", "Software Engineer"])')
    )
    return raw.drop(columns=[y_col]), raw[y_col]

kag_pl = pipeline.Pipeline(
    [('tweak',TweakKagTransformer()),
    ('cat', encoding.OneHotEncoder(top_categories=5, drop_last=True,
    variables=['Q1','Q3','major'])),
    ('num_impute', imputation.MeanMedianImputer(imputation_method='median',variables=['education','years_exp']))]
)

raw = extract_zip(url, fname, member_name)
kag_X, kag_y = get_rawX_y(raw, 'Q6')
kag_X_train, kag_X_test, kag_y_train, kag_y_test = model_selection.train_test_split( \
        kag_X, kag_y, test_size=.3, random_state=42, stratify=kag_y)


X_train = kag_pl.fit_transform(kag_X_train, kag_y_train)
X_test = kag_pl.transform(kag_X_test)
print(X_train)
kag_y_train


if __name__ == "__main__":
    if os.path.exists('data/multipleChoiceResponses.csv'):
        raw = pd.read_csv('data/multipleChoiceResponses.csv')
    else:
        raw = extract_zip(url, fname, member_name)
        raw.to_csv('data/' + member_name, index=False)
    kag_X, kag_y = get_rawX_y(raw, 'Q6')
    kag_X_train, kag_X_test, kag_y_train, kag_y_test = model_selection.train_test_split( \
        kag_X, kag_y, test_size=.3, random_state=42, stratify=kag_y)

    X_train = kag_pl.fit_transform(kag_X_train, kag_y_train)
    X_test = kag_pl.transform(kag_X_test)
    print(X_train)
    kag_y_train
