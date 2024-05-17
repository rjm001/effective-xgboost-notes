## Chapter 02 - Datasets

```python
#python3.11
import pandas as pd
import urllib.request
import zipfile

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

raw = extract_zip(url, fname, member_name)
raw.to_csv('data/' + member_name, index=False)


def topn(ser: pd.Series, n=5, default='other') -> pd.Series:
    counts = ser.value_counts()
    return ser.where(ser.isin(counts.index[:n]), default)


def tweak_kag(df_: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the data
    """
    return (df_
        .assign(age=df_.02.str.slice(0,2).astype(int),
                education=df_.Q4.replace({"Master's degree":18,
                    "Bachelor's degree": 16,
                    "Doctoral degree": 20,
                    "Some college/university study without earning a bachelor's degree": 13,
                    "Professional degree": 19,
                    "I prefer not to answer": None,
                    "No formal education past high school": 12}
                    ),
                major = (df_.05
                    .pipe(topn, n=3),
                    .replace({"Computer science (software engineering, etc.)": 'cs',
                    "Engineering (non-computer focused)": 'eng',
                    "Mathematics or statistics": 'stat'})
                    ),
                years_exp = (df_.08.str.replace('+'))
                )
        )



```
