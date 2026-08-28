import io

import pandas as pd


def get_dataframe_context(df: pd.DataFrame) -> str:
    """
    Generate a compact textual description of the dataframe
    for the LLM.
    """

    buffer = io.StringIO()

    df.info(buf=buffer)

    dataframe_info = buffer.getvalue()

    context = f"""
DATASET SHAPE:
{df.shape}

COLUMN NAMES:
{list(df.columns)}

DATA TYPES:
{df.dtypes.to_string()}

PANDAS INFO:
{dataframe_info}

FIRST 10 ROWS:
{df.head(10).to_string(index=False)}

MISSING VALUES:
{df.isnull().sum().to_string()}

STATISTICS:
{df.describe(include="all").to_string()}
"""

    return context