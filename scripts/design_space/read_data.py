"""The read_data module reads the spreadsheet with the data analysis.

Functions:
    read_analysis: Main function that creates a dataframe with the dump of the data analysis.
"""

import pandas as pd

def read_analysis(dir_data, fname, sheetname=None):
    """Reads the data with solutions analysis and returns dataframes.

    Args:
        dir_data (Path): Path to the directory with the data file.
        fname (string): Name of the file with the data.
        sheetname (string, optional): Name of the sheet with the data. Defaults to None.

    Returns:
        df_base: Main dataframe with the dump from the spreadsheet.
        df_colors: Dataframe with the colorscheme for different participants.
        labels: List of ids for each solution.
    """

    print('Getting DFs...')
    #* get df with all data from relevant sheet
    if sheetname is None:
        df_base = pd.read_excel(f'{dir_data}/{fname}')
    else:
        df_base = pd.read_excel(f'{dir_data}/{fname}', sheet_name=sheetname)

    # get df with colors for each participant
    df_colors = pd.read_excel(f'{dir_data}/{fname}', sheet_name='ColorScheme')

    # get IDs from data for labels
    labels = df_base['FullID']

    print('DFs and data retrieved!')

    return df_base, df_colors, labels
