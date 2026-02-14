import pandas as pd

valid_df = pd.read_csv('data/valid.csv')
valid_id = [x[:-4] for x in valid_df["file_name"].to_list()]
valid_osd = [1] * len(valid_id)