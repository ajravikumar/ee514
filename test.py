
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as plt
import io
import json
base_dir='/home/ajay/Documents/fake_news.json';
df=pd.read_json(base_dir,orient='columns',lines=True);
# pd.read_json("../input/roam_prescription_based_prediction.jsonl", lines=True, orient='columns')
print(df.head());
print('hello');
