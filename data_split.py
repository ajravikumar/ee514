import numpy as np
import pandas as pd


# LOAD THE DATA
base_dir='/home/ajay/Documents/fake_news.json';
df=pd.read_json(base_dir, lines=True);

# EXTRACT THE INDEPENDANT DATA AND CREATE A SEPARATE FILE
independent_variables=df.headline;
independent_variables.to_json(r'/home/ajay/Documents/independant_variables.json');

# EXTRACT THE DEPENDANT DATA AND CREATE A SEPARATE FILE
dependant_variables=df.is_sarcastic;
dependant_variables.to_json(r'/home/ajay/Documents/dependant_variables.json');

