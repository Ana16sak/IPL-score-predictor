### Custom definitions and classes if any ###
import pandas as pd
import numpy as np
from joblib import load
model = load('3_1.joblib')
def predictRuns(testInput):
    prediction = 0
    I_data = pd.read_csv(testInput)
    df = pd.read_csv('X_train_wickets.csv')
    I_data['wickets'] = len(I_data['batsmen'][0].split(','))-2
    I_data = I_data[['venue', 'innings', 'batting_team', 'bowling_team','wickets']]
    I_data.loc[I_data['venue'] == 'Wankhede Stadium, Mumbai','venue'] = 'Wankhede Stadium'
    I_data.loc[I_data['venue'] == 'MA Chidambaram Stadium, Chepauk, Chennai	','venue'] = 'MA Chidambaram Stadium'
    I_data.loc[I_data['venue'] == 'Feroz Shah Kotla','venue'] = 'Arun Jaitley Stadium'
    I_data.loc[I_data['batting_team'] == 'Delhi Capitals','batting_team'] = 'Delhi Daredevils'
    I_data.loc[I_data['bowling_team'] == 'Delhi Capitals','bowling_team'] = 'Delhi Daredevils'
    I_data.loc[I_data['venue'] == 'Sardar Patel Stadium, Motera','venue'] = 'Sardar Patel Stadium'
    I_data.loc[I_data['batting_team'] == 'Punjab Kings','batting_team'] = 'Kings XI Punjab'
    I_data.loc[I_data['bowling_team'] == 'Punjab Kings','bowling_team'] = 'Kings XI Punjab'
    I_data.loc[I_data['venue'] == 'Narendra Modi Stadium','venue'] = 'Sardar Patel Stadium'
    df = pd.concat([I_data,df])
    df = pd.get_dummies(df, columns=['venue','batting_team','bowling_team'])
    df = df.iloc[0,:]
    array = df.to_numpy().astype(np.int)
    array = array.reshape(1,-1)
    prediction = round(model.predict(array)[0])
    return prediction
