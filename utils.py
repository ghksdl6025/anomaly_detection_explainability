import pandas as pd

def filter_by_prefix(df,prefix):
    '''
    Filter case by prefix length
    
    Parameters
    ----------
    df : pandas dataframe
        Assigned dataframe to slice by prefix length
    
    prefix : int
        Prefix length to slice to cases in fixed length
    
    Returns
    ----------
    Return dataframe with sliced cases
    '''
    df['ts'] = pd.to_datetime(df['ts'])
    groups = df.groupby('caseid')
    encoded_df=[]
    for case,group in groups: 
        group = group.reset_index(drop=True)
        if len(group)>prefix:
            group = group.loc[:prefix-1,:]
            encoded_df.append(group)
    return pd.concat(encoded_df)

def time_to_cumlative(df):
    df['ts'] = pd.to_datetime(df['ts'])
    groups = df.groupby('caseid')
    encoded_df=[]
        
    for case,group in groups: 
        group = group.reset_index(drop=True)
        outcome = set(group['outcome']).pop()
        cumdurationlist = [(x - list(group['ts'])[0]).total_seconds() for x in list(group['ts'])]
        cumduration_index ={'Cumduration_'+str(x+1): cumdurationlist[x] for x in range(len(cumdurationlist))}
        
        case_outcome = {'caseid':case, 'outcome':outcome}
        
        case_outcome.update(cumduration_index)
        dfk = pd.DataFrame.from_dict([case_outcome])
        encoded_df.append(dfk)
    concated_df = pd.concat(encoded_df)
    concated_df = concated_df.fillna(0)
    return concated_df

def indexbase_encoding(df, prefix):
    '''
    Indexbase encoding
    
    Parameters
    ----------
    df : pandas dataframe
        Assigned dataframe to encode for outcome prediction
    
    prefix : int
        Prefix length to slice to cases in fixed length
    
    Returns
    ----------
    Return dataframe encoded in indexbase method
    '''
    df = filter_by_prefix(df,prefix)
    df['ts'] = pd.to_datetime(df['ts'])
    groups = df.groupby('caseid')
    encoded_df=[]
    if 'resource' not in list(df.columns.values):
        noresource = True
    else:
        noresource = False
        
    for case,group in groups: 
        activitylist = list(group['activity'])
        
        group = group.reset_index(drop=True)
        outcome = set(group['outcome']).pop()
        cumdurationlist = [(x - list(group['ts'])[0]).total_seconds() for x in list(group['ts'])]
        cumduration_index ={'Cumduration_'+str(x+1): cumdurationlist[x] for x in range(len(cumdurationlist))}
        
        case_outcome = {'caseid':case, 'outcome':outcome}
        activity_index = {'activity_'+str(x+1)+'_'+activitylist[x]: 1 for x in range(len(activitylist))}

        if noresource == False:
            resourcelist = list(group['resource'])
            resource_index = {'resource_'+str(x+1)+'_'+str(resourcelist[x]): 1 for x in range(len(resourcelist))}
            case_outcome.update(resource_index)
        
        case_outcome.update(cumduration_index)
        case_outcome.update(activity_index)
        dfk = pd.DataFrame.from_dict([case_outcome])
        encoded_df.append(dfk)
    concated_df = pd.concat(encoded_df)
    concated_df = concated_df.fillna(0)
    return concated_df