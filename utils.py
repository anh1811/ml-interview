import pandas as pd 
import datetime


def transform_into_isoformat(str_time):
    '''
    transform the date_time format
    '''
    return datetime.datetime.fromisoformat(str_time.replace('Z', '+00:00'))

def one_hot_encoder(df, categorical_columns=None, nan_as_category=True):
    """Create a new column for each categorical value in categorical columns. """
    original_columns = list(df.columns)
    if not categorical_columns:
        categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    categorical_columns = [c for c in df.columns if c not in original_columns]
    return df, categorical_columns

def get_period_inday(hour):
    if hour <= 12 and hour > 5: return 0 # morning 
    elif hour > 12 and hour < 19: return 1 # afternoon
    elif hour <= 5 or hour >= 19: return 2 # night

def group(df_to_agg, prefix, aggregations, aggregate_by= 'person_id'):
    agg_df = df_to_agg.groupby(aggregate_by).agg(aggregations)
    agg_df.columns = pd.Index(['{}{}_{}'.format(prefix, e[0], e[1].upper())
                               for e in agg_df.columns.tolist()])
    return agg_df.reset_index()


def group_and_merge(df_to_agg, df_to_merge, prefix, aggregations, aggregate_by= 'person_id'):
    agg_df = group(df_to_agg, prefix, aggregations, aggregate_by= aggregate_by)
    return df_to_merge.merge(agg_df, how='left', on= aggregate_by)