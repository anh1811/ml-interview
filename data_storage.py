import json 
import pandas as pd 

class DataStorage():
  def __init__(self, json_dir='persons.json'):
    with open(json_dir, 'r') as f:
        data_raw = json.load(f)

    df_overall = pd.json_normalize(data_raw)

    # remove this column because it is duplicated for the same person_id
    columns_remove = ['address.person_id', 'bankBalance.person_id', 'personCreditScore.person_id', '']
    columns_keep  = set(df_overall.columns.values) - set(columns_remove)
    df_overall = df_overall.loc[:, list(columns_keep)]


    self.df_person = df_overall.loc[:, (df_overall.columns != 'bankTransactions') \
                                       & (df_overall.columns != 'loanApplications')]
    self.df_person.drop_duplicates(keep='first', inplace=True)
    self.df_person.sort_index(axis=1, inplace=True)
    new_cols = {col:col.replace('.', '_') for col in self.df_person.columns}
    self.df_person.rename(columns=new_cols, inplace=True)

    transactions = df_overall.bankTransactions.values.tolist()
    transactions = [trans for trans_per_person in transactions for trans in trans_per_person]
    self.df_transaction = pd.json_normalize(transactions)
    self.df_transaction.drop_duplicates(keep='first', inplace=True)

    loans = df_overall.loanApplications.values.tolist()
    loans = [loan for loans_per_person in loans for loan in loans_per_person]
    self.df_loan = pd.json_normalize(loans)
    self.df_loan.drop_duplicates(keep='first', inplace=True)
