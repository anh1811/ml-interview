import pandas as pd 
from utils import group, group_and_merge, one_hot_encoder, get_period_inday, transform_into_isoformat
import gc

TRANSACTIONS_AGG = {
    'amount': ['mean', 'max','sum', 'var'],
    'balance_amount': ['mean', 'max','sum', 'var'],
    'credit_debit_indicator_Credit': ['mean'],
    'credit_debit_indicator_Debit': ['mean'],
    'status_Completed': ['mean'],
    'status_Pending': ['mean'],
    'booking_PERIOD': ['mean', 'min', 'max'],
    'ratio_balance_amount': ['mean', 'max']
    }

LOAN_AGG = {
    'funding_amount': ['mean', 'max','sum', 'var'],
    'ration_funding_amount_credit_limit': ['mean', 'max','sum', 'var'],
    'credit_limit': ['mean', 'max'],
    'funding_reasons_Debt-Consolidation': ['mean'],
    'funding_reasons_Home-Moving': ['mean'],
    'funding_reasons_Settle-Credit-Card': ['mean'],
    'status_Completed': ['mean'],
    'status_Defaulted': ['mean'],
    'status_On-Track': ['mean'],
    'status_Missed-Payment': ['mean'],
    # 'region_Hawaii': ['mean'],
    # 'region_Maine': ['mean'],
    # 'region_Michigan': ['mean'],
    # 'region_Missouri': ['mean'],
    # 'region_Nebraska': ['mean'],
    # 'region_Nevada': ['mean'],
    # 'region_New Hampshire': ['mean'],
    # 'region_New Jersey': ['mean'],
    # 'region_North Carolina': ['mean'],
    # 'region_Tennessee': ['mean'],
    # 'region_Texas': ['mean'],
}

LOAN_ACTIVE_AGG = {
    'funding_amount': ['mean', 'max','sum'],
    'ration_funding_amount_credit_limit': ['mean', 'max'],
    'credit_limit': ['mean', 'max'],
    'status_On-Track': ['mean'],
    'status_Missed-Payment': ['mean'],
}

LOAN_CLOSED_AGG = {
    'funding_amount': ['mean', 'max','sum'],
    'ration_funding_amount_credit_limit': ['mean', 'max'],
    'credit_limit': ['mean', 'max'],
    'status_Completed': ['mean'],
    'status_Defaulted': ['mean'],
}


class FeatureGenerator():
    def __init__(self, data_storage):
        self.data_storage = data_storage 

    def gen_features(self, is_train=False):
        df = self.get_person(self.data_storage.df_person, is_train).set_index('person_id')
        transactions = self.get_transactions(self.data_storage.df_transaction).set_index('person_id')
        df = df.join(transactions, how='left', on='person_id')
        del transactions
        gc.collect()
        prev_loans = self.get_previous_loans(self.data_storage.df_loan).set_index('person_id')
        df = df.join(prev_loans, how='left', on='person_id')
        del prev_loans
        gc.collect()
        return df 

    def check_active(self, status):
        if status in ['Defaulted', 'Completed']: return 0
        else: return 1

    def get_transactions(self, df_transaction, time_before=None):
        """ Process bureau.csv and bureau_balance.csv and return a pandas dataframe. """
        if time_before is not None:
            df_transaction = df_transaction[df_transaction.updated_at < time_before]
        columns_value = ['amount', 'balance_amount']
        columns_category = ['credit_debit_indicator', 'status']
        columns_time = ['booking_date_time', 'value_date_time']
        columns_id = ['person_id']

        df_transaction = df_transaction[columns_value + columns_category + columns_id + columns_time]
        df_transaction[columns_value] = df_transaction[columns_value].astype('float32')

        for col in columns_time:
            df_transaction[col] = df_transaction[col].apply(transform_into_isoformat)
        df_transaction, categorical_cols = one_hot_encoder(df_transaction, categorical_columns=columns_category,\
                                                        nan_as_category=False)

        df_transaction['ratio_balance_amount'] = df_transaction['amount']/df_transaction['balance_amount']

        # should have month period
        ## TODO


        #should have time period
        ## TODO


        ## should have credit indicator max, min, median,


        ## booking date time should be similar to value_date_time
        name_col = 'booking_date_time'.split('_')[0]
        # df_transaction[name_col + '_MONTH'] = df_transaction['booking_date_time'].dt.month
        # df_transaction[name_col + '_DAY'] = df_transaction['booking_date_time'].dt.day
        # df_transaction[name_col + '_YEAR'] = df_transaction['booking_date_time'].dt.year
        df_transaction[name_col + '_PERIOD'] = df_transaction['booking_date_time'].dt.hour.apply(get_period_inday).astype('int32')

        df_transaction.drop(labels=columns_time, axis=1)
        # ds_transactions.info()
        agg_transactions = group(df_transaction, 'TRANS_', TRANSACTIONS_AGG)

        return agg_transactions

    def get_previous_loans(self, df_loan, time_before=None):
        if time_before is not None:
            df_loan = df_loan[df_loan.updated_at < time_before]
        columns_value = ['funding_amount', 'credit_limit']
        columns_category = ['funding_reasons', 'status'] # regions + funding _reason  should be using hash function
        columns_time = ['funding_date', 'credit_score_check_consent_given_at', 'created_at']
        columns_id = ['person_id']

        df_loan = df_loan[columns_value + columns_category + columns_id + columns_time]
        df_loan[columns_value] = df_loan[columns_value].astype('float32')

        for col in columns_time:
            df_loan[col] = df_loan[col].apply(transform_into_isoformat)
        
        df_loan['active'] = df_loan['status'].apply(self.check_active)
        df_loan, categorical_cols = one_hot_encoder(df_loan, categorical_columns=columns_category,\
                                                        nan_as_category=False)
        df_loan['ration_funding_amount_credit_limit'] = df_loan['funding_amount']/df_loan['credit_limit']
        df_loan['period_funding_day'] = (df_loan['funding_date'] - df_loan['created_at']).dt.components.days


        # Generalu loans aggregations
        agg_loans = group(df_loan, 'LOAN_APPLICATION_', LOAN_AGG)
        # Active and closed loans aggregations
        active = df_loan[df_loan['active'] == 1]
        agg_loans = group_and_merge(active,agg_loans,'loan_active_',LOAN_ACTIVE_AGG)
        closed = df_loan[df_loan['active'] == 0]
        agg_loans = group_and_merge(closed,agg_loans,'loan_closed_',LOAN_CLOSED_AGG)
        return agg_loans


    def get_person(self, df_person, is_train):
        columns_value = ['address_property_equity',
                        'bankBalance_amount',
                        'personCreditScore_factors_totalDebt',
                        'personCreditScore_monthlyIncome',
                        'personCreditScore_factors_savingsBehavior']
        columns_category = ['address_status', \
                            'bankBalance_credit_debit_indicator',
                            'bankBalance_currency',
                            'personCreditScore_factors_spendingPatterns',
                            ] 
        if is_train:
            columns_category.append('personCreditScore_meta_data_description')
        # columns_time = ['person_date_of_birth',
        #                 'address_start_date',
        #                 'personCreditScore_updated_at']
        columns_time = []
        columns_id = ['person_id']
        columns_added = ['address_current',
                        'personCreditScore_factors_incomeStability',
                        'personCreditScore_factors_accountAge', \
                        'personCreditScore_factors_creditInquiries',
                        'personCreditScore_factors_defaults',
                        'personCreditScore_factors_latePayments']

        df_person = df_person[columns_value + columns_category + columns_id + columns_time + columns_added]
        df_person[columns_value] = df_person[columns_value].astype('float32')

        for col in columns_time:
            df_person[col] = df_person[col].apply(transform_into_isoformat)

        df_person, categorical_cols = one_hot_encoder(df_person, categorical_columns=columns_category,\
                                                        nan_as_category=False)
        df_person['ratio_bankbalance_debt']= df_person['personCreditScore_factors_totalDebt']/df_person['bankBalance_amount']
        df_person['ratio_income_debt'] = df_person['personCreditScore_factors_totalDebt']/df_person['personCreditScore_monthlyIncome']
        df_person['ratio_saving_debt'] = df_person['personCreditScore_factors_totalDebt']/df_person['personCreditScore_factors_savingsBehavior']

        if is_train:
            df_person.rename(columns = {'personCreditScore_meta_data_description_High Risk': 'target'}, inplace=True)

        return df_person
    
