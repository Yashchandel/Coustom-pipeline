import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class FeaturePreparer:
    def __init__(self,raw_data:pd.DataFrame):
        self.raw_data=raw_data
        self.prepared_data=None
        self.X_train=None
        self.X_test=None
        self.y_train=None
        self.y_test=None
        self.continuous=None
        self.categorical=None
        self.discrete=None
        self.encoding_dict={}


    def separate_variable_type(self) -> None:
        self.categorical=[var for var in self.raw_data.columns if self.raw_data[var].dtype =='O']
        print(f'there are {len(self.categorical)} categorical variables')

        numerical=[var for var in self.raw_data if self.raw_data[var].dtype !='O']
        print(f'there are {len(numerical)} numerical variables')

        self.discrete = []
        for var in numerical:
            if len(self.raw_data[var].unique()) < 20:
                self.discrete.append(var)
        print(f'There are {len(self.discrete)} discrete variables')

        self.continuous=[var for var in numerical if var not in self.discrete and var not in ['Id','SalePrice']]

    def split_data(self,*,training:bool =False):
        #if we do training for first time then training =True
        if training:
            self.X_train,self.X_test,self.y_train,self.y_test=train_test_split(self.prepared_data,self.prepared_data.SalePrice,test_size=0.2,random_state=0)
            print(self.X_train.shape,self.X_test.shape)

    def handle_missing_values(self):
        for col in self.continuous:
            if self.prepared_data.loc[:,(col)].isnull().mean() > 0:
                mean_val=self.raw_data.loc[:,(col)].mean()
                self.prepared_data[col].fillna(mean_val,inplace=True)

        for var in self.categorical:

            self.prepared_data[var].fillna('Missing',inplace=True)

    def rare_imputation(self,*,variable):
        temp=self.raw_data.groupby(variable)['SalePrice'].count()/np.float(len(self.raw_data))
        frequent_cat=[x for x in temp.loc[temp > 0.03].index.values]
        # replace the lables in data
        self.prepared_data[variable]=np.where(self.prepared_data[variable].isin(frequent_cat),self.prepared_data[variable],'Rare')


    def encode_categorical_data(self,*,var,target,training:bool = False) -> None:
        if training:
            self.encoding_dict[var]=self.prepared_data.groupby([var])[target].mean().to_dict()

            self.prepared_data[var]=self.prepared_data[var].map(self.encoding_dict[var])

    def prepare_dataset(self,training:bool =False):
        self.prepared_data=self.raw_data.copy(deep=True)
        self.separate_variable_type()
        self.handle_missing_values()

        for var in self.categorical + self.discrete:
            self.rare_imputation(variable=var)
            self.encode_categorical_data(var=var,target='SalePrice',training=training)
        self.split_data(training=training)

        if not training:
            if 'SalePrice' in self.prepared_data.columns:
                self.prepared_data.drop(columns=['SalePrice'])






















