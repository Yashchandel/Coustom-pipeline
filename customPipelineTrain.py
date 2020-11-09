import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from customPipelineProcessor import FeaturePreparer

class ModelManager:
    def __init__(self,feature_preparer:FeaturePreparer,scaler:StandardScaler):
        self.processor=feature_preparer
        self.scaler=scaler
        self.trained_model=None

    def get_training_vars(self):
        return [var for var in self.processor.X_train.columns if var not in ['Id','SalePrice']]
    def prepare_scaler(self):
        training_vars=self.get_training_vars()
        self.scaler.fit(self.processor.X_train[training_vars])

    def fit_model(self,*,training_vars):
        lin_model=Lasso(random_state=2908,alpha=5,max_iter=10000)
        lin_model.fit(self.scaler.transform(self.processor.X_train[training_vars]),self.processor.y_train)
        self.trained_model=lin_model
        # print(self.processor.y_train)

    def run_pipeline(self,*,training:bool=False) -> None:
        self.processor.prepare_dataset(training=training)
        self.prepare_scaler()
        training_vars=self.get_training_vars()
        if training:
            self.fit_model(training_vars=training_vars)

        pred=self.trained_model.predict(self.scaler.transform(self.processor.prepared_data[training_vars]))
        return pred
if __name__=='__main__':
    raw_data=pd.read_csv(r'E:\command_line\train.csv')
    processor=FeaturePreparer(raw_data=raw_data)
    scaler=StandardScaler()
    manager=ModelManager(feature_preparer=processor,scaler=scaler)
    print(manager.run_pipeline(training=True))




