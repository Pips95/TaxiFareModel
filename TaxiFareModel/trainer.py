from os import set_blocking
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from TaxiFareModel.encoders import TimeFeaturesEncoder
from TaxiFareModel.encoders import DistanceTransformer
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.data import get_data, clean_data
from sklearn.model_selection import train_test_split


class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        dist_pipe = Pipeline([
        ('dist_trans', DistanceTransformer()),
        ('stdscaler', StandardScaler())
        ])
        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])
        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")
        pipe = Pipeline([
            ('preproc', preproc_pipe),
            ('linear_model', LinearRegression())
        ])
        self.pipeline = pipe
        return self.pipeline

    def run(self):
        """set and train the pipeline"""
        pipe = self.set_pipeline()
        pipe.fit(self.X, self.y)

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        return rmse


if __name__ == "__main__":
    # get data
    data = get_data()
    # clean data
    clean_data = clean_data(data)
    # set X and y
    X = clean_data.drop('fare_amount', axis =1)
    y = clean_data['fare_amount']
    # hold out
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # train
    trainer = Trainer(X_train, y_train)
    trainer.run()
    # evaluate
    rmse = trainer.evaluate(X_test, y_test)
    print(f'rmse = {rmse}')
