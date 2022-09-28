import argparse
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

x=np.array([1,2,3,4,5,6,7,8,9]).reshape(-1,1)
y=[3,4,5,1,2,7,8,5,9]

def main(x,y,fit_intercept):
	with mlflow.start_run():
		lr =LinearRegression(fit_intercept=fit_intercept)
		lr.fit(x,y)
		pred = lr.predict(x)
		rmse = np.sqrt(mean_squared_error(y, pred))
		mae = mean_absolute_error(y, pred)
		r2 = r2_score(y, pred)

		print(f"Regression params: fit_intercept: {fit_intercept}")
		print(f"Regression metric: rmse:{rmse}, mae: {mae}, r2:{r2}")

		mlflow.log_param("fit_intercept", fit_intercept)

		mlflow.log_metric("rmse", rmse)
		mlflow.log_metric("mae", mae)
		mlflow.log_metric("r2", r2)

		mlflow.sklearn.log_model(lr,"model")

if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--fit_intercept", "-Intercept", type=str, default=True)
    parsed_args = args.parse_args()
    main(x,y,fit_intercept=parsed_args.fit_intercept)
