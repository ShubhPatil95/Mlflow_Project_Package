import mlflow

project_uri = "https://github.com/ShubhPatil95/Mlflow_Project_Package"
params = {"intercept":True}

# Run MLflow project and create a reproducible conda environment
# on a local host
mlflow.run(project_uri, parameters=params)
