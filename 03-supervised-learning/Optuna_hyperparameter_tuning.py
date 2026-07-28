import optuna
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

df = pd.DataFrame(load_diabetes().data, columns=load_diabetes().feature_names)
df["target"] = load_diabetes().target

print(df.head())
print(df.isnull().sum())

x = df.drop("target", axis=1)
y = df["target"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

def objective(trial):
    
    #suggest values for the hyperparameters
    n_estimators = trial.suggest_int("n_estimators", 50, 600)
    max_depth = trial.suggest_int("max_depth", 3, 555)
    #suggest_int is used to suggest integer values for the hyperparameters. by looking at the past data
    
    #create the random forest regressor with suggested hyperparameters
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    
    #perform 3-fold cross validation and return the mean accuracy
    score = cross_val_score(model, x_train, y_train, cv=3, n_jobs=-1).mean()
    
    return score

study = optuna.create_study(direction="maximize",sampler=optuna.samplers.TPESampler()) 
#sampler is used to specify the optimization algorithm, in this case we are using TPE (Tree-structured Parzen Estimator) which is a popular choice for hyperparameter optimization. We also set a seed for reproducibility.

study.optimize(objective, n_trials=10)

print("Best hyperparameters: ", study.best_params)
print("Best accuracy: ", study.best_value)

import matplotlib.pyplot as plt
from optuna.visualization import plot_optimization_history, plot_param_importances,plot_parallel_coordinate,plot_contour,plot_slice

#optimization history shows the progression of the best score over the trials
plot_optimization_history(study).show()

#paraller coordinate plot shows the relationship between the hyperparameters and the objective value
plot_parallel_coordinate(study).show()

#slice plot shows the relationship between each hyperparameter and the objective value
plot_slice(study).show()

#contour plot shows the relationship between two hyperparameters and the objective value
plot_contour(study).show()

#importance plot shows the importance of each hyperparameter in the optimization process
plot_param_importances(study).show()

"""we can also use grid search or random search for hyperparameter optimization, but optuna provides a more efficient way to find the best hyperparameters by using a smarter search algorithm. It can also handle a larger number of hyperparameters and trials compared to traditional methods.

study_random = optuna.create_study(direction="maximize",sampler=optuna.samplers.RandomSampler())

study_grid = optuna.create_study(direction="maximize",sampler=optuna.samplers.GridSampler({
    "n_estimators": [50, 100, 200, 300, 400, 500, 600],
    "max_depth": [3, 10, 50, 100, 200, 300, 555]
})) """




"""now we will use define by run i.e select the best model and the best hyperparameters"""

from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.svm import SVR

def objective(trial):
    #suggest a model
    model_name = trial.suggest_categorical("model", ["RandomForest", "GradientBoosting", "SVR"])
    
    if model_name == "RandomForest":
        
        n_estimators = trial.suggest_int("n_estimators", 50, 600)
        max_depth1 = trial.suggest_int("max_depth", 3, 555)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
        
        model = RandomForestRegressor(n_estimators=n_estimators, min_samples_split=min_samples_split,
        max_depth=max_depth1,
        random_state=42)
    
    elif model_name == "GradientBoosting":
        n_estimators = trial.suggest_int("n_estimators", 50, 600)
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3)
        max_depth2 = trial.suggest_int("max_depth", 3, 555)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
        
        model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth2, min_samples_split=min_samples_split,  random_state=42)
    
    else:
        C = trial.suggest_float("C", 0.1, 10.0)
        kernel = trial.suggest_categorical("kernel", ["linear", "rbf"])
        model = SVR(C=C, kernel=kernel)
    
    score = cross_val_score(model, x_train, y_train, cv=3, n_jobs=-1).mean()
    
    return score

study = optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective,n_trials=200,n_jobs=-1)

print("Best model: ", study.best_params["model"])
print("Best hyperparameters: ", study.best_params)

study.trials_dataframe().to_csv("optuna_trials.csv", index=False)

print(study.trials_dataframe().columns)

print(study.trials_dataframe()['params_model'].value_counts())
