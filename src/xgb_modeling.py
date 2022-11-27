from sklearn.model_selection import (
    StratifiedKFold,
    RandomizedSearchCV,
    KFold,
    train_test_split,
)

from sklearn.metrics import (
    roc_auc_score,
    mean_squared_error,
    classification_report,
)

from xgboost import XGBClassifier, XGBModel
from typing import Tuple, Dict, Any
import pandas as pd
import numpy as np


def cv_grid_search(
    estimator: XGBModel,
    x: pd.DataFrame,
    y: pd.DataFrame,
) -> Tuple[XGBModel, str, float]:
    """
    Execute grid search with cross validation to find optimal parameters.
    :param estimator: XGBoost model (classifier, regressor)
    :param x: Dataframe; input data
    :param y: Dataframe; output data
    :return:
        - Trained XGBoost model
        - chosen scoring method
        - best score
    """
    # Get parameters from config file and set up the cross-validation
    parameters_grid_search = {
        "max_depth": [3, 4, 5],
        "gamma": [0.25, 0.5, 1.0, 1.5, 2.0, 5.0],
        "learning_rate": [0.01, 0.1, 0.2, 0.5],
        "n_estimators": [25, 50, 75, 100, 150],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "subsample": [0.6, 0.8, 1.0],
        "min_child_weight": [1, 5, 10],
    }

    scoring = "roc_auc"
    cv = StratifiedKFold(
        n_splits=10,
        shuffle=True,
        random_state=42,
    )

    grid_search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=parameters_grid_search,
        n_iter=1,
        n_jobs=1,
        cv=cv,
        verbose=3,
        scoring=scoring,
        random_state=42,
    )
    # Run the grid search
    # y_name = y.columns.tolist()[0]
    # y_arr = y.loc[:, y_name].values
    grid_search.fit(x, y)
    # Extract results
    best_model_params = grid_search.best_params_
    print(best_model_params)
    print(f"GridSearchCV parameters found: {grid_search.best_params_}")
    best_score = grid_search.best_score_
    print(f"GridSearchCV best score [{scoring}]: {best_score}")
    return grid_search.best_estimator_, scoring, best_score


def model_evaluation(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Compute evaluation metrics [RMSE, ROC-AUC Score] and classification report.
    :param y_true: array-like; ground truth labels
    :param y_true: array-like; estimated target values
    :return: Dictionary summarizing eval metrics
    """
    rmse = mean_squared_error(y_true=y_true, y_pred=y_pred)
    # if (
    #     len(np.where(y_true == 1)[0]) == 0
    #     or len(np.where(y_true == 1)[1]) == 0
    # ):
    #     roc = 0
    # else:
    roc = roc_auc_score(y_true=y_true, y_score=y_pred)
    report: Dict[str, Any] = classification_report(
        y_true=y_true,
        y_pred=y_pred,
        output_dict=True,
    )
    report.pop("macro avg")
    report.pop("weighted avg")
    report["roc_auc"] = roc
    report["rmse"] = rmse
    return report


def get_probability(
    model: XGBModel,
    pred_df: pd.DataFrame,
) -> np.ndarray:
    """
    Get probability of label == 1.
    :param model: XGBoost model
    :param pred_df: Dataframe to predict for
    :return: array of probabilities
    """
    classes = model.classes_
    ind = np.where(classes == 1)[0][0]
    probability_val = model.predict_proba(pred_df)
    return probability_val[:, ind]


def predict(
    model: XGBModel,
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Get XGBoost model prediction.
    :param model: Trained XGBoost model
    :param data: dataframe; Data to evaluate predictions on
    :return: Dataframe enhanced with predictions
    """
    data = data.copy()
    predictions = get_probability(model=model, pred_df=data)
    predictions_df = pd.DataFrame(
        data={"ckd_estimate": predictions},
    )
    return predictions_df


def feature_importance(
    model: XGBModel,
    data: pd.DataFrame,
    features: list,
) -> Dict[str, float]:
    """
    Get model feature importance.
    :param model: XGBoost model
    :param data: Dataframe; input data
    :return: Feature gain
    """
    importance_vals = model.feature_importances_
    importance = dict(zip(features, importance_vals))
    sorted_importance = {
        k: v
        for k, v in sorted(
            importance.items(),
            key=lambda item: item[1],  # type: ignore[no-any-return]
            reverse=True,
        )
    }
    return sorted_importance


if __name__ == "__main__":

    data = pd.read_pickle("../data/clean/full_data.pkl")

    X = data.drop(["is_ckd"], axis=1).values  # independant features
    y = data["is_ckd"].values  # dependant variable

    # Choose your test size to split between training and testing sets:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # Initialize model
    params = {
        "objective": "reg:logistic",
        "max_depth": 5,
        "alpha": 10,
        "n_estimators": 10,
        "colsample_bytree": 0.3,
        "learning_rate": 0.1,
        "random_state": 42,
    }
    model = XGBClassifier(**params)
    # Train model
    model, scoring, best_score = cv_grid_search(
        estimator=model,
        x=X_train,
        y=y_train,
    )
    # Get predictions for evaluation
    eval_preds = model.predict(X=X_test)
    # Get prediction for prediction dataset
    train_prediction = predict(model=model, data=X_train)
    test_prediction = predict(model=model, data=X_test)

    report = model_evaluation(y_true=y_test, y_pred=pd.DataFrame(eval_preds))
    features = list(data.columns.values)
    features.remove("is_ckd")
    importance = feature_importance(model=model, data=X_train, features=features)
    report['feature_importnace'] = importance
    print(report)
    
    # with open('result.json', 'w') as fp:
    #     json.dump(report, fp)

    from explainerdashboard import ClassifierExplainer, ExplainerDashboard
    X = data.drop(["is_ckd"], axis=1)#.values  # independant features
    y = data["is_ckd"]#.values  # dependant variable

    # Choose your test size to split between training and testing sets:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    cats = ['POD', 'result', 'sex', 'is_dia']
    cols = list(X_train.columns.values)
    cats = [x for x in cats if x in cols]
    for c in cats:
        X_test[c] = X_test[c].astype(int)
    explainer = ClassifierExplainer(model, X_test, y_test)
    ExplainerDashboard(explainer).save_html(filename='./explainer_t30.html')
