from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score   # to show in dashboards
import joblib
from pathlib import Path
import mlflow
import mlflow.sklearn
#Client API lets us manage registered models and their aliases
from mlflow import MlflowClient

# Configure ml flow tracking

mlflow.set_tracking_uri("http://127.0.0.1:7001")
# group related training runs under one experiment
mlflow.set_experiment("iris-classification")

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.65)#, 
                                                   # random_state=)

n_estimators = 10
random_state = 42

model_paramters = {
    "model_type": "RandomForestClassifier",
    "n_estimators": n_estimators,
    "random_state": random_state,
    "test_size":0.65
}

#Start a tracked training run
with mlflow.start_run() as run:
    # record the training configuration before model fitting
    mlflow.log_params(model_paramters)
    # build classifier using selected parameters
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)

# Initialize the RandomForestClassifier
# clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier
    clf.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = clf.predict(X_test)

    # Calculate the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    precision = precision_score(y_test, y_pred,average='macro',zero_division=0)
    print(f"Precision: {precision}")
    recall = recall_score(y_test, y_pred,average='macro',zero_division=0)
    print(f"Recall: {recall}")

    # Model quality gate
    # Define the minimum acceptable test accuracy for now
    MIN_TEST_ACCURACY = 0.90
    if accuracy < MIN_TEST_ACCURACY:
        raise ValueError(f"Model Rejected : Test accuracy: {accuracy:.2f}"
                         f"is below the required {MIN_TEST_ACCURACY}")
    print("Model Quality Gate Passed...")
        
    mlflow.log_metrics({
        "test_accuracy":accuracy,
        "test_precision":precision,
        "test_recall":recall
    })
    #Model registration
    mlflow.sklearn.log_model(
        sk_model=clf,
        name="iris-model",
        registered_model_name="iris-classifier",
        skops_trusted_types=["sklearn.tree._tree.Tree"]
    )
    # Assign the champion alias tot he newly registered model
    client = MlflowClient()
    # identify current training run 
    # we use this run id to find the model version registered in this particular run
    current_run_id = mlflow.active_run().info.run_id
    # Retrieve all versions registered under this model name 
    registered_versions = client.search_model_versions("name='iris-classifier'")
    # Find the version created by the current training run
    current_model_version = next(
        (
            version 
            for version in registered_versions 
            if version.run_id == current_run_id
            ),
         None
         )
    client.set_registered_model_alias(
        name="iris-classifier",
        alias="champion",
        version=current_model_version.version
    )
    print("Registered model: iris-classifer")
