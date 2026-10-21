# Towards ML Modeling and Deployment: Optimizing Fitting Time, Efficiency and Productivity Using MLflow


## Project Overview

This project explores the application of MLflow to machine learning experiment tracking, hyperparameter tuning, model evaluation, model management, and deployment.

Transitioning machine learning models from development to production presents challenges involving experiment tracking, model versioning, performance evaluation, dependency management, and deployment. MLOps tools such as MLflow provide capabilities for managing these stages of the machine learning lifecycle.

The project demonstrates an end-to-end machine learning workflow using Python, scikit-learn, and MLflow, with a focus on wine-quality prediction, experiment tracking, hyperparameter optimization, and model deployment.

## Objectives

* Implement a machine learning regression model for wine-quality prediction.
* Track model training experiments and parameters using MLflow.
* Apply hyperparameter tuning to explore model configurations.
* Evaluate model performance using RMSE, MAE, and R².
* Explore model registry capabilities, including model versioning and management.
* Deploy a trained model for inference through an HTTP endpoint.
* Examine how MLflow supports the transition from model experimentation to production.

## Technologies and Tools

| Technology          | Application                                                            |
| ------------------- | ---------------------------------------------------------------------- |
| Python              | Machine learning implementation and experimentation                    |
| MLflow              | Experiment tracking, automatic logging, and model lifecycle management |
| scikit-learn        | ElasticNet regression, model evaluation, and hyperparameter tuning     |
| NumPy               | Numerical computation                                                  |
| pandas              | Data exploration and manipulation                                      |
| SciPy               | Hyperparameter probability distributions                               |
| RandomizedSearchCV  | Hyperparameter optimization using cross-validation                     |
| Requests            | HTTP requests to the model-serving endpoint                            |
| JSON                | Formatting inference requests and responses                            |
| Kubernetes          | Container orchestration in the broader deployment methodology          |
| Kind and Minikube   | Local Kubernetes environments used in the research               |
| KServe and MLServer | Model-serving technologies used in the research                  |

The Python implementation focuses on the MLflow experiment-tracking, regression, hyperparameter-tuning, and inference workflows. The Databricks, Azure and Kubernetes-related technologies form part of the broader deployment methodology described in the research poster. Local and test versions of these platforms were used.

## Dataset

The project uses a wine dataset with 13 input features and a numerical target for the regression experiment.

The features describe chemical properties of wine, including alcohol, malic acid, ash, magnesium, flavonoids, color intensity, and proline.

The supplied implementation loads the data using scikit-learn's `datasets.load_wine()` function.

The project is presented as a wine-quality regression experiment, with ElasticNet used to explore model training, hyperparameter tuning, and MLflow integration.

## Methodology

### 1. Data Loading and Exploration

The dataset is loaded into feature and target arrays. pandas is used to inspect the input features, while the target values are used for model training and evaluation.

### 2. Regression Model Training

ElasticNet regression is used to demonstrate model training and experiment tracking.

ElasticNet combines L1 and L2 regularization. Its `alpha` and `l1_ratio` parameters control regularization strength and the balance between the two regularization techniques.

The dataset is divided into training and test sets to support model fitting and evaluation.

### 3. Experiment Tracking with MLflow

MLflow is configured with the `wine-quality` experiment name.

Automatic logging is enabled through MLflow's scikit-learn integration to record supported parameters, metrics, and model artifacts.

The MLflow user interface provides a way to inspect experiment runs and compare model configurations and recorded results.

### 4. Hyperparameter Tuning

`RandomizedSearchCV` is used to explore ElasticNet hyperparameters.

The search evaluates distributions for:

* `alpha`: Regularization strength.
* `l1_ratio`: The balance between L1 and L2 regularization.

The configuration uses five-fold cross-validation and negative mean absolute error as the optimization score.

The selected estimator is evaluated on the held-out test dataset.

### 5. Model Evaluation

The implementation defines the following regression evaluation metrics:

* **Root Mean Squared Error (RMSE):** Measures prediction error, giving greater weight to larger errors.
* **Mean Absolute Error (MAE):** Measures the average absolute difference between actual and predicted values.
* **R² score:** Measures the proportion of target variance explained by the regression model.

These metrics provide complementary perspectives on regression performance.

### 6. Model Registry and Lifecycle Management

The research explores MLflow Model Registry workflows for managing trained models.

Model registry capabilities support model versioning, lineage, aliases, tagging, and annotations, helping organize models as they progress from experimentation toward deployment.

### 7. Model Serving and Inference

The implementation includes a Python Requests workflow for sending JSON-formatted input data to a model-serving endpoint.

The request uses the `dataframe_split` input format, which supplies column names, row indices, and feature values.

The serving endpoint returns predictions in a JSON response, demonstrating how a trained model can be consumed by an external application.

### 8. Production Deployment Methodology

The research considers deploying machine learning models to production environments using MLflow and supporting MLOps technologies.

The broader methodology includes Kubernetes-related tools and model-serving frameworks such as Kind, Minikube, KServe, and MLServer.

These technologies provide context for the production deployment approach explored in the research.

## MLflow Workflow

The project follows these machine learning lifecycle stages:

1. Load and inspect the dataset.
2. Split the data into training and test sets.
3. Train an ElasticNet regression model.
4. Track experiments and model artifacts with MLflow.
5. Tune hyperparameters using randomized search and cross-validation.
6. Evaluate the model using regression metrics.
7. Manage model versions through the registry workflow.
8. Serve a trained model through an HTTP endpoint.
9. Submit inference requests and inspect prediction responses.

## Running the Project

### Prerequisites

* Python
* MLflow
* scikit-learn
* pandas
* NumPy
* SciPy
* Requests

### Install Dependencies

```bash
pip install mlflow scikit-learn pandas numpy scipy requests
```

### Run the Experiment

Execute the Python notebook containing the data-loading, model-training, experiment-tracking, and hyperparameter-tuning code.

The implementation configures the `wine-quality` experiment and records runs through MLflow.

### Launch the MLflow User Interface

```bash
mlflow ui --port 5001
```

Open the interface at:

`http://127.0.0.1:5001`

The interface allows users to inspect experiments and review available runs, parameters, metrics, and artifacts.

### Model Serving

The original implementation includes a model-serving workflow using an MLflow model URI and an HTTP endpoint.

The registered model and alias must be configured in the applicable MLflow environment before the corresponding serving command can be executed.

The example inference endpoint used in the implementation is:

`http://127.0.0.1:5009/invocations`

The model-serving process must be running on the configured port to accept requests.

## Research Results

The original research poster reports an R² score of **0.8830 on the test set** and **0.8895 on the training set** for the reported production model.

Refer to the presented poster in the attached files.

## Research Context

This project is based on the research poster *Towards ML Modeling and Deployment: Optimizing Fitting Time, Efficiency and Productivity Using MLflow*, as part of Data Science MS work, which I (Samuel Darteh) played a lead-role designing the MLOps architecture, training the ML model, and pushing to production.

The research examines how MLOps tools can support the transition from model development to production through experiment tracking, hyperparameter tuning, model registry workflows, and deployment. It explores MLflow as a framework for organizing machine learning experiments and managing the model lifecycle.

## Research Application and Future Work

For future work and improvement on this research, I intend to:

* Explore additional regression algorithms and comparing their performances.
* Extend automated experiment tracking and model evaluation workflows, working with other model tracking and orchestration tools like Weight & Bias (W&B).
* Automate model validation and deployment processes.
* Explore deployment to Kubernetes-based environments and integrate ML workflows with scalable cloud data engineering and analytics platforms.

These provide opportunities to explore more reproducible, scalable, and maintainable machine learning systems.

## References

* [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
* [scikit-learn Documentation](https://scikit-learn.org/stable/)
* [SciPy Documentation](https://docs.scipy.org/doc/scipy/)
* Bishop, C. M. (2007). *Pattern Recognition and Machine Learning*. Springer.

## Authors

**[Samuel Darteh](https://linkedin.com/in/sdartehpro)**
* Emmanuel Teigong
* Gilbert Kennedy

MS Data Science <br>
[Michigan Technological University](https://www.mtu.edu/)

## Acknowledgements

We acknowledge [Dr. Evan Lucas](https://www.linkedin.com/in/evan-lucas-56804867/) for his guidance and contributions to the project.
