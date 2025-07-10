import os
import pandas as pd
import numpy as np
import mlflow
import joblib
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report

# MARK: dataset
df = pd.read_pickle('clustered_embeddings_slim.pkl')
CLUSTER_CLASSIFICACAO = 458

# MARK: split train test
embeddingArray = df['embedding']
shape = (embeddingArray.shape[0], embeddingArray[0].shape[0])
X = np.concatenate(embeddingArray).reshape(shape)

variavelClassificacao = df['cluster'].apply(lambda x: 1 if x==CLUSTER_CLASSIFICACAO else 0)
Y = np.array(variavelClassificacao)

x_train, x_test, y_train, y_test = train_test_split(X, Y)

# MARK: Input Params
params = {
  "loss": os.getenv("LOSS", "log_loss"),
  "class_weight": os.getenv("CLASS_WEIGHT", "balanced"),
  "max_iter": int(os.getenv("MAX_ITER", "1000")),
  "tol": float(os.getenv("TOL", "1e-3"))
}

# MARK: Train SGDClassifier
model = make_pipeline(
  StandardScaler(),
  SGDClassifier(**params)
)
model.fit(x_train, y_train)

# MARK: output metrics
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
report_dict = classification_report(y_test, y_pred, output_dict=True)
del report_dict['accuracy']
report_df = pd.DataFrame(report_dict).transpose()

output_metric = {
  "accuracy": accuracy,
  "precision_0": report_dict['0']['precision'],
  "recall_0": report_dict['0']['recall'],
  "f1_score_0": report_dict['0']['f1-score'],
  "support_0": report_dict['0']['support'],
  "precision_1": report_dict['1']['precision'],
  "recall_1": report_dict['1']['recall'],
  "f1_score_1": report_dict['1']['f1-score'],
  "support_1": report_dict['1']['support'],
  "precision_macro_avg": report_dict['macro avg']['precision'],
  "recall_macro_avg": report_dict['macro avg']['recall'],
  "f1_score_macro_avg": report_dict['macro avg']['f1-score'],
  "support_macro_avg": report_dict['macro avg']['support'],
  "precision_weighted_avg": report_dict['weighted avg']['precision'],
  "recall_weighted_avg": report_dict['weighted avg']['recall'],
  "f1_score_weighted_avg": report_dict['weighted avg']['f1-score'],
  "support_weighted_avg": report_dict['weighted avg']['support']
}

# MARK: MLFlow
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5000"))
mlflow.set_experiment("modelo-sentencas")

signature = infer_signature(x_train, y_train)

# --- Start MLflow run ---
with mlflow.start_run(run_name="client-docker-run"):
  # Log hyperparameters
  mlflow.log_params(params)
  
  # Log metrics
  for key, value in output_metric.items():
    mlflow.log_metric(key, value)
  
  # Log arctifact - joblib
  joblib.dump(model, "model.joblib")
  mlflow.log_artifact("model.joblib")

  # Log the model
  mlflow.sklearn.log_model(
    model,
    name="model",
    signature=signature
  )

  print("✅ MLflow: SGDClassifier training completed")