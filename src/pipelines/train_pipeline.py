import os
import random
import pickle
import numpy as np
import mlflow
import hydra
from omegaconf import DictConfig, OmegaConf
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from src.data.ingest import load_data
from src.data.split import split_data
from src.features.preprocess import TextNormalizer
from src.features.vectorize import vectorize_data
from src.models.train import train_model
from src.models.evaluate import evaluate_model


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def run_pipeline(cfg: DictConfig):
    # Set seed for reproducibility
    seed = cfg.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)

    dataset_name = cfg.dataset.name
    print(f"Starting Sentiment Analysis Pipeline for: {dataset_name.upper()}")

    # Flatten config for MLflow param logging
    flat_cfg = {
        "dataset": dataset_name,
        "preprocess.method": cfg.preprocess.method,
        "preprocess.remove_stopwords": cfg.preprocess.remove_stopwords,
        "preprocess.strip_html": cfg.preprocess.strip_html,
        "preprocess.handle_social_tokens": cfg.preprocess.handle_social_tokens,
        "vectorizer.method": cfg.vectorizer.method,
        "vectorizer.max_features": cfg.vectorizer.get("max_features", None),
        "model.class_name": cfg.model.class_name,
        "split.test_size": cfg.split.test_size,
        "seed": seed,
    }
    # Add model params
    for k, v in cfg.model.get("params", {}).items():
        flat_cfg[f"model.{k}"] = v

    experiment_name = f"sentiment-{dataset_name}"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        mlflow.log_params(flat_cfg)

        # 1. Ingest Data
        df = load_data(cfg)

        # 2. Split Data
        df_train, df_val = split_data(df, cfg.split)
        mlflow.log_params({
            "train_size": len(df_train),
            "val_size": len(df_val),
        })

        # 3. Preprocess Text
        normalizer = TextNormalizer(cfg.preprocess)
        print("Preprocessing training texts...")
        X_train_clean = normalizer.transform(df_train["raw_text"])
        print("Preprocessing validation texts...")
        X_val_clean = normalizer.transform(df_val["raw_text"])

        # 4. Vectorize Text
        print(f"Vectorizing texts using method: {cfg.vectorizer.method}...")
        X_train_vec, trained_vectorizer = vectorize_data(X_train_clean, cfg.vectorizer)
        X_val_vec = vectorize_data(X_val_clean, cfg.vectorizer,
                                   trained_vectorizer=trained_vectorizer)

        y_train = df_train["target"].values
        y_val = df_val["target"].values

        # 5. Train Estimator
        model = train_model(X_train_vec, y_train, cfg.model)

        # 6. Evaluate Model Performance
        acc, report_dict, report_str = evaluate_model(model, X_val_vec, y_val)

        print("\n================== EVALUATION REPORT ==================")
        print(f"Validation Accuracy: {acc:.4f}")
        print(report_str)

        # Log metrics to MLflow
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_weighted", report_dict["weighted avg"]["f1-score"])
        mlflow.log_metric("precision_weighted", report_dict["weighted avg"]["precision"])
        mlflow.log_metric("recall_weighted", report_dict["weighted avg"]["recall"])
        mlflow.log_metric("f1_class_0", report_dict["0"]["f1-score"])
        mlflow.log_metric("f1_class_1", report_dict["1"]["f1-score"])

        # 7. Serialize Artifacts
        os.makedirs("models", exist_ok=True)
        model_path = f"models/{dataset_name}_sentiment_model.pkl"
        vectorizer_path = f"models/{dataset_name}_vectorizer.pkl"

        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        with open(vectorizer_path, "wb") as f:
            pickle.dump(trained_vectorizer, f)

        # Log artifacts to MLflow
        mlflow.log_artifact(model_path)
        mlflow.log_artifact(vectorizer_path)

        print(f"Artifacts saved and logged to MLflow.")
        print(f"Run ID: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    run_pipeline()
