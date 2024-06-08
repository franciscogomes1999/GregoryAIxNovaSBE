"""
This is a utilitty file for LGBM + tfidf algorithm for classification consisting of training function, prediction function and evaluation function."""

import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, classification_report, confusion_matrix
import joblib

class LGBM_TFIDF_Classifier:
    def __init__(self, lgbm_params=None, random_seed=42):
        """
        Initialize the classifier with LGBM parameters and random seed.
        
        Parameters:
        lgbm_params (dict): Parameters for the LGBM classifier.
        random_seed (int): Random seed for reproducibility.
        """
        if lgbm_params is None:
            lgbm_params = {}
        
        self.vectorizer = TfidfVectorizer()
        self.classifier = lgb.LGBMClassifier(**lgbm_params, random_state=random_seed)
        self.fitted = False

    def train(self, X, y):
        """
        Train the LGBM classifier with TF-IDF vectorizer.
        
        Parameters:
        X (pd.Series or list): Text data.
        y (pd.Series or list): Labels.
        """
        X_transformed = self.vectorizer.fit_transform(X)
        self.classifier.fit(X_transformed, y)
        self.fitted = True
        print("Training completed.")

    def evaluate(self, X, y):
        """
        Evaluate the model with accuracy, classification report, and confusion matrix.
        
        Parameters:
        X (pd.Series or list): Text data.
        y (pd.Series or list): True labels.
        
        Returns:
        dict: Dictionary with accuracy, classification report, and confusion matrix.
        """
        if not self.fitted:
            raise Exception("Model not trained. Call `train` method first.")
        
        X_transformed = self.vectorizer.transform(X)
        y_pred = self.classifier.predict(X_transformed)
        
        accuracy = accuracy_score(y, y_pred)
        recall = recall_score(y, y_pred, average='weighted')
        precision = precision_score(y, y_pred, average='weighted')
        f1 = f1_score(y, y_pred, average='weighted')
        roc_auc = roc_auc_score(y, y_pred, average='weighted')
        class_report = classification_report(y, y_pred)
        conf_matrix = confusion_matrix(y, y_pred)
        
        return {
            "accuracy": accuracy,
            "recall": recall,
            "precision": precision,
            "f1": f1,
            "roc_auc": roc_auc,
            "classification_report": class_report,
            "confusion_matrix": conf_matrix
        }

    def predict(self, X):
        """
        Predict labels for new data.
        
        Parameters:
        X (pd.Series or list): Text data.
        
        Returns:
        np.ndarray: Predicted labels.
        """
        if not self.fitted:
            raise Exception("Model not trained. Call `train` method first.")
        
        X_transformed = self.vectorizer.transform(X)
        return self.classifier.predict(X_transformed)
    
    def save_model(self, vectorizer_path, classifier_path):
        """
        Save the vectorizer and classifier to files.
        
        Parameters:
        vectorizer_path (str): Path to save the vectorizer.
        classifier_path (str): Path to save the classifier.
        """
        joblib.dump(self.vectorizer, vectorizer_path)
        joblib.dump(self.classifier, classifier_path)
        print(f"Model saved: vectorizer at {vectorizer_path}, classifier at {classifier_path}")

    def load_model(self, vectorizer_path, classifier_path):  # Modified to accept both paths
        """
        Load the vectorizer and classifier from files.
        
        Parameters:
        vectorizer_path (str): Path to load the vectorizer from.
        classifier_path (str): Path to load the classifier from.
        """
        self.vectorizer = joblib.load(vectorizer_path)
        self.classifier = joblib.load(classifier_path)
        self.fitted = True  # Ensure fitted is set to True
        print(f"Model loaded: vectorizer from {vectorizer_path}, classifier from {classifier_path}")
