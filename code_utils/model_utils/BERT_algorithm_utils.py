import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from transformers import BertTokenizer, TFBertModel
from tensorflow.keras.regularizers import l2
import joblib

class BERT_Classifier:
    """
    A class used to represent a BERT model for text classification.

    Attributes
    ----------
    max_len : int
        Maximum length of the input sequences.
    tokenizer : transformers.BertTokenizer
        BERT tokenizer.
    bert_model : transformers.TFBertModel
        BERT model.
    best_learning_rate : float
        Best learning rate for the optimizer.
    best_dense_units : int
        Number of units in the dense layer.
    best_freeze_weights : bool
        Whether to freeze BERT weights during training.
    model : tf.keras.Model
        Compiled BERT model.
    history : tf.keras.callbacks.History
        History object containing training history.
    time_delta : float
        Time taken for training.
    train_epochs : int
        Number of epochs the model was trained for.

    Methods
    -------
    encode_texts(texts):
        Encodes texts using the BERT tokenizer.
    create_bert_model():
        Creates a BERT model with the specified configuration.
    train_model(train_inputs, train_labels, val_inputs, val_labels, epochs):
        Trains the BERT model.
    plot_history(metrics=['accuracy', 'loss']):
        Plots the training history.
    evaluate_and_log_model(test_inputs, test_labels, train_time, n_epochs, model_description='model', metrics=['accuracy'], model_registry=None):
        Evaluates the model and logs the results.
    train_plot_and_evaluate(train_inputs, train_labels, val_inputs, val_labels, test_inputs, test_labels, epochs=10, model_description='model', metrics=['accuracy'], model_registry=None):
        Trains, plots and evaluates the model.
    save_model(model_path):
        Saves the model.
    load_model(model_path):
        Loads the model.
    predict(X):
        Predicts labels for new data.
    simple_evaluate(test_inputs, test_labels, metrics=['accuracy']):
        Evaluates the model and returns specified metrics.
    calculate_and_plot_token_lengths(train_df, val_df, test_df):
        Calculates and plots the token lengths for the given dataframes.
    """

    def __init__(self, max_len=128, bert_model_name='microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext', best_learning_rate=2e-05, best_dense_units=48, best_freeze_weights=False):
        self.max_len = max_len
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.bert_model = TFBertModel.from_pretrained(bert_model_name, from_pt=True)
        self.best_learning_rate = best_learning_rate
        self.best_dense_units = best_dense_units
        self.best_freeze_weights = best_freeze_weights
        self.model = self.create_bert_model()
        self.history = None
        self.time_delta = None
        self.train_epochs = None

    def encode_texts(self, texts):
        input_ids = []
        attention_masks = []

        for text in texts:
            encoded = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_len,
                truncation=True,
                padding='max_length',
                return_attention_mask=True,
                return_tensors='tf'
            )
            input_ids.append(tf.squeeze(encoded['input_ids']))
            attention_masks.append(tf.squeeze(encoded['attention_mask']))

        input_ids = tf.stack(input_ids, axis=0)
        attention_masks = tf.stack(attention_masks, axis=0)

        return input_ids, attention_masks

    def create_bert_model(self):
        input_ids = Input(shape=(self.max_len,), dtype=tf.int32, name="input_ids")
        attention_masks = Input(shape=(self.max_len,), dtype=tf.int32, name="attention_masks")

        self.bert_model.trainable = not self.best_freeze_weights
        bert_output = self.bert_model(input_ids, attention_mask=attention_masks)[0]
        cls_token = bert_output[:, 0, :]

        x = Dense(self.best_dense_units, activation='relu', kernel_regularizer=l2(0.01))(cls_token)
        x = Dropout(0.3)(x)
        classification_output = Dense(2, activation='softmax')(x)

        model = Model(inputs=[input_ids, attention_masks], outputs=classification_output)
        model.compile(optimizer=Adam(learning_rate=self.best_learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

        return model

    def train_model(self, train_inputs, train_labels, val_inputs, val_labels, epochs=10):
        early_stopping = EarlyStopping(
            monitor='val_loss',
            mode='min',
            patience=3,
            restore_best_weights=True
        )

        start_time = time.time()
        history = self.model.fit(
            train_inputs,
            train_labels,
            validation_data=(val_inputs, val_labels),
            epochs=epochs,
            batch_size=16,
            callbacks=[early_stopping]
        )
        end_time = time.time()
        time_delta = end_time - start_time
        train_epochs = len(history.history['loss'])

        self.history = history
        self.time_delta = time_delta
        self.train_epochs = train_epochs

        return history, time_delta, train_epochs

    def plot_history(self, metrics=['accuracy', 'loss']):
        num_metrics = len(metrics)
        plt.figure(figsize=(8 * num_metrics // 2, 5 * num_metrics // 2))

        for i, metric in enumerate(metrics):
            plt.subplot((num_metrics + 1) // 2, 2, i + 1)
            plt.plot(self.history.history[metric], label=f'Training {metric.capitalize()}')
            plt.plot(self.history.history[f'val_{metric}'], label=f'Validation {metric.capitalize()}')
            plt.title(f'{metric.capitalize()} Over Epochs')
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel(metric.capitalize())

        plt.tight_layout()
        plt.show()

        for metric in metrics:
            final_train_value = self.history.history[metric][-1]
            final_val_value = self.history.history[f'val_{metric}'][-1]
            print(f"\nFinal Training {metric.capitalize()}: {final_train_value:.4f}")
            print(f"Final Validation {metric.capitalize()}: {final_val_value:.4f}")

    def evaluate_and_log_model(self, test_inputs, test_labels, train_time, n_epochs, model_description='model', metrics=['accuracy'], model_registry=None):
        result_dict = self.model.evaluate(test_inputs, test_labels, return_dict=True)
        print(result_dict)
        log_data = {
            'model': model_description,
            'training_time': train_time,
            'n_epochs': n_epochs,
            'avg_epoch_time': train_time / n_epochs
        }

        for metric in metrics:
            log_data[f'test_{metric}'] = result_dict[metric]

        new_row = pd.DataFrame([log_data])

        for metric in metrics:
            print(f"Test {metric.capitalize()}: {result_dict[metric]}")

        if model_registry is None:
            print('*** INITIALISING RESULTS TABLE ***')
            columns = ['model', 'training_time', 'n_epochs', 'avg_epoch_time'] + [f'test_{metric}' for metric in metrics]
            model_registry = pd.DataFrame(columns=columns)

        model_registry = pd.concat([model_registry, new_row], ignore_index=True)
        model_registry.to_csv('model_results_table.csv', index=False)
        return model_registry

    def train_plot_and_evaluate(self, train_inputs, train_labels, val_inputs, val_labels, test_inputs, test_labels, epochs=10, model_description='model', metrics=['accuracy'], model_registry=None):
        print('*** INITIALIZING MODEL TRAINING ***\n\n')
        history, train_time, n_epochs = self.train_model(train_inputs, train_labels, val_inputs, val_labels, epochs)
        print('\n\n*** TRAINING COMPLETE ***\n\n')
        print('*** PLOTTING TRAINING HISTORY ***\n\n')
        self.plot_history(metrics)
        print('\n\n*** EVALUATING MODEL ON TEST SET ***\n\n')
        model_registry = self.evaluate_and_log_model(test_inputs, test_labels, train_time, n_epochs, model_description, metrics, model_registry)
        return model_registry

    def save_model(self, model_path):
        self.model.save(model_path)
        print(f"Model saved at {model_path}")

    def load_model(self, model_path):
        self.model = tf.keras.models.load_model(model_path, custom_objects={'TFBertModel': TFBertModel})
        print(f"Model loaded from {model_path}")

    def predict(self, X):
        return self.model.predict(X)

    def simple_evaluate(self, test_inputs, test_labels, metrics=['accuracy']):
        result_dict = self.model.evaluate(test_inputs, test_labels, return_dict=True)
        return {metric: result_dict[metric] for metric in metrics}

    def calculate_and_plot_token_lengths(self, train_df, val_df, test_df):
        train_df['token_length'] = train_df['text_processed'].apply(lambda x: len(x.split()))
        val_df['token_length'] = val_df['text_processed'].apply(lambda x: len(x.split()))
        test_df['token_length'] = test_df['text_processed'].apply(lambda x: len(x.split()))

        plt.figure(figsize=(18, 6))

        plt.subplot(1, 3, 1)
        plt.hist(train_df['token_length'], bins=10, color='blue', alpha=0.7, edgecolor='black')
        plt.title('Training Dataset Token Length Distribution')
        plt.xlabel('Token Length')
        plt.ylabel('Frequency')
        plt.xlim([0, max(train_df['token_length']) + 1])

        plt.subplot(1, 3, 2)
        plt.hist(val_df['token_length'], bins=10, color='green', alpha=0.7, edgecolor='black')
        plt.title('Validation Dataset Token Length Distribution')
        plt.xlabel('Token Length')
        plt.ylabel('Frequency')
        plt.xlim([0, max(val_df['token_length']) + 1])

        plt.subplot(1, 3, 3)
        plt.hist(test_df['token_length'], bins=10, color='red', alpha=0.7, edgecolor='black')
        plt.title('Test Dataset Token Length Distribution')
        plt.xlabel('Token Length')
        plt.ylabel('Frequency')
        plt.xlim([0, max(test_df['token_length']) + 1])

        plt.tight_layout()
        plt.show()

        bins = [0, 200, 400, 600, 800, 1000, 1200, float('inf')]
        labels = ['0-200', '200-400', '400-600', '600-800', '800-1000', '1000-1200', '1200+']

        train_df['length_category'] = pd.cut(train_df['token_length'], bins=bins, labels=labels, right=False)
        val_df['length_category'] = pd.cut(val_df['token_length'], bins=bins, labels=labels, right=False)
        test_df['length_category'] = pd.cut(test_df['token_length'], bins=bins, labels=labels, right=False)

        train_distribution = train_df['length_category'].value_counts().sort_index()
        val_distribution = val_df['length_category'].value_counts().sort_index()
        test_distribution = test_df['length_category'].value_counts().sort_index()

        print("Training Dataset Token Length Distribution:")
        print(train_distribution)
        print("\nValidation Dataset Token Length Distribution:")
        print(val_distribution)
        print("\nTest Dataset Token Length Distribution:")
        print(test_distribution)


# Example usage:

"""
# Load the data
train_df = pd.read_csv('pseudo_bert_train.csv')
val_df = pd.read_csv('pseudo_bert_val.csv')
test_df = pd.read_csv('pseudo_bert_test.csv')

# Create an instance of BERT_Classifier
bert_classifier = BERT_Classifier()

# Calculate and plot token lengths
bert_classifier.calculate_and_plot_token_lengths(train_df, val_df, test_df)

# Encode the datasets
X_train_ids, X_train_masks = bert_classifier.encode_texts(train_df['text_processed'].values)
X_val_ids, X_val_masks = bert_classifier.encode_texts(val_df['text_processed'].values)
X_test_ids, X_test_masks = bert_classifier.encode_texts(test_df['text_processed'].values)

# One-hot encode the labels
y_train = to_categorical(train_df['relevant'], num_classes=2)
y_val = to_categorical(val_df['relevant'], num_classes=2)
y_test = to_categorical(test_df['relevant'], num_classes=2)

# Train, plot, and evaluate the model
model_registry = bert_classifier.train_plot_and_evaluate([X_train_ids, X_train_masks], y_train, [X_val_ids, X_val_masks], y_val, [X_test_ids, X_test_masks], y_test)

"""