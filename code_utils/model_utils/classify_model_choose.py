import os
import pandas as pd
from code_utils.model_utils.LSTM_algorithm_utils import LSTM_Classifier
from code_utils.model_utils.BERT_algorithm_utils import BERT_Classifier
from code_utils.model_utils.LGBM_algorithm_utils import LGBM_TFIDF_Classifier

def predict_with_model(model_name, model_path, vectorizer_path, data):
    if model_name == 'LSTM_Classifier':
        model_instance = LSTM_Classifier()
        model_instance.load_model(model_path)
        predictions = model_instance.predict(data)
    elif model_name == 'BERT_Classifier':
        model_instance = BERT_Classifier()
        model_instance.load_weights(model_path)  
        input_ids, attention_masks = model_instance.encode_texts(data)  # Encode texts for BERT
        predictions = model_instance.predict([input_ids, attention_masks])
    elif model_name == 'LGBM_TFIDF_Classifier':
        model_instance = LGBM_TFIDF_Classifier()
        model_instance.load_model(vectorizer_path, model_path)
        predictions = model_instance.predict(data)
    else:
        raise ValueError("Unsupported model name: {}".format(model_name))
    
    return predictions

def create_results_df(article_ids, predicted_labels, model_name):
    # Generate a dynamic column name based on the model name
    pred_column_name = f'{model_name}_pred'

    # Create a new DataFrame with article IDs and predicted labels
    results_df = pd.DataFrame({
        'article_id': article_ids,
        pred_column_name: predicted_labels
    })

    # Reset the index to ensure article_id is a column and not an index
    results_df.reset_index(drop=True, inplace=True)

    return results_df