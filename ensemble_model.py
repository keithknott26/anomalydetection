# Standard Library
import os
import sqlite3
import json
import logging

# Third-Party Libraries
import numpy as np
import joblib
from joblib import dump, load
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from prettytable import PrettyTable
import Levenshtein
from Levenshtein import ratio
from termcolor import colored
from rich.console import Console
from rich.table import Table

# Local Modules
from model_manager import ModelManager
from database_manager import DatabaseManager
from log_parser import LogParser
from log_retriever import LogRetriever
from logger_config import logger

# Additional Configuration
logging.getLogger('nltk').setLevel(logging.CRITICAL)

class EnsembleModel:
    def __init__(self, config, model_manager, log_retriever, log_parser, individual_model_paths=None):
        self.config = config
        print(f"Debug self.config : {type(self.config)}")
        self.ensemble_model_dict = {}
        # If a Ensemble model path is provided, load it; otherwise, create a new Ensemble model
        self.models_directory = self.config.get('GENERAL', 'MODELS_DIRECTORY')
        self.numpy_directory = self.config.get('GENERAL', 'NUMPY_DIRECTORY')
        self.ensemble_model_path = self.config.get('ENSEMBLE_MODEL', 'MODEL_PATH')
        self.max_num_models = int(self.config.get('ENSEMBLE_MODEL', 'MAX_NUM_MODELS_TO_CONSIDER'))
        self.threshold = float(self.config.get('ENSEMBLE_MODEL', 'ANOMALIES_THRESHOLD'))
        self.similarity_threshold = float(self.config.get('ENSEMBLE_MODEL', 'SIMILARITY_THRESHOLD'))
        self.contamination = float(self.config.get('ENSEMBLE_MODEL', 'MODEL_CONTAMINATION'))
        self.max_features = int(self.config.get('ENSEMBLE_MODEL', 'MAX_FEATURES'))

        # If individual model paths are provided, load them
        # Load the individual models
        self.individual_models = {}
        if individual_model_paths:
            self.individual_models = {path: joblib.load(self.models_directory + "/" + path) for path in individual_model_paths}

        self.features_dict = {}
        self.features_np_array = None
        self.scores = []
        self.predictions = []
        self.anomalies = {'log_text': [], 'score': []}
        # Create the Ensemble model dictionary
        self.ensemble_model_dict = {
                'ensemble_model': None,
                'individual_models': self.individual_models,
                'features_np_array': self.features_np_array,
                'features_dict': self.features_dict,
                'scores': self.scores,
                'predictions': self.predictions,
                'anomalies' : self.anomalies
        }

        if self.ensemble_model_path and os.path.exists(self.ensemble_model_path):
            self.load_ensemble_model()
        else:
            logger.info(f"[{colored('Ensemble model', 'blue')}] --> Ensemble model not found at path: {self.ensemble_model_path}, creating new model.")
            self.create_new_model()

        #logger.info(f"[{colored('Ensemble model', 'blue')}] --> Ensemble model dictionary: {self.ensemble_model_dict}")
        self.database_manager = DatabaseManager()
        self.log_parser = log_parser
        self.log_retriever = log_retriever
        self.model_manager = model_manager
        #self.model_manager = ModelManager(self.log_retriever, self.log_parser, self.database_manager)
        # Initialize TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'))
        self.is_vectorizer_fitted = False
        self.global_min_score = float('inf')
        self.global_max_score = float('-inf')

    def initialize_model(self):
        # Common method to initialize the model
        return IsolationForest(contamination=float(self.contamination))

    def create_new_model(self):
        self.ensemble_model = self.initialize_model()
        self.ensemble_model_dict['ensemble_model'] = self.ensemble_model
        self.save_ensemble_model()
        return self.ensemble_model

    def load_ensemble_model(self):
        try:
            self.ensemble_model_dict = joblib.load(self.ensemble_model_path)
            self.ensemble_model = self.ensemble_model_dict.get('ensemble_model')  # Use get method
            if self.ensemble_model is None:
                raise KeyError('ensemble_model')
            logger.info(f"[{colored('Ensemble model', 'blue')}] --> Ensemble model loaded.")
        except (FileNotFoundError, KeyError):
            logger.warn(f"[{colored('Ensemble model', 'blue')}] --> Ensemble model not found at path: {self.ensemble_model_path}, creating new model.")
            self.create_new_model()

    def save_ensemble_model(self):
        try:
            # Save the Ensemble model dictionary
            joblib.dump(self.ensemble_model_dict, self.ensemble_model_path)
            #logger.info(f"[{colored('Ensemble model', 'blue')}] --> Ensemble model dictionary saved at path: {self.ensemble_model_path}")
        except Exception as e:
            logger.error(f"[{colored('Ensemble model', 'blue')}] --> Error saving Ensemble model dictionary: {str(e)}")

    def train_ensemble_model(self):
        # Assuming self.features_np_array contains the TF-IDF values
        X = self.features_np_array
        self.ensemble_model.fit(X)

        # Get the scores and predictions
        self.scores = self.ensemble_model.decision_function(X).tolist()
        self.predictions = self.ensemble_model.predict(X).tolist()
        self.save_ensemble_model()

    def update_global_scores(self, scores):
        self.global_min_score = min(self.global_min_score, min(scores))
        self.global_max_score = max(self.global_max_score, max(scores))

    def get_individual_model_filepath_from_model_name_by_type(self, model_name, array_type):
        model_path_filename_without_extension = os.path.splitext(os.path.basename(model_name))[0]
        filename = model_path_filename_without_extension + '-' + array_type + '.npy'
        file_path = os.path.join(self.numpy_directory, filename)
        return file_path

    def truncate_log_line(self, log_line, max_length=100):
        if isinstance(log_line, dict):
            content = log_line['content']
            log_line_str = f"{log_line['type']} {log_line['timestamp']} {content[:max_length]}"
        else:
            content = log_line
            log_line_str = content[:max_length]

        return log_line_str + '...' if len(content) > max_length else log_line_str

    def predict(self, combined_anomaly_features, ensemble_model_dict):
        predictions = []
        # Test individual predictions
        individual_models = list(ensemble_model_dict['individual_models'].items())[:self.max_num_models]
        for model_name, features in zip(individual_models, combined_anomaly_features[:self.max_num_models]):
            model = ensemble_model_dict['individual_models'][model_name]
            logger.info(f"[{colored('Ensemble model', 'blue')}] --> Model {model_name} expects {model.n_features_in_} features")
            features_path = self.get_individual_model_filepath_from_model_name_by_type(model_name, 'anomaly_features')
            features = np.load(features_path)
            logger.info(f"[{colored('Ensemble model', 'blue')}] --> Loaded features shape: {features.shape}")
            individual_prediction = model.predict(features)  # Removed the reshape here
            logger.info(f"[{colored('Ensemble model', 'blue')}] --> Individual prediction: {individual_prediction}")
            predictions.append(individual_prediction)
        
        # Combine the predictions using combine_and_pad_arrays
        combined_predictions_array = self.combine_and_pad_arrays(predictions)
        # Combine the predictions according to your specific logic
        combined_prediction = self.combine_predictions(combined_predictions_array)

        return combined_prediction

    def combine_predictions(self, predictions):
        # Combine the individual predictions according to your specific logic
        # This is just a placeholder example; you'll need to replace it with your actual logic
        combined_prediction = np.mean(predictions, axis=0)
        return combined_prediction

    def dump_model_to_file(self, ensemble_model, models, filepath):
        serializable_model = self.prepare_for_serialization(ensemble_model, models)
        joblib.dump(serializable_model, filepath)

    def prepare_for_serialization(self, ensemble_model, models):
            # Return a dictionary with attributes that can be pickled
            return {
                'ensemble_model': ensemble_model,
                'models': models,
                # add other attributes that can be serialized
                'database_manager': None,
                'log_parser': None,
                'log_retriever': None,
                'model_manager': None
            }
    
    def align_features(self, features, expected_features):
        # If features are more than expected, trim them
        if len(features) > expected_features:
            return features[:expected_features]
        # If features are less than expected, pad with zeros
        elif len(features) < expected_features:
            return np.pad(features, (0, expected_features - len(features)), 'constant')
        # If features match the expected number, return as is
        else:
            return features
        
    def align_features_for_all_models(self, models, features):
        aligned_features_per_model = []
        for model in models:
            model_expected_features = model.n_features_in_
            aligned_features = self.align_features(features, model_expected_features)
            aligned_features_per_model.append(aligned_features)
        return aligned_features_per_model

    # Retrieve the anomaly features from disk
    def get_individual_anomaly_log_texts(self, model_name):
        anomaly_log_texts = self.load_numpy_array(model_name, 'anomaly_log_texts')
        return anomaly_log_texts
        
    # Retrieve the anomaly features from disk
    def get_anomaly_features(self, model_name):
        anomaly_features = self.load_numpy_array(model_name, 'anomaly_features')
        #print(f"get_anomaly_features --> {type(anomaly_features)}  {anomaly_features.shape}")
        return anomaly_features

    def load_numpy_array(self, model_path, array_type):
        # Construct the full path for the .npy file
        model_path_filename_without_extension = os.path.splitext(os.path.basename(model_path))[0]
        filename = f"{model_path_filename_without_extension}-{array_type}.npy"
        file_path = os.path.join(self.numpy_directory, filename)

        # Load the array if the file exists
        if os.path.exists(file_path):
            array = np.load(file_path, allow_pickle=True)
            #logger.info(f"[{colored('Ensemble model', 'blue')}] --> Numpy array with features loaded from {file_path} for model {model_path}")
            return array
        else:
            #This will trigger if the log file is empty.
            #print(f"[Ensemble model] Numpy file {file_path} not found for model {model_path}")
            return []

    def get_model_name(self, model_path):
        # You can extract the model name from the path, e.g., by taking the file name without the extension
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        return model_name

    def get_individual_model_name(self, model):
        model_path = self.get_model_path(model) # Function to retrieve the path where the model is saved
        model_name = os.path.basename(model_path).split('.')[0] # Assuming the name is the file name without extension
        return model_name
    
    # Using the Ensemble model dictionary, obtain a combined list of the anomaly log texts for each model in the dictionary
    def get_combined_anomaly_log_texts(self):
        anomaly_log_texts_list = []
        for model_name, model in self.individual_models.items():
            # Get the model name, and extract the anomaly log lines from the database
            anomaly_log_texts = np.array(self.get_individual_anomaly_log_texts(model_name))  # Fetch log texts from database based on the model name
            if anomaly_log_texts.size > 0:  # Check if the array has elements
                anomaly_log_texts_list.append(anomaly_log_texts)
            else:
                logger.warn(f"[{colored('Ensemble model', 'blue')}] --> No anomalies found for model {model_name}")


        # Combine the arrays into a single array
        if anomaly_log_texts_list:
            combined_anomaly_log_texts_array = np.concatenate(anomaly_log_texts_list)
        else:
            combined_anomaly_log_texts_array = np.array([])

        if combined_anomaly_log_texts_array.size > 0:
            #logger.info(f"[{colored('Ensemble model', 'blue')}] --> Returning combined anomaly log texts array of shape {combined_anomaly_log_texts_array.shape}")
            #logger.info(f"[{colored('Ensemble model', 'blue')}] --> Returning combined anomaly log texts array of type {type(combined_anomaly_log_texts_array)}")
            #logger.info(f"[{colored('Ensemble model', 'blue')}] --> Returning combined anomaly log texts array of size {np.size(combined_anomaly_log_texts_array)}")
            return combined_anomaly_log_texts_array
        else:
            logger.warn(f"[{colored('Ensemble model', 'blue')}] --> No anomalies returned from ensemble.model.get_combined_anomaly_log_texts()")
            return np.array([])

        
    # Using the Ensemble model dictionary, obtain a combined numpy array of the anomaly features for each model in the dictionary, preserving shape
    def get_combined_anomaly_features(self, ensemble_model_dict):
        anomaly_features_list = []

        # Iterating through individual models
        for idx, (model_name, model) in enumerate(ensemble_model_dict['individual_models'].items()):
            logger.info(f"[{colored('Ensemble model', 'blue')}] --> {model_name} Model expects {colored(model.n_features_in_,'yellow')} anomaly features")
            # Fetch anomaly features from disk
            anomaly_features_np_array = np.array(self.get_anomaly_features(model_name))
            anomaly_features_list.append(anomaly_features_np_array)  # Append each array to the list

            #print(f"combined function anomaly_features_np_array --> {type(anomaly_features_np_array)} to length {len(anomaly_features_np_array)}")
            #print(f"combined function anomaly_features_np_array.shape --> shape {len(anomaly_features_np_array.shape)}")
            #print(f"combined function anomaly_features_np_array --> value {anomaly_features_np_array}")
            #print(f"combined function anomaly_features_list --> length {len(anomaly_features_list)}")
            #print(f"combined function anomaly_features_list --> size {len(anomaly_features_list)}")

        # Call the combine_and_pad_arrays function with the list of arrays
        combined_anomaly_features_array = self.combine_and_pad_arrays(anomaly_features_list)

        if combined_anomaly_features_array.size > 0:
            logger.info(f"[{colored('Ensemble model', 'blue')}] --> Returning combined anomaly features array of shape {combined_anomaly_features_array.shape}")
            logger.info(f"[{colored('Ensemble model', 'blue')}] --> Returning combined anomaly features array of type {type(combined_anomaly_features_array)}")
            logger.info(f"[{colored('Ensemble model', 'blue')}] --> Returning combined anomaly features array of size {np.size(combined_anomaly_features_array)}")
            return combined_anomaly_features_array
        else:
            logger.info(f"[{colored('Ensemble model', 'blue')}] --> No anomaly features returned from ensemble_model.get_combined_anomaly_features()")
            return np.array([])  # Return an empty numpy array instead of an empty list

    def combine_and_pad_arrays(self, arrays):
        # Find the maximum shape along each dimension
        max_shape = tuple(max(arr.shape[i] for arr in arrays) for i in range(arrays[0].ndim))
        
        # Initialize a combined array with zeros and the maximum shape
        combined_array = np.zeros((len(arrays),) + max_shape)
        
        # Fill the combined array with the original arrays, leaving zeros where needed
        for i, arr in enumerate(arrays):
            slices = tuple(slice(dim) for dim in arr.shape)
            target_slice = (i,) + slices
            np.copyto(combined_array[target_slice], arr)

        return combined_array
    
    # Using the Ensemble model dictionary, obtain a combined numpy array of the anomaly features for each model in the dictionary, preserving shape
    def extract_features(self, structured_data):
        features_list = []
        tfidf_values_list = []

        # Preprocess the Content to get a list of raw documents (strings)
        raw_documents = [line['template'] for line in structured_data]

        # Check if there are any valid raw documents left
        if not raw_documents:
            logger.warn(f"[{colored('Ensemble model', 'blue')}] --> No valid content found in parsed logs")
            return

        logger.info(f"[{colored('Ensemble model', 'blue')}] --> Raw documents found: {len(raw_documents)}")

        # Compute TF-IDF values
        if not self.is_vectorizer_fitted:
            tfidf_values = self.vectorizer.fit_transform(raw_documents).toarray()
            self.is_vectorizer_fitted = True
        else:
            tfidf_values = self.vectorizer.transform(raw_documents).toarray()

        for line, tfidf_value in zip(structured_data, tfidf_values):
            feature = {
                'template': line['template'],
                'parameters': line['parameters'],
                'tfidf_values': tfidf_value,
                'content': line['content'] # original log line
            }
            #logger.info(f"[{colored('Ensemble model', 'blue')}] --> Feature added: {feature}")
            features_list.append(feature)
            tfidf_values_list.append(tfidf_value)

        # Convert the TF-IDF values to a NumPy array
        self.features_np_array = np.array(tfidf_values_list)

        # Convert the features list to a dictionary, using a unique identifier for each log line
        self.features_dict = {line['content']: feature for line, feature in zip(structured_data, features_list)}

    def align_features(self, features, expected_features):
        features_array = np.array(features)  # Convert to a NumPy array if not already
        logger.info(f"[{colored('Ensemble model', 'blue')}] --> Number of dimensions (features):", np.ndim(features_array))
        logger.info(f"[{colored('Ensemble model', 'blue')}] --> Shape of features:", features_array.shape)
        logger.info(f"[{colored('Ensemble model', 'blue')}] --> Shape of expected features:", features_array.shape)
        aligned_features = []
        for feature in features_array:
            if len(feature) > expected_features:
                aligned_feature = feature[:expected_features]
            # If features are less than expected, pad with zeros
            elif len(feature) < expected_features:
                aligned_feature = np.pad(feature, (0, expected_features - len(feature)), 'constant')
            # If features match the expected number, return as is
            else:
                aligned_feature = feature
            aligned_features.append(aligned_feature)
        return np.array(aligned_features)
    
    # Using the Ensemble model dictionary, obtain a combined numpy array of the anomaly features for each model in the dictionary, preserving shape
    def detect_anomalies(self, combined_anomaly_log_texts_list):
        
        # Iterate through the features dictionary
        for idx, (log_text, feature) in enumerate(self.features_dict.items()):
            score = self.scores[idx]  # Use the original score
            prediction = self.predictions[idx]

            # Update global min and max scores
            self.global_min_score = min(self.global_min_score, score)
            self.global_max_score = max(self.global_max_score, score)

            if prediction == -1 and score < self.threshold:  # Condition to consider higher scores
                self.anomalies['log_text'].append(log_text)
                self.anomalies['score'].append(score)  # Keep the score in its original form
				
   # Using the Ensemble model dictionary, obtain a combined numpy array of the anomaly features for each model in the dictionary, preserving shape
    def display_anomalies(self):
        print("\n\n\n")
        console = Console()
        table = Table(
            show_header=True, 
            expand=True,
            header_style="bold white",
            title_style="underline", 
            caption=f"anomaly threshold: {self.threshold} similarity threshold: {self.similarity_threshold}"

        )
        table.title = f"Ensemble model anomalies"
        table.add_column("Anomaly Probability (%)", justify="left", style="cyan")
        table.add_column("Anomaly Score", justify="left", style="cyan")
        table.add_column("Similar", justify="left", style="green")
        table.add_column("(Combined Ensemble model) Log Line", justify="left", style="white")
        rows = []  # Initialize an empty list to store rows

        if self.anomalies and len(self.anomalies['log_text']) > 0:
            groups = []
            for log_text, score in zip(self.anomalies['log_text'], self.anomalies['score']):
                normalized_score = 100 - ((score - self.global_min_score) / (self.global_max_score - self.global_min_score)) * 100
                found_group = False
                for group in groups:
                    if ratio(log_text, group['log_text']) >= self.similarity_threshold:
                        group['count'] += 1
                        if normalized_score > group['normalized_score']:
                            group['normalized_score'] = normalized_score
                            group['score'] = score
                            group['log_text'] = log_text
                        found_group = True
                        break
                if not found_group:
                    groups.append({'log_text': log_text, 'score': score, 'normalized_score': normalized_score, 'count': 1})

            # Sort groups by 'normalized_score' in descending order
            groups.sort(key=lambda x: x['normalized_score'], reverse=True)

            for group in groups:
                anomaly_probability = "{:.3f}".format(group['normalized_score'])
                anomaly_score = "{:.4f}".format(group['score'])
                rows.append([anomaly_probability, anomaly_score, str(group['count']), self.truncate_log_line(group['log_text'], 175)])

            # Add sorted rows to the table
            for row in rows:
                table.add_row(*row)
        else:
            table.add_row("-", "-", "None found", "None found")

        console.print(table)
        print("\n\n\n")
