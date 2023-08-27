# Standard Library
import sys
import os
import logging
import sqlite3
import traceback
import json
import io
import time
from collections import defaultdict

# Third-Party Libraries
import nltk
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import joblib
from threading import Lock
from prettytable import PrettyTable
import Levenshtein
from Levenshtein import ratio
from termcolor import colored
from rich.console import Console
from rich.table import Table
from rich.style import Style

# Local Modules
from database_manager import DatabaseManager
from log_parser import LogParser
from log_retriever import LogRetriever
from logger_config import logger

# Additional Configuration
logging.getLogger('nltk').setLevel(logging.CRITICAL)

class ModelManager:
    def __init__(self, log_retriever, log_parser, filepath, config):
        self.config = config
        self.database_manager = DatabaseManager()
        self.log_retriever = log_retriever
        self.log_parser = log_parser
        self.individual_model_dict = {}
        self.ensemble_model_path = config.get('ENSEMBLE_MODEL', 'MODEL_PATH')
        self.max_features = int(config.get('INDIVIDUAL_MODELS', 'MAX_FEATURES'))
        self.threshold = float(config.get('INDIVIDUAL_MODELS', 'ANOMALIES_THRESHOLD'))
        self.contamination = float(config.get('INDIVIDUAL_MODELS', 'MODEL_CONTAMINATION'))
        self.similarity_threshold = float(config.get('INDIVIDUAL_MODELS', 'SIMILARITY_THRESHOLD'))
        self.models_directory = config.get('GENERAL', 'MODELS_DIRECTORY')
        self.numpy_directory = config.get('GENERAL', 'NUMPY_DIRECTORY')
        self.log_file_id =  int(self.log_retriever.get_id_from_filepath(filepath))
        
        if self.database_manager.get_model_filename_from_log_filepath(filepath) is None:
            self.model_path = self.generate_model_filename(filepath)
        else:
            self.model_path = self.database_manager.get_model_filename_from_log_filepath(filepath)
        self.logfile_path = filepath

        if len(self.model_path) > 0 and os.path.exists(self.model_path):
            self.load_individual_model()
        else:
            logger.info(f"[{colored(self.logfile_path, 'yellow')}] ---> model not found at path: {self.model_path}, creating new model.")
            if self.create_new_model(self.log_file_id, filepath):
                logger.info(f"[{colored(self.logfile_path, 'yellow')}] ---> creating model id: {self.log_file_id} logfile path: {self.logfile_path} model path: {self.model_path}")
                logger.info(f"[{colored(self.logfile_path, 'yellow')}] ---> model created at path: {self.model_path}")
            else:
                logger.info(f"[{colored(self.logfile_path, 'yellow')}] ---> ERROR: model could not be created at path: {self.model_path}, log file id: {self.log_file_id}, filepath: {filepath}")
        
        self.features_dict = {}
        self.scores = []
        self.predictions = []
        self.anomalies = {'log_text': [], 'score': []}
        self.features_np_array = None
        self.individual_model_dict = {
                'individual_model': self.individual_model,
                'features_np_array': self.features_np_array,
                'features_dict': self.features_dict,
                'scores': self.scores,
                'predictions': self.predictions,
                'anomalies' : self.anomalies,
                'log_id': self.log_file_id,
                'model_filepath': self.model_path,
                'log_filepath': self.logfile_path
            }
        # Initialize TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'))
        self.is_vectorizer_fitted = False

    def initialize_model(self):
        # Common method to initialize the model
        return IsolationForest(contamination=float(self.contamination))
    
    def generate_model_filename(self, filepath):
        # Ensure directory exists
        os.makedirs(self.models_directory, exist_ok=True)
        converted_path = filepath.replace("/", "_").replace(".log", "_log.pkl")
        return f"{self.models_directory}/model_{converted_path}"
    
    def generate_model_filename(self, filepath):
        # Ensure directory exists
        os.makedirs(f"{self.models_directory}", exist_ok=True)
        # Replace slashes with tildes and keep the dots in place
        converted_path = filepath.replace("/", "~")
        
        return f"{self.models_directory}/{converted_path}.pkl"
    
    def create_new_model(self, log_file_id, filepath):
        try:
            self.individual_model_dict['individual_model'] = self.initialize_model()
            self.individual_model = self.individual_model_dict['individual_model']
            self.model_path = self.generate_model_filename(filepath)
            self.database_manager.set_model_filename(log_file_id, filepath, self.model_path)
        except Exception as e:
            logger.info(f"[{colored(self.logfile_path, 'yellow')}] ---> An error occurred while creating new model: {e}")
            return False
        return True
    
    def load_individual_model(self):
        try:
            self.individual_model_dict = joblib.load(self.model_path)
            self.individual_model = self.individual_model_dict['individual_model']
            logger.info(f"[{colored(self.logfile_path, 'yellow')}] ---> individual model loaded.")
        except FileNotFoundError:
            logger.warn(f"[{colored('Ensemble Model', 'yellow')}] ---> individual model not found at path: {self.ensemble_model_path}, creating new model.")
            self.create_new_model(self.log_file_id, self.logfile_path)
        except KeyError:
            logger.error(f"[{colored('Ensemble Model', 'yellow')}] ---> Key 'individual_model' not found in {self.model_path}. Creating new model.")
            self.create_new_model(self.log_file_id, self.logfile_path)

    def save_individual_model(self):
        try:
            # Save the master model dictionary
            joblib.dump(self.individual_model_dict, self.model_path)
            #print(f"[Individual Model] --> model dictionary saved at path: {self.model_path}")
        except Exception as e:
            logger.info(f"[{colored(self.logfile_path, 'yellow')}] ---> ERROR: exception saving individual model dictionary: {str(e)}")
    
    def train_individual_model(self):
        # Assuming self.features_np_array contains the TF-IDF values
        X = self.features_np_array
        self.individual_model.fit(X)

        # Get the scores and predictions
        self.scores = self.individual_model.decision_function(X).tolist()
        self.predictions = self.individual_model.predict(X).tolist()

        self.save_individual_model()

    def update_global_scores(self, scores):
        self.global_min_score = min(self.global_min_score, min(scores))
        self.global_max_score = max(self.global_max_score, max(scores))

    def extract_features(self, structured_logs):
        features_list = []
        tfidf_values_list = []
        # Preprocess the Content to get a list of raw documents (strings)
        raw_documents = [line['template'] for line in structured_logs]

        # Check if there are any valid raw documents left
        if not raw_documents:
            logger.info(f"[{colored(self.logfile_path, 'yellow')}] ---> No valid content found in parsed logs")
            return

        logger.info(f"[{colored(self.logfile_path, 'yellow')}] ---> raw documents found: {len(raw_documents)}")

        # Compute TF-IDF values
        if not self.is_vectorizer_fitted:
            tfidf_values = self.vectorizer.fit_transform(raw_documents).toarray()
            self.is_vectorizer_fitted = True
        else:
            tfidf_values = self.vectorizer.transform(raw_documents).toarray()

        for line, tfidf_value in zip(structured_logs, tfidf_values):
          feature = {
              'template': line['template'],
              'parameters': line['parameters'],
              'tfidf_values': tfidf_value,
              'log_text': line['content'] # original log line
          }
          #print(f"[Master Model] --> Feature added: {feature}")
          features_list.append(feature)
          tfidf_values_list.append(tfidf_value)

        # Convert the TF-IDF values to a NumPy array
        self.features_np_array = np.array(tfidf_values_list)

        for line, tfidf_value in zip(structured_logs, tfidf_values):
            feature = {
                'template': line['template'],
                'parameters': line['parameters'],
                'tfidf_values': tfidf_value,
                'log_text': line['content'] # original log line
            }
            features_list.append(feature)

        self.features_dict = {feature['log_text']: feature for feature in features_list}
        self.save_individual_model()

    def detect_anomalies(self, threshold=-0.04):
        # Extract the TF-IDF values and align them to the expected number of features
        X = [feature['tfidf_values'] for feature in self.features_dict.values()]
        expected_features = self.individual_model.n_features_in_
        aligned_X = self.align_features(X, expected_features)

        self.scores = self.individual_model.decision_function(aligned_X)
        predictions_with_threshold = [-1 if score < self.threshold else 1 for score in self.scores]

        anomaly_indices = [i for i in range(len(predictions_with_threshold)) if predictions_with_threshold[i] == -1]
        anomaly_features = [aligned_X[i] for i in anomaly_indices]
        anomaly_log_texts = [feature['log_text'] for index, feature in enumerate(self.features_dict.values()) if index in anomaly_indices]

        # Initialize or update global min and max scores
        self.global_min_score = min(self.scores) if not hasattr(self, 'global_min_score') else min(self.global_min_score, min(self.scores))
        self.global_max_score = max(self.scores) if not hasattr(self, 'global_max_score') else max(self.global_max_score, max(self.scores))

        for idx, anomaly_log_text in zip(anomaly_indices, anomaly_log_texts):
            log_text = list(self.features_dict.keys())[idx]
            score = self.scores[idx]
            prediction = predictions_with_threshold[idx]

            if prediction == -1 and score < self.threshold:
                self.anomalies['log_text'].append(log_text)
                self.anomalies['score'].append(score)
                self.anomalies['log_text'].append(anomaly_log_text) # Directly append the log text

        return anomaly_features, anomaly_log_texts
    
    def display_anomalies(self):
        print("\n\n\n")
        console = Console()

        table = Table(
            title=f"Anomalies - individual model for {self.logfile_path}",
            show_header=True,
            header_style="bold white",
            title_style="underline",
            expand=False,
            caption=f"anomaly threshold: {self.threshold} similarity threshold: {self.similarity_threshold}"
        )
        
        table.add_column("Anomaly Probability(%)", justify="left", style="cyan")
        table.add_column("Anomaly Score", justify="left", style="cyan")
        table.add_column("Similarity", justify="left", style="green")
        table.add_column(f"(Model for {self.logfile_path}) Log Line", justify="left")

        red_style = Style(color="yellow")
        white_style = Style(color="white")
        alternating_style = [red_style, white_style]

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
                            group['log_text'] = log_text[:500]
                        found_group = True
                        break
                if not found_group:
                    groups.append({'log_text': log_text, 'score': score, 'normalized_score': normalized_score, 'count': 1})

            # Sort the groups by anomaly score
            sorted_groups = sorted(groups, key=lambda group: group['score'], reverse=True)

            for idx, group in enumerate(sorted_groups):
                anomaly_probability = "{:.4f}".format(group['normalized_score'])
                anomaly_score = "{:.4f}".format(group['score'])

                style = alternating_style[idx % len(alternating_style)]
                table.add_row(
                    anomaly_probability, anomaly_score, str(group['count']), 
                    group['log_text'], style=style
                )
        else:
            table.add_row("-", "-", "None found")

        console.print(table)
        print("\n\n\n")

    def truncate_log_line(self, log_line, max_length=100):
        if isinstance(log_line, dict):
            content = log_line['content']
            log_line_str = f"{log_line['type']} {log_line['timestamp']} {content[:max_length]}"
        else:
            content = log_line
            log_line_str = content[:max_length]

        return log_line_str + '...' if len(content) > max_length else log_line_str

    
    def align_features(self, features, expected_features):
        features_array = np.array(features)  # Convert to a NumPy array if not already
        logger.info(f"[{colored(self.logfile_path, 'yellow')}] ---> Features (number of dimensions): {np.ndim(features_array)}")
        logger.info(f"[{colored(self.logfile_path, 'yellow')}] ---> Features shape: {np.shape(features_array)}")
        logger.info(f"[{colored(self.logfile_path, 'yellow')}] ---> Features expected: {np.shape(expected_features)}")
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

    def calculate_levenshtein_distances(self, log_lines):
        distances = []
        for i, line1 in enumerate(log_lines):
            row = []
            for j, line2 in enumerate(log_lines):
                if i != j:
                    row.append(Levenshtein.distance(line1, line2))
                else:
                    row.append(0) # Distance to itself is 0
            distances.append(row)
        return distances
    
    def insert_anomaly_log_texts(self, model_name, anomaly_log_texts):
        model_name = os.path.basename(model_name)
        logger.info(f"[{colored(self.logfile_path, 'yellow')}] ---> Storing anomaly log lines for model {model_name}")
        result = self.save_numpy_array(model_name, 'anomaly_log_texts', anomaly_log_texts)
        return result
        
    def insert_anomaly_features(self, model_name, anomaly_features):
        model_name = os.path.basename(model_name)

        logger.info(f"[{colored(self.logfile_path, 'yellow')}] ---> Storing model {model_name} as type {type(anomaly_features)}")
        result = self.save_numpy_array(model_name, 'anomaly_features', anomaly_features)
        return result

    # Function to save NumPy array
    def save_numpy_array(self, model_path, array_type, array):
        # Create numpy directory if it doesn't exist
        if not os.path.exists(self.numpy_directory):
            os.makedirs(self.numpy_directory)

        # Create the full path for the .npy file
        model_path_filename_without_extension = os.path.splitext(os.path.basename(model_path))[0]
        filename = model_path_filename_without_extension + '-' + array_type + '.npy'
        file_path = os.path.join(self.numpy_directory, filename)
 
        # Save the array
        try:
            np.save(file_path, array)
            logger.info(f"[{colored(self.logfile_path,'yellow')}] [{colored('Individual model','magenta')}] ---> Numpy Array saved to {array_type} in {file_path}")
            return True
        except Exception as e:
            logger.info(f"[{colored(self.logfile_path, 'yellow')}] ---> An error occurred while storing {array_type} numpy array: {e}")
            return False
        
    def list_individual_model_paths(self):
        models_directory = self.config.get('GENERAL', 'MODELS_DIRECTORY')
        ensemble_model_file = os.path.basename(self.config.get('ENSEMBLE_MODEL', 'MODEL_PATH'))
        model_files = [f for f in os.listdir(f"{models_directory}") if f.endswith('.pkl') and ensemble_model_file not in f]
        return model_files

