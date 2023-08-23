# Standard Library
import sys
import os
import logging
import sqlite3
import traceback
import json
import io
import time

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

# Local Modules
from database_manager import DatabaseManager
from log_parser import LogParser
from log_retriever import LogRetriever

# Additional Configuration
logging.getLogger('nltk').setLevel(logging.CRITICAL)

#sys.path.insert(0, 'logparser')
#from logparser import Drain

# class NumpyArrayEncoder(JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, np.ndarray):
#             return obj.tolist()
#         return JSONEncoder.default(self, obj)

class ModelManager:
    def __init__(self, log_retriever, log_parser, filepath, anomalies_threshold=-0.04, model_contamination=0.1, max_features=10000, max_samples=100000, models_directory='models/', numpy_directory='numpy/'):
        self.log_retriever = log_retriever
        self.log_parser = log_parser
        self.database_manager = DatabaseManager()
        self.individual_model_dict = {}
        self.contamination = model_contamination
        self.max_features = max_features
        self.max_samples = max_samples
        self.log_file_id =  self.log_retriever.get_id_from_filepath(filepath)

        if self.database_manager.get_model_filename_from_log_filepath(filepath) is None:
            self.model_path = self.generate_model_filename(filepath)
        else:
            self.model_path = self.database_manager.get_model_filename_from_log_filepath(filepath)

        self.logfile_path = filepath

        if len(self.model_path) > 0 and os.path.exists(self.model_path):
            self.load_individual_model()
        else:
            print(f"[{self.logfile_path}] [Individual Model] ---> model not found at path: {self.model_path}, creating new model.")
            if self.create_new_model(self.log_file_id, filepath):
                print(f"[{self.logfile_path}] [Individual Model] ---> creating model id: {self.log_file_id} logfile path: {self.logfile_path} model path: {self.model_path}")
                print(f"[{self.logfile_path}] [Individual Model] ---> model created at path: {self.model_path}")
            else:
                print(f"[{self.logfile_path}] [Individual Model] ---> ERROR: model could not be created at path: {self.model_path}, log file id: {self.log_file_id}, filepath: {filepath}")
        
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
        self.max_samples = max_samples
        self.max_features = max_features
        self.threshold = anomalies_threshold
        self.models_directory = models_directory
        self.numpy_directory = numpy_directory
        # Initialize TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'))
        self.is_vectorizer_fitted = False

    def initialize_model(self):
        # Common method to initialize the model
        contamination = self.contamination
        return IsolationForest(contamination=contamination)
    
    def generate_model_filename(self, filepath):
        # Ensure directory exists
        os.makedirs('models', exist_ok=True)
        converted_path = filepath.replace("/", "_").replace(".log", "_log.pkl")
        return f"models/model_{converted_path}"
    
    def create_new_model(self, log_file_id, filepath):
        try:
            self.individual_model_dict['individual_model'] = self.initialize_model()
            self.individual_model = self.individual_model_dict['individual_model']
            self.model_path = self.generate_model_filename(filepath)
            self.database_manager.set_model_filename(log_file_id, filepath, self.model_path)
        except Exception as e:
            print(f"[{self.logfile_path}] [Individual Model] ---> An error occurred while creating new model: {e}")
            return False
        return True
    
    def load_individual_model(self):
        try:
            self.individual_model_dict = joblib.load(self.model_path)
            self.individual_model = self.individual_model_dict['individual_model']
            print(f"[{self.logfile_path}] ---> individual model loaded.")
        except FileNotFoundError:
            print(f"[Master Model] ---> individual model not found at path: {self.master_model_path}, creating new model.")
            self.create_new_model(self.log_file_id, self.logfile_path)
        except KeyError:
            print(f"[Master Model] ---> Key 'individual_model' not found in {self.model_path}. Creating new model.")
            self.create_new_model(self.log_file_id, self.logfile_path)

    def save_individual_model(self):
        # result = self.database_manager.get_model_filename_from_log_filepath(self.logfile_path)
        # self.model_path = result[0]
        # if self.model_path is None:
        #     print("[Individual Model] --> ERROR: individual model path is not set.")
        #     return
        try:
            # Save the master model dictionary
            joblib.dump(self.individual_model_dict, self.model_path)
            #print(f"[Individual Model] --> model dictionary saved at path: {self.model_path}")
        except Exception as e:
            print(f"[{self.logfile_path}] [Individual Model] --> ERROR: exception saving individual model dictionary: {str(e)}")
    
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
            print("No valid content found in parsed logs")
            return

        print(f"[{self.logfile_path}] [Individual Model] ---> {len(raw_documents)} raw documents found")

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
        # Create a PrettyTable object
        table = PrettyTable()
        table.field_names = ["Anomaly Probability (%)", "Anomaly Score", f"(Model for {self.logfile_path}) Log Line"]
        table.align["Anomaly Probability (%)"] = "l"
        table.align["Anomaly Score"] = "l"
        table.align[f"(Model for {self.logfile_path}) Log Line"] = "l"

        # Check if there are any anomalies
        if self.anomalies and len(self.anomalies['log_text']) > 0:
            # Normalize the scores to a percentage
            for log_text, score in zip(self.anomalies['log_text'], self.anomalies['score']):
                normalized_score = 100 - ((score - self.global_min_score) / (self.global_max_score - self.global_min_score)) * 100
                table.add_row([normalized_score, score, self.truncate_log_line(log_text, 175)])

            # Sort the table by "Anomaly Score" in ascending order
            table.sortby = "Anomaly Score"
            table.reversesort = False
        else:
            # Add a row indicating no anomalies found
            table.add_row(["-", "-", "None found"])

        print(table)

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
        print(f"[{self.logfile_path}] [Individual Model] ---> Features (number of dimensions): {np.ndim(features_array)}")
        print(f"[{self.logfile_path}] [Individual Model] ---> Features shape: {np.shape(features_array)}")
        print(f"[{self.logfile_path}] [Individual Model] ---> Features expected: {np.shape(expected_features)}")
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

    def insert_anomaly_log_texts(self, model_name, anomaly_log_texts):
        model_name = os.path.basename(model_name)
        print(f"[{self.logfile_path}] [Individual Model] ---> Storing anomaly log lines for model {model_name}")
        result = self.save_numpy_array(model_name, 'anomaly_log_texts', anomaly_log_texts)
        return result
        
    def insert_anomaly_features(self, model_name, anomaly_features):
        model_name = os.path.basename(model_name)

        print(f"[{self.logfile_path}] [Individual Model] ---> Storing model {model_name} as type {type(anomaly_features)}")
        result = self.save_numpy_array(model_name, 'anomaly_features', anomaly_features)
        return result

    # Function to save NumPy array
    def save_numpy_array(self, model_path, array_type, array):
        # Create numpy directory if it doesn't exist
        numpy_directory = 'numpy/'
        if not os.path.exists(numpy_directory):
            os.makedirs(numpy_directory)

        # Create the full path for the .npy file
        model_path_filename_without_extension = os.path.splitext(os.path.basename(model_path))[0]
        filename = model_path_filename_without_extension + '-' + array_type + '.npy'
        file_path = os.path.join(numpy_directory, filename)
 
        # Save the array
        try:
            np.save(file_path, array)
            print(f"[{self.logfile_path}] [Individual Model] ---> Numpy Array saved to {array_type} in {file_path}")
            return True
        except Exception as e:
            print(f"[{self.logfile_path}] [Individual Model] ---> An error occurred while storing {array_type} numpy array: {e}")
            return False
        
    def list_individual_model_paths(cls):
       model_files = [f for f in os.listdir('models/') if f.endswith('.pkl') and f != 'master_model.pkl']  # Exclude master_models.pkl
       return model_files

