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

# Local Modules
from model_manager import ModelManager
from database_manager import DatabaseManager
from log_parser import LogParser
from log_retriever import LogRetriever

# Additional Configuration
logging.getLogger('nltk').setLevel(logging.CRITICAL)

MASTER_MODEL_PATH = 'models/master_model.pkl'

class MasterModel:
    def __init__(self, model_manager, log_retriever, log_parser, individual_model_paths=None, master_model_path=None, threshold=0.1, master_contamination=0.005, max_num_models=20, max_features=1000, max_samples=10000):
        self.master_model_dict = {}
        # If a master model path is provided, load it; otherwise, create a new master model
        self.master_model_path = master_model_path
        self.max_num_models = max_num_models
        self.threshold = threshold
        self.contamination = master_contamination
        self.max_samples = max_samples
        self.max_features = max_features

        # If individual model paths are provided, load them
        # Load the individual models
        self.individual_models = {}
        if individual_model_paths:
            self.individual_models = {path: joblib.load("models/" + path) for path in individual_model_paths}

        self.features_dict = {}
        self.features_np_array = None
        self.scores = []
        self.predictions = []
        self.anomalies = {'log_text': [], 'score': []}
        # Create the master model dictionary
        self.master_model_dict = {
                'master_model': None,
                'individual_models': self.individual_models,
                'features_np_array': self.features_np_array,
                'features_dict': self.features_dict,
                'scores': self.scores,
                'predictions': self.predictions,
                'anomalies' : self.anomalies
        }

        if master_model_path and os.path.exists(master_model_path):
            self.load_master_model()
        else:
            print(f"[Master Model] --> Master Model not found at path: {master_model_path}, creating new model.")
            self.create_new_model()

        #print(f"[Master Model] --> Master Model dictionary: {self.master_model_dict}")
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
        return IsolationForest(contamination=0.1)

    def create_new_model(self):
        self.master_model = self.initialize_model()
        self.master_model_dict['master_model'] = self.master_model
        self.save_master_model()
        return self.master_model

    def load_master_model(self):
        try:
            self.master_model_dict = joblib.load(self.master_model_path)
            self.master_model = self.master_model_dict.get('master_model')  # Use get method
            if self.master_model is None:
                raise KeyError('master_model')
            print("[Master Model] --> Master Model loaded.")
        except (FileNotFoundError, KeyError):
            print(f"[Master Model] --> Master Model not found at path: {self.master_model_path}, creating new model.")
            self.create_new_model()

    def save_master_model(self):
        try:
            # Save the master model dictionary
            joblib.dump(self.master_model_dict, self.master_model_path)
            #print(f"[Master Model] --> Master Model dictionary saved at path: {self.master_model_path}")
        except Exception as e:
            print(f"[Master Model] --> Error saving Master Model dictionary: {str(e)}")

    def train_master_model(self):
        # Assuming self.features_np_array contains the TF-IDF values
        X = self.features_np_array
        self.master_model.fit(X)

        # Get the scores and predictions
        self.scores = self.master_model.decision_function(X).tolist()
        self.predictions = self.master_model.predict(X).tolist()
        self.save_master_model()

    def update_global_scores(self, scores):
        self.global_min_score = min(self.global_min_score, min(scores))
        self.global_max_score = max(self.global_max_score, max(scores))

    def get_individual_model_filepath_from_model_name_by_type(self, model_name, array_type):
        numpy_directory = "numpy/"
        model_path_filename_without_extension = os.path.splitext(os.path.basename(model_name))[0]
        filename = model_path_filename_without_extension + '-' + array_type + '.npy'
        file_path = os.path.join(numpy_directory, filename)
        return file_path

    def truncate_log_line(self, log_line, max_length=100):
        if isinstance(log_line, dict):
            content = log_line['content']
            log_line_str = f"{log_line['type']} {log_line['timestamp']} {content[:max_length]}"
        else:
            content = log_line
            log_line_str = content[:max_length]

        return log_line_str + '...' if len(content) > max_length else log_line_str

    def predict(self, combined_anomaly_features, master_model_dict):
        predictions = []
        # Test individual predictions
        individual_models = list(master_model_dict['individual_models'].items())[:self.max_num_models]
        for model_name, features in zip(individual_models, combined_anomaly_features[:self.max_num_models]):
            model = master_model_dict['individual_models'][model_name]
            print(f"Model {model_name} expects {model.n_features_in_} features")
            features_path = self.get_individual_model_filepath_from_model_name_by_type(model_name, 'anomaly_features')
            features = np.load(features_path)
            print(f"Loaded features shape: {features.shape}")
            individual_prediction = model.predict(features)  # Removed the reshape here
            print(f"Individual prediction: {individual_prediction}")
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

    def dump_model_to_file(self, master_model, models, filepath):
        serializable_model = self.prepare_for_serialization(master_model, models)
        joblib.dump(serializable_model, filepath)

    def prepare_for_serialization(self, master_model, models):
            # Return a dictionary with attributes that can be pickled
            return {
                'master_model': master_model,
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
        print(f"get_anomaly_features --> {type(anomaly_features)}  {anomaly_features.shape}")
        return anomaly_features

    def load_numpy_array(self, model_path, array_type):
        # Construct the full path for the .npy file
        numpy_directory = "numpy/"
        model_path_filename_without_extension = os.path.splitext(os.path.basename(model_path))[0]
        filename = f"{model_path_filename_without_extension}-{array_type}.npy"
        file_path = os.path.join(numpy_directory, filename)

        # Load the array if the file exists
        if os.path.exists(file_path):
            array = np.load(file_path, allow_pickle=True)
            #print(f"[Master Model] --> Numpy array with features loaded from {file_path} for model {model_path}")
            return array
        else:
            #This will trigger if the log file is empty.
            #print(f"[Master Model] Numpy file {file_path} not found for model {model_path}")
            return []

    def get_model_name(self, model_path):
        # You can extract the model name from the path, e.g., by taking the file name without the extension
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        return model_name

    def get_individual_model_name(self, model):
        model_path = self.get_model_path(model) # Function to retrieve the path where the model is saved
        model_name = os.path.basename(model_path).split('.')[0] # Assuming the name is the file name without extension
        return model_name
    
    # Using the master model dictionary, obtain a combined list of the anomaly log texts for each model in the dictionary
    def get_combined_anomaly_log_texts(self):
        anomaly_log_texts_list = []
        for model_name, model in self.individual_models.items():
            # Get the model name, and extract the anomaly log lines from the database
            anomaly_log_texts = np.array(self.get_individual_anomaly_log_texts(model_name))  # Fetch log texts from database based on the model name
            if anomaly_log_texts.size > 0:  # Check if the array has elements
                anomaly_log_texts_list.append(anomaly_log_texts)
            else:
                print(f"[Master Model] --> No anomalies found for model {model_name}")


        # Combine the arrays into a single array
        if anomaly_log_texts_list:
            combined_anomaly_log_texts_array = np.concatenate(anomaly_log_texts_list)
        else:
            combined_anomaly_log_texts_array = np.array([])

        if combined_anomaly_log_texts_array.size > 0:
            #print(f"[Master Model] --> Returning combined anomaly log texts array of shape {combined_anomaly_log_texts_array.shape}")
            #print(f"[Master Model] --> Returning combined anomaly log texts array of type {type(combined_anomaly_log_texts_array)}")
            #print(f"[Master Model] --> Returning combined anomaly log texts array of size {np.size(combined_anomaly_log_texts_array)}")
            return combined_anomaly_log_texts_array
        else:
            print(f"[Master Model] --> No anomalies returned from master.model.get_combined_anomaly_log_texts()")
            return np.array([])

        
    # Using the master model dictionary, obtain a combined numpy array of the anomaly features for each model in the dictionary, preserving shape
    def get_combined_anomaly_features(self, master_model_dict):
        anomaly_features_list = []

        # Iterating through individual models
        for idx, (model_name, model) in enumerate(master_model_dict['individual_models'].items()):
            print(f"{model_name} --> Model expects {model.n_features_in_} anomaly features")
            # Fetch anomaly features from disk
            anomaly_features_np_array = np.array(self.get_anomaly_features(model_name))
            anomaly_features_list.append(anomaly_features_np_array)  # Append each array to the list

            print(f"combined function anomaly_features_np_array --> {type(anomaly_features_np_array)} to length {len(anomaly_features_np_array)}")
            print(f"combined function anomaly_features_np_array.shape --> shape {len(anomaly_features_np_array.shape)}")
            print(f"combined function anomaly_features_np_array --> value {anomaly_features_np_array}")
            print(f"combined function anomaly_features_list --> length {len(anomaly_features_list)}")
            print(f"combined function anomaly_features_list --> size {len(anomaly_features_list)}")

        # Call the combine_and_pad_arrays function with the list of arrays
        combined_anomaly_features_array = self.combine_and_pad_arrays(anomaly_features_list)

        if combined_anomaly_features_array.size > 0:
            print(f"[Master Model] --> Returning combined anomaly features array of shape {combined_anomaly_features_array.shape}")
            print(f"[Master Model] --> Returning combined anomaly features array of type {type(combined_anomaly_features_array)}")
            print(f"[Master Model] --> Returning combined anomaly features array of size {np.size(combined_anomaly_features_array)}")
            return combined_anomaly_features_array
        else:
            print(f"[Master Model] --> No anomaly features returned from master_model.get_combined_anomaly_features()")
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
    
    # Using the master model dictionary, obtain a combined numpy array of the anomaly features for each model in the dictionary, preserving shape
    def extract_features(self, structured_data):
        features_list = []
        tfidf_values_list = []

        # Preprocess the Content to get a list of raw documents (strings)
        raw_documents = [line['template'] for line in structured_data]

        # Check if there are any valid raw documents left
        if not raw_documents:
            print("No valid content found in parsed logs")
            return

        print(f"[Master Model] --> Raw documents found: {len(raw_documents)}")

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
            #print(f"[Master Model] --> Feature added: {feature}")
            features_list.append(feature)
            tfidf_values_list.append(tfidf_value)

        # Convert the TF-IDF values to a NumPy array
        self.features_np_array = np.array(tfidf_values_list)

        # Convert the features list to a dictionary, using a unique identifier for each log line
        self.features_dict = {line['content']: feature for line, feature in zip(structured_data, features_list)}

    def align_features(self, features, expected_features):
        features_array = np.array(features)  # Convert to a NumPy array if not already
        print("[Master Model] --> Number of dimensions (features):", np.ndim(features_array))
        print("[Master Model] --> Shape of features:", features_array.shape)
        print("[Master Model] --> Shape of expected features:", features_array.shape)
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
    
    # Using the master model dictionary, obtain a combined numpy array of the anomaly features for each model in the dictionary, preserving shape
    def detect_anomalies(self, combined_anomaly_log_texts_list, threshold=-0.4):
        threshold = self.threshold

        # Iterate through the features dictionary
        for idx, (log_text, feature) in enumerate(self.features_dict.items()):
            score = self.scores[idx]  # Use the original score
            prediction = self.predictions[idx]

            # Update global min and max scores
            self.global_min_score = min(self.global_min_score, score)
            self.global_max_score = max(self.global_max_score, score)

            if prediction == -1 and score < threshold:  # Condition to consider higher scores
                self.anomalies['log_text'].append(log_text)
                self.anomalies['score'].append(score)  # Keep the score in its original form

    def display_anomalies(self):
        # Create a PrettyTable object
        table = PrettyTable()
        table.field_names = ["Anomaly Probability (%)", "Anomaly Score", "Similar", "(Combined Master Model) Log Line"]
        table.align["Anomaly Probability (%)"] = "l"
        table.align["Anomaly Score"] = "l"
        table.align["Similar"] = "l"
        table.align["(Combined Master Model) Log Line"] = "l"

        # Similarity threshold
        self.similarity_threshold = 0.8

        # Check if there are any anomalies
        if self.anomalies and len(self.anomalies['log_text']) > 0:
            # Group similar log lines
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

            # Add rows to the table
            for group in groups:
                anomaly_probability = "{:.3f}".format(group['normalized_score'])
                anomaly_score = "{:.4f}".format(group['score'])
                table.add_row([anomaly_probability, anomaly_score, group['count'], self.truncate_log_line(group['log_text'], 175)])

            # Sort the table by "Anomaly Score" in ascending order
            table.sortby = "Anomaly Score"
            table.reversesort = True
        else:
            # Add a row indicating no anomalies found
            table.add_row(["-", "-", "None found", "None found"])

        print(table)


    
    @classmethod
    def load_individual_models_deprecated(cls):
        model_files = [f for f in os.listdir('models/') if f.endswith('.pkl') and f != 'master_model.pkl']  # Exclude master_models.pkl
        models = []
        models_expected_features = []
        for model_file in model_files[:self.max_num_models]:
            model_path = os.path.join('models/', model_file)
            print(f"[Individual Model] --> Attempting to load model from: {model_path}") # Debugging print statement
            if os.path.exists(model_path):
                try:
                    individual_model = joblib.load(model_path)
                    if isinstance(individual_model, IsolationForest):
                        first_estimator = individual_model.estimators_[0]
                        models_expected_features.append(first_estimator.tree_.n_features)
                        models.append(individual_model)
                except Exception as e:
                    print(f"Error loading model from {model_path}: {str(e)}") # Error handling
            else:
                print(f"Individual Model at {model_path} does not exist.")
                return models, models_expected_features
        return models, models_expected_features












