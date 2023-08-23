# Standard Library
import argparse
import time
import os
import logging
from multiprocessing import Pool
from functools import partial

# Third-Party Libraries
import joblib
import numpy as np
from nltk.corpus import stopwords

# Local Modules
from log_retriever import LogRetriever
from database_manager import DatabaseManager
from log_monitor import LogMonitor
from task_scheduler import TaskScheduler
from log_parser import LogParser
from model_manager import ModelManager
from model_manager_factory import ModelManagerFactory
from master_model import MasterModel

# Additional Configuration
logging.getLogger('nltk').setLevel(logging.CRITICAL)

# Set the polling interval to 1 minute for filesystem logs
POLLING_INTERVAL_MINUTES = 1
# Where to store the master model
MASTER_MODEL_PATH = 'models/master_model.pkl'
# Where to store the individual/master models and drain3 state file
MODELS_DIRECTORY = 'models/'
# Where to store the numpy files for the individual/master models
NUMPY_DIRECTORY = 'numpy/'
# The proportion of outliers in the individual models. It's used to define the threshold for outlier scores. Increasing this value will result in more anomalies being detected (per log).
INDIVIDUAL_MODEL_CONTAMINATION = 0.1
# The proportion of outliers in the master model. It's used to define the threshold for outlier scores. Increasing this value will result in more anomalies being detected in the master model.
MASTER_MODEL_CONTAMINATION = 0.1
# The percentage of anomalies expected in the data (0.00%) - there is an anomaly found if the score less than threshold, with the lower the score the more likely it is to be an anomaly. It's used to calculate the threshold in the individual (per log) models. Often the anomaly scores are negative numbers
INDIVIDUAL_ANOMALIES_THRESHOLD = -0.04
# The percentage of anomalies expected in the data (0.00%) - there is an anomaly found if the score less than threshold, with the lower the score the more likely it is to be an anomaly. It's used to calculate the threshold in the combined master model. Often the anomaly scores are negative numbers
MASTER_MODEL_ANOMALIES_THRESHOLD = 0.00
# The maximum number of features to consider when extracting features from the log lines (higher numbers mean more memory usage and this could affect performance)
INDIVIDUAL_MODEL_MAX_FEATURES = 3000
# The maximum number of features to consider when extracting features from the log lines (a higher number means more memory usage but potentially less sensitive to anomalies)
MASTER_MODEL_MAX_FEATURES = 1000
# The maximum number of samples to consider when extracting features from the log lines (lower numbers mean less memory usage and make the model more sensitive to anomalies)
INDIVIDUAL_MODEL_MAX_SAMPLES = 0.2
# The maximum number of samples to consider when extracting features from the individual models (lower numbers mean less memory usage and make the model more sensitive to anomalies)
MASTER_MODEL_MAX_SAMPLES = 0.2
# The number of individual models to consider when doing calculations on the master model for anomaly detection
MAX_NUM_MODELS_TO_CONSIDER = 20

class AnomalyHandler:
    def handle_anomaly(self, anomaly):
    # Handle the anomaly (e.g., send an alert, log it)
        pass

class Logger:
    def log(self, message):
    # Log the message (e.g., to a file or console)
        pass

def parse_arguments():
    parser = argparse.ArgumentParser(description='Log processing and anomaly detection.')
    # Cloudwatch
    # parser.add_argument('--log_group', required=True, help='Log group name.')
    # parser.add_argument('--log_stream', required=True, help='Log stream name.')
    # parser.add_argument('--start_time', required=True, help='Start time for log retrieval.')
    # parser.add_argument('--end_time', required=True, help='End time for log retrieval.')
    parser.add_argument('--log_dir', required=True, help='Directory for filesystem logs.')
    return parser.parse_args()

def is_log_content_valid(log_content):
    # Initialize stop_words 
    stop_words = set(stopwords.words('english'))
    stripped_content = log_content.strip()
    return bool(stripped_content) and stripped_content not in stop_words

def truncate_log_line(log_line, max_length=100):
    if isinstance(log_line, dict):
        content = log_line['content']
        log_line_str = f"{log_line['type']} {log_line['timestamp']} {content[:max_length]}"
    else:
        content = log_line
        log_line_str = content[:max_length]

    return log_line_str + '...' if len(content) > max_length else log_line_str

def get_files_to_process(log_dir, log_retriever, db_manager):
    file_paths_with_hash = []
    
    # Traverse through the log directory
    for root, _, files in os.walk(log_dir):
        file_paths = [os.path.join(root, filename) for filename in files]

        # Calculate hash values and map them to filepaths
        hash_values = log_retriever.parallel_hash_files(file_paths)
        for filepath, hash_value in zip(file_paths, hash_values):
            #print(f"Inspecting { filepath } ")
            if db_manager.should_process_file(filepath, hash_value, threshold_minutes=POLLING_INTERVAL_MINUTES) is True:
                #print(f"Considering {filepath} a new file.")
                file_paths_with_hash.append((filepath, hash_value))
            else:
                log_retriever.update_file_last_checked_timestamp(filepath)
                #print(f"Already processed {filepath} skipping...")

    return file_paths_with_hash

def process_with_args(log_dir, args):
    filepath, hash_value = args
    process_file(filepath, hash_value, log_dir)

def process_file(filepath, hash_value, log_dir):
    # Create a new connection inside the process
    database_manager_instance = DatabaseManager()
    local_connection = database_manager_instance.get_connection()
    log_retriever = LogRetriever()  # Create a local instance if necessary
    log_parser = LogParser()  # Create a local instance if necessary
    print(f"[{filepath}] ---> Starting work on chunk {hash_value}")
    #factory = ModelManagerFactory()
    #manager = factory.get_manager(filepath, log_retriever, log_parser, database_manager_instance) #Create a separate instance of model manager per file
    #model_manager = manager # use the model manager for that specific file

    raw_log, log_file_id = log_retriever.retrieve_from_filesystem(filepath, hash_value)
    #print(f"[{filepath}] ---> Marking chunk id: {log_file_id} as in process")
    model_manager_instance = ModelManager(log_retriever, log_parser, str(filepath), INDIVIDUAL_ANOMALIES_THRESHOLD, INDIVIDUAL_MODEL_CONTAMINATION, INDIVIDUAL_MODEL_MAX_FEATURES, INDIVIDUAL_MODEL_MAX_SAMPLES, MODELS_DIRECTORY, NUMPY_DIRECTORY)
    # Start processing the file
    database_manager_instance.start_processing(log_file_id)

    if len(raw_log) > 1:
        print(f"[{filepath}] [Individual model] ---> Pre-procesing {len(raw_log)} raw log...")
        #parsed_logs = [log_parser.parse_log_lines(line) for line in raw_log]
        print(f"[{filepath}] [Individual model] ---> Processing {len(raw_log)} log...")
        structured_logs = log_parser.parse_log_lines(filepath, raw_log)
        print(f"[{filepath}] [Individual model] ---> Storing cleaned logs: {len(structured_logs)} lines")
        storage_result = database_manager_instance.store_logs_drain3(filepath, structured_logs)
        if storage_result:
            print(f"[{filepath}] [Individual model] ---> Extracting TF-IDF Features from {len(structured_logs)} lines")
            model_manager_instance.extract_features(structured_logs)
            print(f"[{filepath}] [Individual model] ---> Completed extraction of TF-IDF features, {len(model_manager_instance.features_dict)} items (features) found, array size: {np.size(model_manager_instance.features_np_array)} shape: {np.shape(model_manager_instance.features_np_array)}")
            #log_file_ids = log_retriever.get_all_ids()
            if len(model_manager_instance.features_dict) > 0:
                #Training model with features
                print(f"[{filepath}] [Individual model] ---> Training model with {len(model_manager_instance.features_dict)} features.")
                model_manager_instance.train_individual_model()
                
                #Detect anomalies
                print(f"[{filepath}] [Individual model] ---> Starting anomaly detection using {len(structured_logs)} cleaned log lines")
                result = model_manager_instance.detect_anomalies()
                if result is not None:
                    anomaly_features, anomaly_log_texts = result
                else:
                    anomaly_features, anomaly_log_texts = [], []
               
                #print(f"{filepath} [Individual model] ---> anomalies scores ({len(model_manager_instance.anomalies)}): {model_manager_instance.anomalies['score']}")
                #print(f"{filepath} [Individual model] ---> anomalies predictions ({len(model_manager_instance.predictions)}): {model_manager_instance.predictions}")
                #Display anomalies found in a table
                model_manager_instance.display_anomalies()
                #print(f"{filepath} [Individual model] ---> anomaly feature extraction complete ({len(anomaly_features)}) found.")
                #print(f"{filepath} [Individual model] ---> anomaly features ({anomaly_features}")
                
                
                #print(f"{filepath} [Individual model] ---> Storing records (anomaly features).")
                insert_anomaly_features_result = model_manager_instance.insert_anomaly_features(model_manager_instance.model_path, np.array(anomaly_features))
                #Store anomaly features
                if insert_anomaly_features_result:
                    print(f"[{filepath}] [Individual model] ---> Stored records (anomaly features).")
                else:
                    print(f"[{filepath}] [Individual model] ---> ERROR: Failed to store records (anomaly features).")

                #print(f"{filepath} [Individual model] ---> Storing records (anomaly log lines)")
                insert_anomaly_log_texts_result = model_manager_instance.insert_anomaly_log_texts(model_manager_instance.model_path,np.array(anomaly_log_texts))
                if insert_anomaly_log_texts_result:
                    print(f"[{filepath}] [Individual model] ---> Stored records (anomaly features).")
                else:
                    print(f"[{filepath}] [Individual model] ---> ERROR: Failed to store records (anomaly log texts).")
                
                
                #print(f"{filepath} [Individual model] ---> anomalies identified: {len(anomaly_log_texts)} (truncated to 200 characters)")
                #print(f"{filepath} --> ---------------------------------------------------------------------------------------------------------------------------------------------------\n\n")
                #for anomaly in anomaly_log_texts:
                #    print(f"{truncate_log_line(anomaly['content'], 200)}")
                #print(f"\n{filepath} --> ---------------------------------------------------------------------------------------------------------------------------------------------------")

                # Train model if features are produced on TD-IDF values represented in anomaly_features
                #print(f"{filepath} [Individual model] ---> training individual model with anomaly features ({len(anomaly_features)}).")
                #model_manager_instance.train_model_anomalies(log_file_id, filepath, features, anomaly_features)
                #print(f"{filepath} [Individual model] ---> trained on Anomaly Features.")
                #print(f"{filepath} --> [Master model] loading individual models.")

                # Process master model
                #process_master_model(model_manager_instance, model_manager_instance.features_dict, anomaly_features, anomaly_log_texts)
                process_master_model(model_manager_instance, log_retriever, log_parser)
                
            else:
                print("No features to train the model.")
        else:
            print("Storage failed.")
    database_manager_instance.end_processing(log_file_id)
    local_connection.close()

def process_master_model(model_manager_instance, log_retriever, log_parser):
    # Check if at least 2 individual models exist
    if len(os.listdir('models/')) >= 4:
        #Returns a list of models (model file names)
        model_paths = model_manager_instance.list_individual_model_paths()
        master_model_instance = MasterModel(model_manager_instance, log_retriever, log_parser, model_paths, MASTER_MODEL_PATH, MASTER_MODEL_ANOMALIES_THRESHOLD, MAX_NUM_MODELS_TO_CONSIDER, MASTER_MODEL_MAX_FEATURES, MASTER_MODEL_MAX_SAMPLES)
        
        # Creating a dictionary for individual models
        if len(model_paths) < 2:
            print(f"[Master Model] --> At least 2 individual models must be loaded, but only {len(model_paths)} were found of {type(model_paths)}, waiting for more models.")
        elif len(model_paths) > 2:
            # Creating a dictionary for individual models (contains joblib models)
            #individual_models = {model_paths[i]: joblib.load("models/" + model_paths[i]) for i in range(len(model_paths))}

            # Creating a master model object
            #master_model_obj = master_model_instance.initialize_model()

            # Constructing the final dictionary structure
            #master_model_w_models_dict = {'master_model': master_model_obj, 'individual_models': individual_models}

            #print(f"[Master model] --> Master model with individual models: {master_model_w_models_dict}")
            
            # Combine individual models into master model
            # {'master_model': IsolationForest(contamination=0.01), 
            #   'individual_models': {
            #       'model_sample_input_logs_file-instance-k_log.pkl': IsolationForest(contamination=0.01),
            #       'model_sample_input_logs_file-instance-j_log.pkl': IsolationForest(contamination=0.01), 
            #       'model_sample_input_logs_file-instance-l_log.pkl': IsolationForest(contamination=0.01)
            #    }
            # }

            print(f"[Master model] --> saving master model.")
            master_model_instance.save_master_model()
            #joblib.dump(master_model_w_models_dict, MASTER_MODEL_PATH)
            
            #Load recently saved model from disk
            #print(f"[Master model] --> reloading master model.")
            #master_model_dict = joblib.load(MASTER_MODEL_PATH)
            #print(f"[Master model] --> successfully compiled {len(model_paths)} individual models into a dictionary.")

            # Use master_model for predictions
            print(f"[Master Model] --> [Master model] generating predictions for anomalies based on individual models")
            #print(f"[Master Model] --> Shape of individual model's features: {np.shape(features)}")
            #print(f"[Master Model] --> Shape of individual model's anomaly features: {np.shape(anomaly_features)}")
#           print(f"[Master Model] --> Getting combined anomaly features array")
#           combined_anomaly_features_array = master_model_instance.get_combined_anomaly_features(master_model_dict)
            print(f"[Master Model] --> Getting combined anomaly log texts list")
           # print(f"[Master Model] --> Master Model Dict: {master_model_instance.master_model_dict}")                         
            combined_anomaly_log_texts_list = master_model_instance.get_combined_anomaly_log_texts()
            raw_lines = []
            for line in combined_anomaly_log_texts_list.tolist():
                 raw_lines.append(line)

            if len(combined_anomaly_log_texts_list.tolist()) > 0: 
                print(f"[Master Model] --> Combined anomaly log texts list length: ({len(combined_anomaly_log_texts_list.tolist())}) items")
                print(f"[Master Model] --> Parsing combined anomaly texts: ({len(raw_lines)}) items")
                structured_logs = log_parser.parse_log_lines("Master Model", raw_lines)
                print(f"[Master Model] --> Structured Logs: ({len(structured_logs)})")
                if len(structured_logs) > 0:
                    master_model_instance.extract_features(structured_logs)
                    print(f"[Master Model] --> Master features: ({len(master_model_instance.features_dict)}) dictionary items, numpy array size: {np.size(master_model_instance.features_np_array)} shape: {np.shape(master_model_instance.features_np_array)}")
                    master_model = master_model_instance.train_master_model()
                    print(f"[Master Model] --> Model trained.")
                    print(f"[Master Model] --> Detecting anomalies")
                    master_model_instance.detect_anomalies(combined_anomaly_log_texts_list, MASTER_MODEL_ANOMALIES_THRESHOLD)
                    print(f"[Master Model] --> Displaying anomalies")
                    master_model_instance.display_anomalies()
                else:
                    print(f"[Master Model] --> No combined anomalies found, skipping master model anomaly detection...")
            else:
                print(f"[Master Model] --> No anomalies found yet, skipping master model anomaly detection...")

        #    # master_anomaly_features, master_anomaly_log_texts = master_model_instance.detect_anomalies(model_manager, master_model_dict, combined_anomaly_features_array, combined_anomaly_log_texts_list, anomalies_threshold)
        #     print(f"[Master Model] --> anomaly features ({master_anomalies['features']})")
        #     print(f"[Master Model] --> anomalies identified: {len(master_anomalies['log_line'])} (truncated to 200 characters)")
        #     print(f"[Master Model] --> ---------------------------------------------------------------------------------------------------------------------------------------------------\n\n")
        #     for anomaly in master_anomalies['log_line']:
        #         print(f"{truncate_log_line(anomaly, 200)}")
        #     print(f"\n[Master Model] --> ---------------------------------------------------------------------------------------------------------------------------------------------------")
        else:
            print(f"[Master Model] --> Not enough individual models to combine.")
def main():
    #Initialize Arguments
    args = parse_arguments()
    # Cloudwatch
    # log_group = args.log_group
    # log_stream = args.log_stream
    # start_time = int(args.start_time)
    # end_time = int(args.end_time)
    log_dir = args.log_dir

    # Initialize the DatabaseManager
    database_manager = DatabaseManager()
    
    #database_manager.create_table()

    #Set up and start the log monitor, Task Scheduler, Log Parser, Model Manager
    log_monitor = LogMonitor(database_manager)
    task_scheduler = TaskScheduler()
    log_parser = LogParser()

    # Initialize LogRetriever and Retrieve logs
    # Cloudwatch
    # log_retriever = LogRetriever(log_group, log_stream)  # Pass log_group and log_stream here
    log_retriever = LogRetriever()  # Pass log_group and log_stream here
    working_file_hashes = {} # A list of hashes which have already been processed

    # Initialize ModelManager using log_retriever and log_parser
    conn = database_manager.get_connection()
    #model_manager = ModelManager(log_retriever, log_parser, database_manager) # Pass the DatabaseManager object

    # Cloudwatch
    #logs = log_retriever.retrieve_from_cloudwatch(start_time, end_time)  # Pass start_time and end_time here
    #logs += log_retriever.retrieve_from_filesystem(log_dir)
    filepath = None
    while True:
        file_paths_with_hash = get_files_to_process(log_dir, log_retriever, database_manager)
        #print(f"Found {len(file_paths_with_hash)} files to process.")
        
        # Process only the files that meet the conditions
        for (filepath, hash_value) in file_paths_with_hash:
            if database_manager.should_process_file(filepath, hash_value, threshold_minutes=POLLING_INTERVAL_MINUTES) is True:
                process_with_args(log_dir, (filepath, hash_value))
            else:
                print(f"{filepath} --> File is not ready to be processed.")
        if filepath:
           log_retriever.update_file_last_checked_timestamp(filepath)
        time.sleep(60) # wait 1 minute in between polls for files to process

    # Monitor logs for changes
    # Main loop
    # from functools import partial
    # # Define pool outside of loop
    # pool = Pool()

    # while True:
    #     file_paths_with_hash = get_files_to_process(log_dir, log_retriever, database_manager)
    #     print(f"Found {len(file_paths_with_hash)} files to process.")
    #     # Process only the files that meet the conditions

    #     for (filepath, hash_value) in file_paths_with_hash:
    #         if database_manager.should_process_file(filepath, hash_value, threshold_minutes=POLLING_INTERVAL_MINUTES) is True:
    #             if file_paths_with_hash:
    #                 process_func = partial(process_with_args, log_dir)

    #                 # Add new jobs individually
    #                 async_results = []
    #                 result = pool.apply_async(process_func, ((filepath, hash_value),))
    #                 async_results.append(result)
    #                 time.sleep(1)

    #                 # Wait for the tasks to complete, if needed
    #                 [res.get() for res in async_results]
    #         else:
    #             print(f"File {filepath} is not ready to be processed.")
    #     log_retriever.update_file_last_checked_timestamp(filepath)

        # # Sleep for 60 seconds (1 minute) before checking for files again
        # time.sleep(5)
        # # When you are done with the pool
        # pool.close()
        # pool.join()



if __name__ == "__main__":
    main()
