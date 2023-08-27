# Standard Library
import argparse
import time
import os
import sys
import logging
import configparser
from multiprocessing import Pool
from functools import partial

# Third-Party Libraries
import joblib
import numpy as np
from nltk.corpus import stopwords
from loguru import logger
from termcolor import colored

# Local Modules
from log_retriever import LogRetriever
from database_manager import DatabaseManager
from log_monitor import LogMonitor
from task_scheduler import TaskScheduler
from log_parser import LogParser
from model_manager import ModelManager
from model_manager_factory import ModelManagerFactory
from ensemble_model import EnsembleModel

# Additional Configuration
logging.getLogger('nltk').setLevel(logging.CRITICAL)

def load_config():
    config = configparser.ConfigParser()
    config.read('drain3.ini')
    return config

config = load_config()
#Setup color logging
# Customize the logger format
# Create a logger for your module
import logging
from termcolor import colored

class ColoredFormatter(logging.Formatter):
    LEVEL_COLORS = {
        'INFO': 'white',
        'DEBUG': 'yellow',
        'WARNING': 'magenta',
        'ERROR': 'red',
        'CRITICAL': 'red',
    }

    LINE_COLORS = {
        'INFO': 'white',
        'DEBUG': 'white',
        'WARNING': 'magenta',
        'ERROR': 'magenta',
        'CRITICAL': 'magenta',
    }

    FILENAME_COLOR = 'yellow'

    def format(self, record):
        log_message = super().format(record)
        line_color = self.LINE_COLORS.get(record.levelname, 'magenta')
        colored_message = colored(log_message, line_color)
        
        levelname_color = self.LEVEL_COLORS.get(record.levelname, 'magenta')
        colored_levelname = colored(record.levelname, levelname_color, attrs=['bold'])
        
        colored_filename = colored(record.filename, self.FILENAME_COLOR, attrs=['bold'])
        
        colored_message = colored_message.replace(record.levelname, colored_levelname)
        return colored_message.replace(record.filename, colored_filename)

# Create a logger for your module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create a handler and formatter
handler = logging.StreamHandler()
formatter = ColoredFormatter('%(asctime)s - [%(levelname)s] - [%(filename)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(handler)

class AnomalyHandler:
    def handle_anomaly(self, anomaly):
    # Handle the anomaly (e.g., send an alert, log it)
        pass

def parse_arguments():
    parser = argparse.ArgumentParser(description='Log processing and anomaly detection.')
    # Cloudwatch
    # parser.add_argument('--log_group', required=True, help='Log group name.')
    # parser.add_argument('--log_stream', required=True, help='Log stream name.')
    # parser.add_argument('--start_time', required=True, help='Start time for log retrieval.')
    # parser.add_argument('--end_time', required=True, help='End time for log retrieval.')
    parser.add_argument('--log-dir', required=True, help='Directory for filesystem logs.')
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
    polling_interval=config.get('GENERAL', 'POLLING_INTERVAL_MINUTES')
    # Traverse through the log directory
    for root, _, files in os.walk(log_dir):
        file_paths = [os.path.join(root, filename) for filename in files]

        # Calculate hash values and map them to filepaths
        hash_values = log_retriever.parallel_hash_files(file_paths)
        for filepath, hash_value in zip(file_paths, hash_values):
            #print(f"Inspecting { filepath } ")
            if db_manager.should_process_file(filepath, hash_value, threshold_minutes=polling_interval) is True:
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
    logger.info(f"[{colored({filepath},'yellow')}] ---> Starting work on chunk {colored(hash_value, 'blue')}")

    #factory = ModelManagerFactory()
    #manager = factory.get_manager(filepath, log_retriever, log_parser, database_manager_instance) #Create a separate instance of model manager per file
    #model_manager = manager # use the model manager for that specific file

    raw_log, log_file_id = log_retriever.retrieve_from_filesystem(filepath, hash_value)
    #print(f"[colored({filepath},'yellow')] ---> Marking chunk id: {log_file_id} as in process")
    model_manager_instance = ModelManager(log_retriever, log_parser, str(filepath), config)
    # Start processing the file
    database_manager_instance.start_processing(log_file_id)

    if len(raw_log) > 1:
        logger.info(f"[{colored(filepath,'yellow')}] [{colored('Individual model','magenta')}] ---> Pre-procesing {colored(len(raw_log), 'yellow')} raw log lines...")
        logger.info(f"[{colored(filepath,'yellow')}] [{colored('Individual model','magenta')}] ---> Starting work on chunk {colored(hash_value, 'blue')}")


        #parsed_logs = [log_parser.parse_log_lines(line) for line in raw_log]
        logger.info(f"[{colored(filepath,'yellow')}] [{colored('Individual model','magenta')}] ---> Processing {colored(len(raw_log), 'yellow')} log...")
        structured_logs = log_parser.parse_log_lines(filepath, raw_log)
        logger.info(f"[{colored(filepath,'yellow')}] [{colored('Individual model','magenta')}] ---> Storing cleaned logs: {colored(len(structured_logs), 'yellow')} lines")
        storage_result = database_manager_instance.store_logs_drain3(filepath, structured_logs)
        if storage_result:
            logger.info(f"[{colored(filepath,'yellow')}] [{colored('Individual model','magenta')}] ---> Extracting TF-IDF Features from {colored(len(structured_logs), 'yellow')} lines")
            model_manager_instance.extract_features(structured_logs)
            logger.info(f"[{colored(filepath,'yellow')}] [{colored('Individual model','magenta')}] ---> Completed extraction of TF-IDF features, {colored(len(model_manager_instance.features_dict), 'yellow')} items (features) found, array size: {colored(np.size(model_manager_instance.features_np_array), 'red')} shape: {colored(np.shape(model_manager_instance.features_np_array), 'red')}")
            #log_file_ids = log_retriever.get_all_ids()
            if len(model_manager_instance.features_dict) > 0:
                #Training model with features
                logger.info(f"[{colored(filepath,'yellow')}] [{colored('Individual model','magenta')}] ---> Training model with {colored(len(model_manager_instance.features_dict),'yellow')} features.")
                model_manager_instance.train_individual_model()
                
                #Detect anomalies
                logger.info(f"[{colored(filepath,'yellow')}] [{colored('Individual model','magenta')}] ---> Starting anomaly detection using {colored(len(structured_logs),'yellow')} cleaned log lines")
                result = model_manager_instance.detect_anomalies()
                if result is not None:
                    anomaly_features, anomaly_log_texts = result
                else:
                    anomaly_features, anomaly_log_texts = [], []

                model_manager_instance.display_anomalies()
                insert_anomaly_features_result = model_manager_instance.insert_anomaly_features(model_manager_instance.model_path, np.array(anomaly_features))
                #Store anomaly features
                if insert_anomaly_features_result:
                    logger.info(f"[{colored(filepath,'yellow')}] [{colored('Individual model','magenta')}] ---> Stored records (anomaly features).")
                else:
                    logger.error(f"[{colored(filepath,'yellow')}] [{colored('Individual model','magenta')}] ---> ERROR: Failed to store records (anomaly features).")

                insert_anomaly_log_texts_result = model_manager_instance.insert_anomaly_log_texts(model_manager_instance.model_path,np.array(anomaly_log_texts))
                if insert_anomaly_log_texts_result:
                    logger.info(f"[{colored(filepath,'yellow')}] [{colored('Individual model','magenta')}] ---> Stored records (anomaly features).")
                else:
                    logger.error(f"[{colored(filepath,'yellow')}] [{colored('Individual model','magenta')}] ---> ERROR: Failed to store records (anomaly log texts).")

                process_ensemble_model(model_manager_instance, log_retriever, log_parser, config)
                
            else:
                logger.warn(f"[{colored(filepath,'yellow')}] [{colored('Individual model','magenta')}] ---> No features to train the model.")
        else:
            print("Storage failed.")
    database_manager_instance.end_processing(log_file_id)
    local_connection.close()

def process_ensemble_model(model_manager_instance, log_retriever, log_parser, config):
    # Check if at least 2 individual models exist
    individual_model_paths = model_manager_instance.list_individual_model_paths()
    logger.info(f"[Ensemble model] --> Found {colored(len(individual_model_paths), 'yellow')} individual models: {colored(individual_model_paths, 'yellow')}")
    if len(individual_model_paths) >= 2:
        ensemble_model_instance = EnsembleModel(config,model_manager_instance, log_retriever, log_parser, individual_model_paths)
        # Creating a dictionary for individual models (contains joblib models)
        #individual_models = {model_paths[i]: joblib.load("models/" + model_paths[i]) for i in range(len(model_paths))}
        # Creating a ensemble model object
        #ensemble_model_obj = ensemble_model_instance.initialize_model()
        # Constructing the final dictionary structure
        #ensemble_model_w_models_dict = {'ensemble_model': ensemble_model_obj, 'individual_models': individual_models}
        #print(f"[Ensemble model] --> Ensemble model with individual models: {ensemble_model_w_models_dict}")
        # Combine individual models into ensemble model
        # {'ensemble_model': IsolationForest(contamination=0.01), 
        #   'individual_models': {
        #       'model_sample_input_logs_file-instance-k_log.pkl': IsolationForest(contamination=0.01),
        #       'model_sample_input_logs_file-instance-j_log.pkl': IsolationForest(contamination=0.01), 
        #       'model_sample_input_logs_file-instance-l_log.pkl': IsolationForest(contamination=0.01)
        #    }
        # }

        logger.info(f"[Ensemble model] --> saving ensemble model.")
        ensemble_model_instance.save_ensemble_model()  
        logger.info(f"[Ensemble model] --> generating predictions for anomalies based on individual models")
        logger.info(f"[Ensemble model] --> getting combined anomaly log texts list")
        # print(f"[Ensemble model] --> Ensemble Model Dict: {ensemble_model_instance.ensemble_model_dict}")                         
        combined_anomaly_log_texts_list = ensemble_model_instance.get_combined_anomaly_log_texts()
        raw_lines = []
        for line in combined_anomaly_log_texts_list.tolist():
                raw_lines.append(line)

        if len(combined_anomaly_log_texts_list.tolist()) > 0: 
            logger.info(f"[Ensemble model] --> combined anomaly log texts list length: ({len(combined_anomaly_log_texts_list.tolist())}) items")
            logger.info(f"[Ensemble model] --> parsing combined anomaly texts: ({len(raw_lines)}) items")
            structured_logs = log_parser.parse_log_lines("Ensemble Model", raw_lines)
            logger.info(f"[Ensemble model] --> structured Logs: ({len(structured_logs)})")
            if len(structured_logs) > 0:
                ensemble_model_instance.extract_features(structured_logs)
                logger.info(f"[Ensemble model] --> ensemble features: ({colored(len(ensemble_model_instance.features_dict),'yellow')} dictionary items, numpy array size: {colored(np.size(ensemble_model_instance.features_np_array), 'yellow')} shape: {colored(np.shape(ensemble_model_instance.features_np_array), 'yellow')}")
                ensemble_model = ensemble_model_instance.train_ensemble_model()
                logger.info(f"[Ensemble model] --> model trained.")
                logger.info(f"[Ensemble model] --> detecting anomalies")
                ensemble_model_instance.detect_anomalies(combined_anomaly_log_texts_list)
                logger.info(f"[Ensemble model] --> displaying anomalies")
                ensemble_model_instance.display_anomalies()
            else:
                print(f"[Ensemble model] --> no combined anomalies found, skipping ensemble model anomaly detection...")
        else:
            print(f"[Ensemble model] --> no anomalies found yet, skipping ensemble model anomaly detection...")
    else:
        print(f"[Ensemble model] --> Not enough individual models to combine.")
        print(f"[Ensemble model] --> At least 2 individual models must be loaded, but only {colored(len(individual_model_paths), 'magenta')} were found of {colored(type(individual_model_paths),'yellow')}, waiting for more models.")

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
    polling_interval = config.get('GENERAL', 'POLLING_INTERVAL_MINUTES')
    while True:
        file_paths_with_hash = get_files_to_process(log_dir, log_retriever, database_manager)
        #print(f"Found {len(file_paths_with_hash)} files to process.")
        
        # Process only the files that meet the conditions
        for (filepath, hash_value) in file_paths_with_hash:
            if database_manager.should_process_file(filepath, hash_value, threshold_minutes=polling_interval) is True:
                process_with_args(log_dir, (filepath, hash_value))
            else:
                logger.warn(f"{filepath} --> File is not ready to be processed.")
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