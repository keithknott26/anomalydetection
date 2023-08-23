# Anomaly Detection on Raw Logs Using Ensemble Machine Learning Models (IsolationForest)

## Overview

This project aims to perform anomaly detection on raw logs using ensemble machine learning models, specifically IsolationForest. The following steps outline the process:

1. **Input Data**: Input data can be a root directory of static logs or live logs (logs being written to currently).
2. **Database Record Creation**: The program will walk the directory and find logs to monitor, creating a database record for each one.
3. **Log Feeding**: Logs are fed into the program by them being present in the specific root directory which is being polled for changes.
4. **Log Monitoring**: Similar to tail, each log file's attributes (size, modified date, created date, etc) are continually hashed and compared against the database to detect log changes. This approach allows us to monitor a large number of logs, and process them in chunks (instead of opening file watchers and eating up resources).
5. **File Polling Interval**: The file polling interval is configurable depending on how frequently you'd like to poll for changes to the logs. Each log is checked for changes per poll.
6. **Log Change Handling**: When a log change is found, a chunk of the log is sent to the log parser.
7. **Log Cleanup and Parsing**: Logs are cleaned up and parsed, removing any empty lines and duplicates which may be contained within that chunk.
8. **Individual Model Creation**: One model is created for each log to train the machine learning model with data specific to that application's logging profile.
9. **Master Model**: Once the individual models are trained on the log data, you can enable the master model which polls the individual models for anomalies and performs anomaly detection at a bird's eye level, watching over the individual models. This gives a reasonably accurate view of the anomalies detected across the entire logging root directory or your application suite.
10. **Future Integration**: This is a POC and will likely include integration with a timeseries database to better visualize and tag the anomalies as they come into the monitoring system (similar to Splunk).

## Running:
```
python3 main.py --log_dir sample_input_logs
```
## Requirements

### PostgreSQL (required by Drain logparser)

```
bash
brew install postgresql
pip3 install psycopg2-binary boto3 scikit-learn logparser drain3
export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
```

## Core Classes and Functions

Let's start by defining the core classes, functions, and methods that will be necessary for this project:

1. **`LogRetriever`**: This class will be responsible for retrieving logs from different sources. It will have methods like `retrieve_from_cloudwatch` and `retrieve_from_filesystem`.
2. **`DatabaseManager`**: This class will handle all database operations. It will have methods like `store_log_entry`, `get_log_entry`, `update_log_entry`, and `delete_log_entry`.
3. **`LogMonitor`**: This class will monitor the logs for changes. It will have methods like `check_for_changes` and `handle_log_change`.
4. **`TaskScheduler`**: This class will handle the scheduling of tasks. It will have methods like `schedule_task` and `execute_task`.
5. **`LogParser`**: This class will parse the log lines. It will have methods like `parse_log_line`.
6. **`ModelManager`**: This class will handle all operations related to the model. It will have methods like `feed_log_line`, `extract_features`, `train_model`, `detect_anomalies`, and `update_model`.

