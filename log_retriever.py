#import boto3
import time
import os
import hashlib
import sqlite3
import subprocess
import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from database_manager import DatabaseManager


class LogRetriever:
    def __init__(self):
        # Initialize Cloudwatch client
        #self.cloudwatch_client = boto3.client('logs')
        # Log group and stream name can be customized as needed
        # self.log_group = log_group
        # self.log_stream = log_stream
        self.log_type = "system"
        self.lock = Lock()
        # Initialize SQLite database
        self.db_connection = sqlite3.connect('logs.db', check_same_thread=False)
        self.db_connection.row_factory = sqlite3.Row
        self.database_manager = DatabaseManager()
        # Initialize the task scheduler
        self.scheduler = BackgroundScheduler()
        self.scheduler.start()

    def start(self):
        self.scheduler.start()

    def stop(self):
        self.scheduler.shutdown()  # Shut down the scheduler when done
        self.db_connection.close()

    def retrieve_from_database(self, filepath):
        with self.lock:
            #print(f"[{filepath}] ---> Retrieving data from database...")
            cursor = self.db_connection.cursor()
            cursor.execute("SELECT * FROM logs WHERE filepath=?", (filepath,))
            result = cursor.fetchone()
            #print(f"Result from database: {result}")
            return result

    def insert_new_file(self, filepath, current_hash, current_timestamp, new_read_position):
        with self.lock:
            print(f"[{filepath}] ---> Inserting new chunk with hash: {current_hash}")
            cursor = self.db_connection.cursor()
            cursor.execute(
                "INSERT INTO logs (filepath, current_hash, current_hash_timestamp, last_read_position) VALUES (?, ?, ?, ?)",
                (filepath, current_hash, current_timestamp, new_read_position))
            self.db_connection.commit()

    #deprecated
    def update_existing_file(self, filepath, new_hash, new_timestamp, last_read_position):
        with self.lock:
            cursor = self.db_connection.cursor()
            cursor.execute(
                "UPDATE logs SET current_hash=?, current_hash_timestamp=?, last_read_position=? WHERE filepath=?",
                (new_hash, new_timestamp, last_read_position, filepath))
            self.db_connection.commit()

    def update_file_last_checked_timestamp(self, filepath):
        current_timestamp_unix = int(time.time())
        current_timestamp_sqlite = datetime.datetime.fromtimestamp(current_timestamp_unix).strftime('%Y-%m-%d %H:%M:%S')
        with self.lock:
            cursor = self.db_connection.cursor()
            cursor.execute(
                "UPDATE logs SET last_checked_timestamp=? WHERE filepath=?",
                (current_timestamp_sqlite, filepath))
            self.db_connection.commit()

    def update_file_read_position_and_hash(self, filepath, current_hash, current_timestamp, last_read_position):
        with self.lock:
            cursor = self.db_connection.cursor()
            cursor.execute(
                "UPDATE logs SET current_hash=?, current_hash_timestamp=?, last_read_position=? WHERE filepath=?",
                (current_hash, current_timestamp, last_read_position, filepath))
            self.db_connection.commit()

    def is_file_in_process(self, filepath):
        log_id = self.database_manager.get_log_id_from_filepath(filepath)
        if log_id is not None:
            # Assuming there's a method in database_manager to check if a file is in process
            return self.database_manager.is_file_in_process(log_id)
        else:
            return False

    
    def get_all_ids_with_filepath(self):
        with self.lock:
            #returns a ids[filepath] mapping
            try:
                cursor = self.db_connection.cursor()
                query = "SELECT id, filepath FROM logs;"
                cursor.execute(query)
                result = cursor.fetchall()
                ids = {item[1]: item[0] for item in result}
                return ids
            except Exception as e:
                print("Error retrieving log IDs:", str(e))
                return {}
    
    def get_all_ids(self):
        with self.lock:
            try:
                cursor = self.db_connection.cursor()
                query = "SELECT id FROM logs;"
                cursor.execute(query)
                result = cursor.fetchall()
                ids = [item[0] for item in result]
                return ids
            except Exception as e:
                print("Error retrieving log IDs:", str(e))
                return []
    
    def get_id_from_hash(self, hash_value):
        with self.lock:
            try:
                cursor = self.db_connection.cursor()
                query = "SELECT id FROM logs WHERE current_hash = ? LIMIT 1;"
                cursor.execute(query, (hash_value,))
                result = cursor.fetchone()
                return result[0] if result else None
            except Exception as e:
                print("Error retrieving id from chunk:", str(e))
                return None
    
    def get_id_from_filepath(self, filepath):
        with self.lock:
            try:
                cursor = self.db_connection.cursor()
                query = "SELECT id FROM logs WHERE filepath = ? LIMIT 1;"
                cursor.execute(query, (filepath,))
                result = cursor.fetchone()
                return result[0] if result else None
            except Exception as e:
                print("Error retrieving id from file_path:", str(e))
                return None
        
    def get_filepath_from_hash(self, hash_value):
        with self.lock:
            try:
                cursor = self.db_connection.cursor()
                query = "SELECT filepath FROM logs WHERE current_hash = ? LIMIT 1;"
                cursor.execute(query, (hash_value,))
                result = cursor.fetchone()
                return result[0] if result else None
            except Exception as e:
                print("Error retrieving filepath from hash:", str(e))
                return None
    
    def get_hash_from_filepath(self, filepath):
        with self.lock:
            try:
                cursor = self.db_connection.cursor()
                query = "SELECT current_hash FROM logs WHERE filepath = ? LIMIT 1;"
                cursor.execute(query, (filepath,))
                result = cursor.fetchone()
                return result[0] if result else None
            except Exception as e:
                print("Error retrieving hash from filepath:", str(e))
                return None

    def hash_file_attributes(self, file_path):
        with self.lock:
            file_info = os.stat(file_path)
            file_attributes = (file_path, file_info.st_size, file_info.st_ctime, file_info.st_mtime)
            hash_input = ''.join(str(attr) for attr in file_attributes)
            return hashlib.md5(hash_input.encode()).hexdigest()

    def parallel_hash_files(self, files):
        with ThreadPoolExecutor() as executor:
            hashes = list(executor.map(self.hash_file_attributes, files))
        return hashes
    
    def is_filepath_in_db(self, filepath):
        with self.lock:
            try:
                cursor = self.db_connection.cursor()
                query = "SELECT id FROM logs WHERE filepath = ? LIMIT 1;"
                cursor.execute(query, (filepath,))
                result = cursor.fetchone()
                if result is not None:
                    return True
                else:
                    return False
            except Exception as e:
                print("Error checking filepath:", str(e))
                return False
    
    def is_hash_in_db(self, hash_value):
        with self.lock:
            try:
                cursor = self.db_connection.cursor()
                query = "SELECT 1 FROM logs WHERE current_hash = ? LIMIT 1;"
                cursor.execute(query, (hash_value,))
                result = cursor.fetchone()
                if result is not None:
                    return True
                else:
                    return False
            except Exception as e:
                print("Error checking hash in logs:", str(e))
                return False
        
    def process_filesystem_file(self,filepath, hash_value):
        log_content = []
        new_read_position = 0  # Initialize new_read_position here
        print(f"[{filepath}] ---> Analyzing chunk: {hash_value}")  # Debugging
        hash_value = self.get_hash_from_filepath(filepath)

        current_timestamp_unix = int(time.time())
        current_timestamp_sqlite = datetime.datetime.fromtimestamp(current_timestamp_unix).strftime('%Y-%m-%d %H:%M:%S')
    
        #If file has been seen before, retrieve the details about it
        if self.is_filepath_in_db(filepath):

            print(f"[{filepath}] ---> Previously worked file, retrieving data for chunk:", hash_value) # Debugging
            try:
                result = self.retrieve_from_database(filepath)
            except Exception as e:
                print(f"[{filepath}] ---> Error in retrieve_from_database:", str(e))
                return None
            #print("Result from database:", dict(result))
            #If DB Lookup was successful, map results to variables
            if result is not None:
                log_file_id, app_name, filepath, current_hash, current_timestamp, last_read_position, in_process, last_processed_timestamp = result
                last_read_position = int(last_read_position) if last_read_position else 0
            #Open the file at the last read position (as defined by the database row, and set a new read position)
            if os.path.isfile(filepath):
                with open(filepath, 'r') as file:
                    file.seek(last_read_position)
                    content = file.read()
                    ################################
                    # print file content
                    # print("Content:", content)  # Debugging
                    log_content.extend(content.split('\n'))
                    new_read_position = file.tell()            
            else:
                print(f"[{filepath}] ---> Skipping directory")

            
            print(f"[{filepath}] ---> Read position: last: {last_read_position}, new: {new_read_position}")
            
            #Has the file changed?
            if last_read_position == new_read_position:
                print(f"[{filepath}] ---> File read position unchanged, skipping...")
            elif hash_value == current_hash and last_read_position < new_read_position:
                print(f"[{filepath}] ---> File has changed, updating database...")
                self.update_file_read_position_and_hash(filepath, hash_value, current_timestamp_sqlite, new_read_position)
            elif last_read_position < new_read_position:
                print(f"[{filepath}] ---> File has changed, updating database...")
                self.update_file_read_position_and_hash(filepath, hash_value, current_timestamp_sqlite, new_read_position)
            elif hash_value == current_hash:
                print(f"File chunk unchanged, skipping...")
            else:
                self.update_file_read_position_and_hash(filepath, hash_value, current_timestamp_sqlite, new_read_position)
        else:
            #Insert a new record into the logs table, since this file hasn't been seen before
            read_position = 0
            hash_value = self.hash_file_attributes(filepath)
            if os.path.isfile(filepath):
                with open(filepath, 'r') as file:
                    file.seek(read_position)
                    content = file.read()
                    ################################
                    # print file content
                    # print("Content:", content)  # Debugging
                    log_content.extend(content.split('\n'))
                    read_position = file.tell()
            else:
                print(f"[{filepath}] ---> Skipping directory")

            self.insert_new_file(filepath, hash_value, current_timestamp_sqlite, read_position)
            log_file_id = self.get_id_from_filepath(filepath)
            print(f"[{filepath}] ---> Inserted with chunk ID:  {log_file_id}")
        return log_content, log_file_id
    
    def retrieve_from_filesystem(self, filepath, hash_value):
        log_content, log_file_id = self.process_filesystem_file(filepath, hash_value)
        return log_content, log_file_id

    def retrieve_from_cloudwatch(self, start_time, end_time):
        # Implement the code to retrieve logs from Cloudwatch
        # You may use pagination to handle large log volumes
        """
        Retrieves logs from AWS CloudWatch.
        :param start_time: Start time of the logs to be retrieved.
        :param end_time: End time of the logs to be retrieved.
        :return: A list of log events.
        """
        response = self.cloudwatch_client.get_log_events(
            logGroupName=self.log_group,
            logStreamName=self.log_stream,
            startTime=start_time,
            endTime=end_time
        )
        return response['events']

        # TODO: enable pagination

    def schedule_file_check(self, interval):
        # Schedule periodic file check using the task scheduler
        self.start()
        self.scheduler.add_job(self.retrieve_from_filesystem, 'interval', seconds=interval, args=['./logs/'])
        self.stop()

