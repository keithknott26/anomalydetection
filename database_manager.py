import sqlite3
from threading import Lock
from datetime import datetime

class DatabaseManager:
    def __init__(self):
        # prevent race conditions since this is a shared resource        
        self.lock = Lock()
        #set the SQLite DB Path
        self.db_path = 'logs.db'
        #Create logs table
        self.create_table()

    def get_connection(self):
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def create_table(self):
        with self.lock:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        app_name TEXT,
                        filepath TEXT UNIQUE,
                        current_hash TEXT,
                        current_hash_timestamp DATETIME,
                        last_read_position INTEGER,
                        in_process INTEGER DEFAULT 0,
                        last_checked_timestamp DATETIME
                    );
                ''')
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS model_associations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        log_file_id INT,
                        log_filepath TEXT UNIQUE,
                        model_filename VARCHAR(255),
                        in_process INTEGER DEFAULT 0
                    );
                ''')
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS logs_storage (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        filepath TEXT,
                        template TEXT,
                        parameters TEXT,
                        FOREIGN KEY(filepath) REFERENCES logs(filepath)
                    );
                ''')
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS anomaly_log_texts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    log_text TEXT NOT NULL,
                    is_anomaly INTEGER DEFAULT 1
                )
                ''')
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS anomaly_features (
                    model_name TEXT NOT NULL UNIQUE,
                    feature TEXT NOT NULL
                )
                ''')

    def start_processing(self, log_file_id):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE logs SET in_process = 1 WHERE id = ?", (log_file_id,))
            conn.commit()

    def end_processing(self, log_file_id):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE logs SET in_process = 0 WHERE id = ?", (log_file_id,))
            conn.commit()

    def is_file_in_process(self, filepath):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            result = cursor.execute("SELECT in_process FROM logs WHERE filepath = ?", (filepath,)).fetchone()
            return result[0] if result else False
    
    def should_process_file(self, filepath, hash_value, threshold_minutes=1):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            query = '''
                SELECT last_checked_timestamp, in_process
                FROM logs
                WHERE filepath = ? AND current_hash = ?
            '''
            cursor.execute(query, (filepath, hash_value))
            result = cursor.fetchone()
            if result:
                last_checked_timestamp, in_process = result
                if last_checked_timestamp and in_process == 0:
                    last_checked_datetime = datetime.strptime(last_checked_timestamp, '%Y-%m-%d %H:%M:%S')
                    delta = datetime.now() - last_checked_datetime
                    if delta.total_seconds() > int(threshold_minutes) * 60:
                        return True
            else:
                return True  # Return True if no record was found
            return False

    def store_logs_drain3(self, filepath, structured_logs):
        return True
        with self.lock:
            log_file_id = self.get_log_id_from_filepath(filepath)
            print(f"Storing parsed log (drain3) in DB for Logfile ID: {log_file_id}, filepath: {filepath}")
            try:
                with self.get_connection() as conn:
                    cursor = conn.cursor()
                    for log in structured_logs:
                        template = log['template']
                        parameters = ",".join(log['parameters'])
                        cursor.execute("INSERT INTO logs_storage (filepath, template, parameters) VALUES (?, ?, ?)", (filepath, template, parameters))
                conn.commit()
                return True
            except Exception as e:
                print(f"An error occurred while storing logs: {e}")
                return False

    def get_logs_since(self, last_known_log_id):
        with self.lock:
            if last_known_log_id is None:
                last_known_log_id = 0
            with self.get_connection() as conn:
                cursor = conn.cursor()
                return cursor.execute("""
                SELECT * FROM logs WHERE id > ?
                """, (last_known_log_id,)).fetchall()
        
    def get_log_entry(self, log_id):
        with self.lock:
            # Retrieve a log entry from the database
            with self.get_connection() as conn:
                cursor = conn.cursor()
                return cursor.execute("""
                SELECT * FROM logs WHERE id = ?
                """, (log_id,)).fetchone()

    def update_log_entry(self, log_id, log_details):
        with self.lock:
        # Update a log entry in the database
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE logs
                    SET app_name = ?,
                        filepath = ?,
                        current_hash = ?,
                        current_hash_timestamp = ?,
                        last_read_position = ?
                    WHERE id = ?""",
                    (log_details[1], log_details[2], log_details[3], log_details[4], log_details[5], log_id))

    def delete_log_entry(self, log_id):
        with self.lock:
            # Delete a log entry from the database
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                DELETE FROM logs WHERE id = ?
                """, (log_id,))
    
    def get_filepath_from_logs(self, log_file_id):
        with self.lock:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                result = cursor.execute("""
                SELECT filepath
                FROM logs
                WHERE id = ?
                """, (log_file_id,)).fetchone()
                return result[0] if result else None

    def get_log_id_from_filepath(self, filepath):
        print(f"Type of filepath: {type(filepath)}, Value: {filepath}")  # Debugging print statement
        with self.get_connection() as conn:
            cursor = conn.cursor()
            result = cursor.execute("""
            SELECT id
            FROM logs
            WHERE filepath = ?
            """, (filepath,)).fetchone()
            return result[0] if result else None
        
    def get_model_filename(self, log_id):
        with self.lock:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                result = cursor.execute("""
                SELECT ma.model_filename
                FROM model_associations AS ma
                INNER JOIN logs AS l
                ON ma.log_filepath = l.filepath
                WHERE l.id = ?
                """, (log_id,)).fetchone()
                return result[0] if result else None
        
    def get_model_filename_from_log_filepath(self, log_filepath):
        with self.lock:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                result = cursor.execute("""
                SELECT ma.model_filename
                FROM model_associations AS ma
                INNER JOIN logs AS l
                ON ma.log_filepath = l.filepath
                WHERE ma.log_filepath = ?
                """, (log_filepath,)).fetchone()
                return result[0] if result else None

    def get_model_log_filepath(self, log_file_id):
        with self.lock:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                result = cursor.execute("""
                SELECT l.filepath
                FROM logs AS l
                WHERE l.id = ?
                """, (log_file_id,)).fetchone()

                if result:
                    filepath = result[0]
                    result = cursor.execute("""
                    SELECT log_filepath
                    FROM model_associations
                    WHERE log_filepath = ?
                    """, (filepath,)).fetchone()
                    return result[0] if result else None
                else:
                    return None

    def get_model_log_file_id(self, filepath):
        with self.lock:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                result = cursor.execute("""
                SELECT log_file_id FROM model_associations WHERE log_filepath = ?
                """, (filepath,)).fetchone()
                return result[0] if result else None

    def set_model_filename(self, log_file_id, filepath, model_filename):
        with self.lock:
            #print(f"{filepath} --> Inserting log_file_id: {log_file_id}, filepath: {filepath}, model_filename: {model_filename} into model_associations")
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO model_associations (log_file_id, log_filepath, model_filename)
                    VALUES (?, ?, ?)
                """, (log_file_id, filepath, model_filename))

    def insert_anomaly_log_text(self, model_name, log_text):
        with self.lock:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                try:
                    cursor.execute('INSERT INTO anomaly_log_texts (model_name, log_text) VALUES (?, ?)', (model_name, log_text))
                    conn.commit()
                    return True
                except sqlite3.Error as e:
                    print(f"Database error: {e}")
                    return False

    def update_anomaly_status(self, log_text_id, is_anomaly, model_nameh):
        with self.lock:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('UPDATE anomaly_log_texts SET is_anomaly = ?, feedback = 1 WHERE id = ? AND model_name = ?', (is_anomaly, log_text_id, model_name))
                conn.commit()

    def fetch_anomaly_data(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT log_text FROM anomaly_log_texts WHERE is_anomaly = 1')
            return [row[0] for row in cursor.fetchall()]

    def get_anomaly_log_texts(self, model_name):
        with self.lock:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT log_text FROM anomaly_log_texts WHERE model_name = ?', (model_name,))
                log_texts = [row[0] for row in cursor.fetchall()]
                return log_texts

    def get_feedback_count(self, model_name):
        with self.conn:
            cursor = self.conn.execute("SELECT COUNT(*) FROM anomaly_log_texts WHERE model_name = ? AND feedback = 1", (model_name,))
            return cursor.fetchone()[0]

    def get_anomaly_features(self, model_name):
        with self.lock:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT feature FROM anomaly_features WHERE model_name = ?', (model_name,))
                features = [row[0] for row in cursor.fetchall()]
                return features

    def get_all_unique_model_names(self):
        with self.lock:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT DISTINCT model_filename FROM model_associations")
                model_names = [row[0] for row in cursor.fetchall()]
                return model_names

    def update_model_version(self, new_version):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('UPDATE model_metadata SET version = ?', (new_version,))
            conn.commit()
