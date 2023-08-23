class LogMonitor:
    def __init__(self, database_manager):
        self.database_manager = database_manager
        self.last_known_log_id = None
        self.is_running = False

    def check_for_changes(self):
        if self.is_running:
            print("Job is already running. Skipping this run.")
            return
        self.is_running = True
        # Here you might need to define how you identify new logs since the last check
        # This could be based on a timestamp, or other criteria suitable to your application
        new_logs = self.database_manager.get_logs_since(self.last_known_log_id)
        for log in new_logs:
            print("new log found")
            print(log)
            print("end new log found")
            self.handle_log_change(log[0]) # Assuming log_id is the first element in the tuple

        # Update the last known log ID with the latest log's ID
        if new_logs:
            self.last_known_log_id = new_logs[-1][0]
        self.is_running = False

    def handle_log_change(self, log_id):
        # Query the specific log details using the log ID
        log_details = self.database_manager.get_log_entry(log_id)
        print("log details")
        print(log_details)
        print("end log details")
        # Here you can process the log details, perhaps parsing them with the LogParser class
        # or applying anomaly detection algorithms
        # ...

        # You may want to update the log entry with new information, depending on your application's needs
        self.database_manager.update_log_entry(log_id, log_details)
