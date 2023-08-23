from apscheduler.schedulers.background import BackgroundScheduler

class TaskScheduler:
    def __init__(self):
        self.scheduler = BackgroundScheduler()
        self.scheduler.start()
        self.task_running = False
        self.interval_seconds = 60  # number of seconds to poll for new log entries


    def schedule_task(self, task, interval_seconds=60):
        def wrapper():
            if self.task_running:
                #print("Scheduled Job is already running. Skipping this run.")
                return
            self.task_running = True
            try:
                task()
            except Exception as e:
                print(f"An error occurred while executing task: {e}")
            finally:
                self.task_running = False

        # Schedule a task to be executed every `interval_seconds` seconds
        self.scheduler.add_job(wrapper, 'interval', seconds=self.interval_seconds)

    def execute_task(self, task):
        # Execute a task immediately
        task()

    def shutdown(self):
        # Shutdown the scheduler if needed
        self.scheduler.shutdown()
