from model_manager import ModelManager

class ModelManagerFactory:
    def __init__(self):
        self.managers = {}

    def get_manager(self, file_identifier, log_retriever, log_parser, database_manager):
        if file_identifier not in self.managers:
            self.managers[file_identifier] = ModelManager(log_retriever, log_parser, database_manager)
        return self.managers[file_identifier]