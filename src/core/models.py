# Abstract base classes or definitions for AI models

class BaseModel:
    def __init__(self, config=None):
        self.config = config

    def train(self, data):
        raise NotImplementedError

    def predict(self, data):
        raise NotImplementedError
