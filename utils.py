from enum import Enum


class LogLevel(Enum):
    ERROR = "\033[91m🔴 ERROR\033[0m"
    WARNING = "\033[93m⚠️ WARNING\033[0m"
    INFO = "\033[94m🟡 INFO\033[0m"
    DEBUG = "\033[34m🔵 DEBUG\033[0m"
    SUCCESS = "\033[32m🟢 SUCCESS\033[0m"


class Logger:
    def __init__(self, level=LogLevel.INFO):
        self.level = level

    def log(self, level, message, **kwargs):
        if level.value <= self.level.value:
            nl = kwargs.get('nl', False)
            if nl:
                print()
            print(f"{level.value}: {message}")

    def error(self, message, **kwargs):
        self.log(LogLevel.ERROR, message, **kwargs)

    def warning(self, message, **kwargs):
        self.log(LogLevel.WARNING, message, **kwargs)

    def info(self, message, **kwargs):
        self.log(LogLevel.INFO, message, **kwargs)

    def debug(self, message, **kwargs):
        self.log(LogLevel.DEBUG, message, **kwargs)

    def success(self, message, **kwargs):
        self.log(LogLevel.SUCCESS, message, **kwargs)
