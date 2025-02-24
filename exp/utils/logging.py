import logging

RESET = "\033[0m"

# Level-based colors
LEVEL_COLORS = {
    10: "\033[34m",   # DEBUG    -> Blue
    20: "\033[32m",   # INFO     -> Green
    30: "\033[33m",   # WARNING  -> Yellow
    40: "\033[31m",   # ERROR    -> Red
    50: "\033[41m",   # CRITICAL -> Red Background
}

# Other parts
DATE_COLOR = "\033[36m"  # Cyan for date/time
NAME_COLOR = "\033[35m"  # Magenta for logger name

class ColorFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None, style='%'):
        super().__init__(fmt=fmt, datefmt=datefmt, style=style)
    
    def format(self, record):
        # Get the original message from the record
        message = record.getMessage()
        
        # Pick a color based on level (DEBUG=10, INFO=20, etc.)
        level_color = LEVEL_COLORS.get(record.levelno, RESET)
        
        # Format the date/time using the parent's logic
        asctime_str = self.formatTime(record, self.datefmt)
        
        # Build a single, final log string with colors applied
        log_line = (
            f"{DATE_COLOR}{asctime_str}{RESET} "        # Date/time in cyan
            f"| {level_color}{record.levelname}{RESET} " # Level name in color
            f"| {NAME_COLOR}{record.name}{RESET} "       # Logger name in magenta
            f"| {message}"
        )
        
        # If there's exception info (e.g., traceback), include it
        if record.exc_info:
            # Format the exception text via parent’s method
            exception_text = super().formatException(record.exc_info)
            log_line += f"\n{exception_text}"
        
        return log_line

def get_logger(name, file_path, level=logging.DEBUG):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    color_formatter = ColorFormatter(datefmt="%Y-%m-%d %H:%M:%S")
    console_handler.setFormatter(color_formatter)

    file_handler = logging.FileHandler(file_path)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger