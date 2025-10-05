import os
import logging
import builtins

def setup_logging(server_name, log_file_path):
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    debug_log_file = open(log_file_path, "w", encoding="utf-8")

    # Redirigir print a archivo de log
    _orig_print = builtins.print
    def _print_to_logfile(*args, **kwargs):
        kwargs.setdefault("file", debug_log_file)
        return _orig_print(*args, **kwargs)
    builtins.print = _print_to_logfile

    # Configurar logging
    logger = logging.getLogger(server_name)
    logger.setLevel(logging.WARNING)
    logger.handlers.clear()
    file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(file_handler)

    return logger, debug_log_file