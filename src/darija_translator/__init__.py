import warnings

warnings.filterwarnings("ignore",
                        message=".*is_flash_linear_attention_available.*")
import logging

logging.getLogger("transformers").setLevel(logging.ERROR)
