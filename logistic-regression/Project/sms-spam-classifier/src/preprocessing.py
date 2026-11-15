import re
import string

_TABLE = str.maketrans("", "", string.punctuation)

def clean_text(text):
    """Use identical text cleaning during training and inference."""
    if text is None:
        return ""
    text = str(text).lower().translate(_TABLE)
    return re.sub(r"\s+", " ", text).strip()
