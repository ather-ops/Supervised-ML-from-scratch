import string
import re


def clean_text(text):
    if text is None:
        return ""

    text = str(text).lower()
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    return re.sub(r"\s+", " ", text).strip()
