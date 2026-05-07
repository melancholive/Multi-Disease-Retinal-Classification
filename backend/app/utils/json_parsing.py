import re
import json


def extract_json(text):
    if isinstance(text, dict):
        return text
    if not isinstance(text, str):
        text = str(text)

    text = text.strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)

    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass

    return {
        "raw_output": text,
        "parse_warning": "Model did not return valid JSON.",
    }
