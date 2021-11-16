
def update_dictionary(obj, key, value):
    if isinstance(value, dict):
        if key not in obj:
            obj[key] = {}
        for k, v in value.items():
            update_dictionary(obj[key], k, v)
    obj[key] = value
