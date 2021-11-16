
def update_dictionary(obj, key, value):
    if isinstance(value, dict):
        if not hasattr(obj.get(key), "__setitem__"):
            obj[key] = {}
        for k, v in value.items():
            update_dictionary(obj[key], k, v)
    else:
        obj[key] = value
