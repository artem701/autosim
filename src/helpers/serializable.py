
from jsonpickle import encode, decode


class Serializable:
    def to_json(self):
        return encode(self)

    @staticmethod
    def from_json(json):
        return decode(json)

    @classmethod
    def from_file(cls, path):
        obj = None
        with open(path, 'rt') as f:
            obj = decode(f.read())

        if not isinstance(obj, cls):
            raise RuntimeError(f"Expected type {cls}, got {type(obj)}")
