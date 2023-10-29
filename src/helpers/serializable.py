
from jsonpickle import encode, decode


class Serializable:
    def to_json(self):
        return encode(self)

    def to_file(self, path):
        with open(path, 'wt') as f:
            f.write(self.to_json())

    @classmethod
    def from_json(cls, json):
        obj = decode(json)

        if not isinstance(obj, cls):
            raise RuntimeError(f"Expected type {cls}, got {type(obj)}")

    @classmethod
    def from_file(cls, path):
        with open(path, 'rt') as f:
            return cls.from_json(f.read())

