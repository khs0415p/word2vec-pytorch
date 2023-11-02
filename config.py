import json 

class Config:
    def __init__(self, path) -> None:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.__dict__.update(data)


if __name__ == "__main__":
    config = Config('config.json')
    print(config.emb_size)