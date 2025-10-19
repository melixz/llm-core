from typing import List
import dill


class BPE:
    """
    Byte-Pair Encoding (BPE) это метод сжатия текстовых данных, который приспособили для проведения токенизации.
    Основная идея BPE заключается в итеративном объединении наиболее часто встречающихся пар символов и формировании из
    них токенов, которые в последующем используются для токенизации.
    """

    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.id2token = {}
        self.token2id = {}

    def fit(self, text: str):
        """
        Обучает BPE словарь на корпусе текстовых данных.
        """
        unique_tokens = sorted(set(text))
        base_tokens = list(text)

        while len(unique_tokens) < self.vocab_size:
            pairs = {}
            for i in range(len(base_tokens) - 1):
                pair = (base_tokens[i], base_tokens[i + 1])
                if pair in pairs:
                    pairs[pair] += 1
                else:
                    pairs[pair] = 1

            if not pairs:
                break

            most_frequent_pair = max(pairs, key=pairs.get)

            new_token = "".join(most_frequent_pair)
            unique_tokens.append(new_token)

            new_base_tokens = []
            i = 0
            while i < len(base_tokens):
                if i < len(base_tokens) - 1 and (base_tokens[i], base_tokens[i + 1]) == most_frequent_pair:
                    new_base_tokens.append(new_token)
                    i += 2
                else:
                    new_base_tokens.append(base_tokens[i])
                    i += 1

            base_tokens = new_base_tokens

        self.id2token = {i: token for i, token in enumerate(unique_tokens[: self.vocab_size])}
        self.token2id = {token: i for i, token in self.id2token.items()}

    def encode(self, text: str):
        """
        Кодирует текст с использованием построенного словаря BPE.
        """
        tokens = list(text)
        bpe_tokens = sorted(self.token2id.keys(), key=lambda x: -len(x))

        i = 0
        encoded = []
        while i < len(tokens):
            match = None
            match_len = 0
            for token in bpe_tokens:
                t_len = len(token)
                if t_len <= (len(tokens) - i):
                    if "".join(tokens[i : i + t_len]) == token:
                        match = token
                        match_len = t_len
                        break
            if match:
                encoded.append(self.token2id[match])
                i += match_len
            else:
                encoded.append(self.token2id.get(tokens[i], 0))
                i += 1
        return encoded

    def decode(self, token_ids: List[int]):
        """
        Заменяет полученные идентификаторы токенов на их текстовые представления.
        """
        tokens = []
        for token_id in token_ids:
            if token_id in self.id2token:
                tokens.append(self.id2token[token_id])
            else:
                tokens.append("")
        return "".join(tokens)

    def save(self, filename):
        """
        Сохраняет экземпляр класса BPE в файл с помощью dill.
        """
        with open(filename, "wb") as f:
            dill.dump(self, f)
        print(f"Объект сохранён в {filename}")

    @classmethod
    def load(cls, filename):
        """
        Загружает экземпляр класса BPE из файла с помощью dill.
        """
        with open(filename, "rb") as f:
            obj = dill.load(f)

        print(f"Объект загружен из {filename}")
        return obj
