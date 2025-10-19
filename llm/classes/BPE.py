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
