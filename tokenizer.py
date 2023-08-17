class Tokenizer:
    def __init__(self) -> None:
        self.mapping = {}
        self.reverse_mapping = {v: k for k, v in self.mapping.items()}  # Create reverse mapping

    def load(self, string:str) -> None:
        for word in string.split():
            if word not in self.mapping:
                self.mapping[word] = len(self.mapping)
        self.reverse_mapping = {v: k for k, v in self.mapping.items()}  # Create reverse mapping

    def encode(self, string:str) -> list:
        tokenized = []
        for word in string.split():
            if word in self.mapping:
                tokenized.append(self.mapping[word])
            else:
                tokenized.append(0)
        return tokenized
    

    def decode(self, tokens: list) -> str:
        decoded_words = []
        for token in tokens:
            if token in self.reverse_mapping:
                decoded_words.append(self.reverse_mapping[token])
            else:
                decoded_words.append("UNKNOWN_TOKEN")
        decoded_string = ' '.join(decoded_words)
        return decoded_string

if __name__ == '__main__':
    tokenizer = Tokenizer()
    tokenizer.load('Hello this is a test no cap acutally pog')
    print(tokenizer.mapping)
    encoded = tokenizer.encode('Hello no cap')
    print(encoded)
    print(tokenizer.decode(encoded))