class GlobalSequence:
    def __init__(self):
        self.current_sequence = 0

    def generate_sequence(self) -> int:
        self.current_sequence += 1
        return self.current_sequence

    def reset_sequence(self):
        self.current_sequence = 0