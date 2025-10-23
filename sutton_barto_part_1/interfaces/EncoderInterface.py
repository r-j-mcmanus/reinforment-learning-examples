


class EncoderInterface:
    def encode(self, *args, **kwargs) -> int:
        raise NotImplementedError

    def decode(self, state: int) -> tuple[int, int]:
        raise NotImplementedError