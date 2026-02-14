from pydantic import BaseModel


class SastdpExecutionArgs(BaseModel):
    data_dir: str = "./datasets/i2b2/"
    word_embedding_path: str = "embeddings/english/glove.840B.300d.txt"
    word_embedding_size: int = 300
    method: str = "normal"
    task: str = "i2b2"
    seed: int = 42
    epsilon: float = 16
    s_epsilon: float = 8
    p: float = 0.7
    threads: int = 2

    @property
    def replacements_output_dir(self) -> str:
        return f"replacements/{self.method}/{self.task}/n_epsilon_{self.epsilon}_s_epsilon_{self.s_epsilon}"


class SastdpDocument(BaseModel):
    text: str
    text_id: int | str
