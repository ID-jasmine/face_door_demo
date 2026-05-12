import numpy as np


class MatchResult:
    def __init__(self, name, score, authorized):
        self.name = name
        self.score = score
        self.authorized = authorized


class FaceDatabase:
    def __init__(self, embedding_path, name="me", threshold=0.50):
        self.embedding_path = embedding_path
        self.name = name
        self.threshold = threshold

        self.known_embedding = np.load(self.embedding_path)
        self.known_embedding = self._l2_normalize(self.known_embedding)

    def match(self, embedding):
        embedding = self._l2_normalize(embedding)

        score = self._cosine_similarity(embedding, self.known_embedding)

        if score >= self.threshold:
            return MatchResult(
                name=self.name,
                score=score,
                authorized=True
            )

        return MatchResult(
            name="unknown",
            score=score,
            authorized=False
        )

    # @staticmethod: 表示该方法是静态方法，不依赖于类实例（self）或类本身（cls）。
    # 这里用于将两个向量作为独立函数计算余弦相似度。
    @staticmethod
    def _cosine_similarity(a, b):
        a = a.flatten()
        b = b.flatten()

        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(np.dot(a, b) / (norm_a * norm_b))

    @staticmethod
    def _l2_normalize(vec):
        norm = np.linalg.norm(vec)

        if norm == 0:
            return vec

        return vec / norm