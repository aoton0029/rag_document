class RedisClient:
    def __init__(self, host: str, port: int, db: int):
        import redis
        self.client = redis.StrictRedis(host=host, port=port, db=db, decode_responses=True)

    def set(self, key: str, value: str):
        self.client.set(key, value)

    def get(self, key: str) -> str:
        return self.client.get(key)

    def delete(self, key: str):
        self.client.delete(key)

    def exists(self, key: str) -> bool:
        return self.client.exists(key)

    def set_expiry(self, key: str, seconds: int):
        self.client.expire(key, seconds)

    def flush_db(self):
        self.client.flushdb()