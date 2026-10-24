import time


# 依赖（ES、Redis）运行中出故障时，冷却期内直接走降级实现，不让每个请求都先等一次超时；
# 降级原因写进 Services.degraded，/api/health 能看到，恢复后自动清掉
class Breaker:
    def __init__(self, name: str, degraded: dict[str, str], cooldown: float = 30.0):
        self.name = name
        self._degraded = degraded
        self._cooldown = cooldown
        self._open_until = 0.0
        self._tripped = False

    def is_open(self) -> bool:
        return time.monotonic() < self._open_until

    def failure(self, reason: str) -> None:
        self._open_until = time.monotonic() + self._cooldown
        self._tripped = True
        self._degraded[self.name] = reason

    # 只清掉自己写进去的降级原因，启动时就降级的原因不归它管
    def success(self) -> None:
        if self._tripped:
            self._tripped = False
            self._degraded.pop(self.name, None)
