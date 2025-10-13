from __future__ import annotations
from typing import Dict
from langchain.memory import ConversationBufferWindowMemory

class _Mem:
    def __init__(self, window:int=10):
        self.window = window
        self._mem: Dict[str, ConversationBufferWindowMemory] = {}

    def get(self, session_id: str) -> ConversationBufferWindowMemory:
        sid = session_id or "default"
        if sid not in self._mem:
            self._mem[sid] = ConversationBufferWindowMemory(
                k=self.window, memory_key="chat_history", return_messages=True
            )
        return self._mem[sid]

MEMORY = _Mem()