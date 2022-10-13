"""test Trie"""

from libs.alg import Trie as _Trie


class Trie:
    def __init__(self):
        self.trie = _Trie()

    def insert(self, word: str) -> None:
        self.trie.insert(word)

    def search(self, word: str) -> bool:
        return self.trie.search(word)

    def startsWith(self, prefix: str) -> bool:
        return self.trie.starts_with(prefix)
