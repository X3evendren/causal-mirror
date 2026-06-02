"""Phase 1: Suffix Trie construction for efficient substring frequency counting.

Builds a trie of all suffixes up to length L_max+1 from a discrete symbol sequence.
One-pass O(N) construction, recording both occurrence counts and next-symbol counts.
"""

from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class SuffixTrieNode:
    """A node in the suffix trie representing one substring.

    Each node's path from root spells the substring.
    """
    count: int = 0
    next_counts: dict = field(default_factory=lambda: defaultdict(int))
    children: dict = field(default_factory=dict)


class SuffixTrie:
    """Suffix trie for substring frequency queries.

    Supports: frequency lookup, next-symbol count lookup, and enumeration
    of observed suffixes by length.
    """

    def __init__(self, alphabet: set):
        self.alphabet = alphabet
        self.root = SuffixTrieNode()
        self.max_depth = 0

    def build(self, sequence, L_max: int) -> None:
        """One-pass sliding-window construction.

        For each position i, walk forward up to L_max+1 steps, recording
        each observed suffix and (if available) the following symbol.

        Args:
            sequence: List of symbols (hashable).
            L_max: Maximum history length to consider.
        """
        N = len(sequence)
        max_depth = min(L_max + 1, N)
        self.max_depth = max_depth

        for i in range(N):
            node = self.root
            for offset in range(max_depth):
                idx = i + offset
                if idx >= N:
                    break
                sym = sequence[idx]
                if sym not in node.children:
                    node.children[sym] = SuffixTrieNode()
                node = node.children[sym]
                node.count += 1
                # Record next symbol if available
                if idx + 1 < N:
                    node.next_counts[sequence[idx + 1]] += 1

    def get_node(self, suffix: tuple) -> SuffixTrieNode | None:
        """Traverse to the node for a given suffix tuple. Returns None if not found.

        The trie is built oldest-first (position i walks forward),
        so queries must also use oldest-first order.
        """
        node = self.root
        for sym in suffix:
            if sym not in node.children:
                return None
            node = node.children[sym]
        return node

    def get_next_counts(self, suffix: tuple) -> dict:
        """Return {next_symbol: count} for the given suffix."""
        node = self.get_node(suffix)
        if node is None:
            return {}
        return dict(node.next_counts)

    def get_count(self, suffix: tuple) -> int:
        """Return total occurrences of this suffix."""
        node = self.get_node(suffix)
        if node is None:
            return 0
        return node.count

    def suffixes_of_length(self, L: int) -> list[tuple]:
        """Enumerate all observed suffixes of exactly length L.

        Returns:
            List of suffix tuples (oldest-first order, i.e. [s_{t-L}, ..., s_{t-1}]).
        """
        if L == 0:
            return [()]

        result = []

        def collect(node, depth, current):
            if depth == L:
                result.append(tuple(current))
                return
            for sym, child in node.children.items():
                current.append(sym)
                collect(child, depth + 1, current)
                current.pop()

        collect(self.root, 0, [])
        return result

    def suffix_tree_size(self) -> int:
        """Return total number of nodes in the trie (for debugging/memory estimation)."""
        def count_nodes(node):
            n = 1
            for child in node.children.values():
                n += count_nodes(child)
            return n
        return count_nodes(self.root)
