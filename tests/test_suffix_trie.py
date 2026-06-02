"""Tests for SuffixTrie."""

import pytest
from cssr.suffix_trie import SuffixTrie


def test_empty_build():
    trie = SuffixTrie({"A"})
    trie.build([], 3)
    assert trie.root.count == 0
    assert trie.suffixes_of_length(1) == []


def test_single_symbol():
    trie = SuffixTrie({"A"})
    trie.build(["A", "A", "A"], 2)
    # The single-symbol case: only one node chain
    node = trie.get_node(("A",))
    assert node is not None
    assert node.count == 3  # appears at positions 0,1,2


def test_two_symbol_sequence():
    trie = SuffixTrie({"A", "B"})
    trie.build(["A", "B", "A", "B"], 2)
    # suffix "A": appears at positions 0 and 2
    node_a = trie.get_node(("A",))
    assert node_a is not None
    assert node_a.count == 2
    # next symbol after "A" is "B" both times
    assert node_a.next_counts.get("B", 0) == 2
    # suffix ("A","B") = forward "AB", appears at pos [0,1] and [2,3]
    # (trie built oldest-first, queried oldest-first: root→A→B)
    node_ab = trie.get_node(("A", "B"))
    assert node_ab is not None
    assert node_ab.count == 2
    # suffix ("B","A") = forward "BA", appears at pos [1,2]
    node_ba = trie.get_node(("B", "A"))
    assert node_ba is not None
    assert node_ba.count == 1


def test_next_counts():
    trie = SuffixTrie({"0", "1"})
    seq = ["0", "1", "0", "0", "1", "0"]
    trie.build(seq, 2)
    # After "1": at positions 1 and 4, next symbol: position 2="0", position 5="0"
    counts = trie.get_next_counts(("1",))
    assert counts == {"0": 2}
    # After "0": at positions 0,2,3, next: position1="1", position3="0", position4="1"
    counts_0 = trie.get_next_counts(("0",))
    assert counts_0["1"] == 2
    assert counts_0["0"] == 1


def test_suffixes_of_length():
    trie = SuffixTrie({"A", "B"})
    trie.build(["A", "B", "A"], 2)
    suffixes_l1 = trie.suffixes_of_length(1)
    assert len(suffixes_l1) == 2
    assert ("A",) in suffixes_l1
    assert ("B",) in suffixes_l1


def test_suffixes_of_length_zero():
    trie = SuffixTrie({"A"})
    trie.build(["A"], 1)
    suffixes = trie.suffixes_of_length(0)
    assert suffixes == [()]


def test_get_count():
    trie = SuffixTrie({"X", "Y"})
    trie.build(["X", "X", "Y", "X", "X", "Y"], 3)
    assert trie.get_count(("X",)) == 4
    assert trie.get_count(("Y",)) == 2
    assert trie.get_count(("Z",)) == 0
