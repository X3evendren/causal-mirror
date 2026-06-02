"""Tests for Behavior Encoder V2."""
import tempfile

from causal_harness.events.schema import (
    EventType, EventOutcome, new_event,
)
from causal_harness.events.ledger import EventLedger
from causal_harness.events.adapters import tool_event, error_event
from causal_harness.encoding.symbols import ActionSymbol, BOUNDARY, TOOL_TO_SYMBOL
from causal_harness.encoding.encoder import BehaviorEncoderV2


class TestEncoder:
    def test_encode_tool_call(self):
        encoder = BehaviorEncoderV2()
        event = tool_event("read_file", {"file_path": "/tmp/x.py"}, "ep1", 1)
        sym = encoder.encode_event(event)
        assert sym == "READ_FILE"

    def test_encode_search_as_observe(self):
        encoder = BehaviorEncoderV2()
        event = tool_event("web_search", {"query": "test"}, "ep1", 1)
        sym = encoder.encode_event(event)
        assert sym == "OBSERVE"

    def test_encode_edit_file(self):
        encoder = BehaviorEncoderV2()
        event = tool_event("edit_file", {"file_path": "foo.py"}, "ep1", 1)
        sym = encoder.encode_event(event)
        assert sym == "EDIT_FILE"

    def test_unknown_tool_defaults_to_plan(self):
        encoder = BehaviorEncoderV2()
        event = tool_event("some_new_tool", {}, "ep1", 1)
        sym = encoder.encode_event(event)
        assert sym == "PLAN"

    def test_error_then_repair(self):
        encoder = BehaviorEncoderV2()
        err = new_event(EventType.ERROR_DETECTED, "ep1", 1,
                        payload={"error_class": "assertion_failure"})
        sym_err = encoder.encode_event(err)
        assert sym_err == "ERROR"

        # Next tool call within same turn → REPAIR
        repair = tool_event("edit_file", {"file_path": "x.py"}, "ep1", 1)
        sym_repair = encoder.encode_event(repair)
        assert sym_repair == "REPAIR"

    def test_episode_boundaries(self):
        encoder = BehaviorEncoderV2()
        events = [
            tool_event("read_file", {"file_path": "a.py"}, "ep1", 1),
            tool_event("edit_file", {"file_path": "a.py"}, "ep1", 1),
            tool_event("read_file", {"file_path": "b.py"}, "ep2", 1),
            tool_event("edit_file", {"file_path": "b.py"}, "ep2", 1),
        ]
        symbols = encoder.encode_all(iter(events))
        assert BOUNDARY in symbols
        assert symbols == ["READ_FILE", "EDIT_FILE", BOUNDARY, "READ_FILE", "EDIT_FILE"]

    def test_encode_episode_adds_boundary(self):
        encoder = BehaviorEncoderV2()
        events = [
            tool_event("read_file", {"file_path": "x.py"}, "ep1", 1),
            tool_event("shell", {"command": "pytest"}, "ep1", 1),
        ]
        symbols = encoder.encode_episode(events)
        assert symbols == ["READ_FILE", "RUN_TEST", BOUNDARY]

    def test_verification_event(self):
        encoder = BehaviorEncoderV2()
        from causal_harness.events.adapters import verification_event
        event = verification_event("ep1", 1, "test", passed=True)
        sym = encoder.encode_event(event)
        assert sym == "VERIFY"

    def test_error_event(self):
        encoder = BehaviorEncoderV2()
        event = error_event("ep1", 1, "timeout", "Connection timed out")
        sym = encoder.encode_event(event)
        assert sym == "ERROR"

    def test_all_symbols_are_short_strings(self):
        for sym in ActionSymbol:
            assert isinstance(sym.value, str)
            assert len(sym.value) <= 12


class TestEncoderIntegration:
    """End-to-end: events → ledger → encoder → CSSR-ready symbols."""

    def test_full_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = EventLedger(tmp)

            # Simulate a multi-episode run
            # Episode 1: read → edit → test → error → repair → complete
            ledger.append(tool_event("read_file", {"file_path": "f.py"}, "ep1", 1))
            ledger.append(tool_event("edit_file", {"file_path": "f.py"}, "ep1", 1))
            ledger.append(tool_event("shell", {"command": "pytest"}, "ep1", 2))
            ledger.append(error_event("ep1", 2, "assertion_failure", "test_foo failed"))
            ledger.append(tool_event("edit_file", {"file_path": "f.py"}, "ep1", 3))
            ledger.append(tool_event("shell", {"command": "pytest"}, "ep1", 3))

            # Episode 2: observe → plan → complete
            ledger.append(tool_event("web_search", {"query": "how to X"}, "ep2", 1))
            ledger.append(tool_event("edit_file", {"file_path": "g.py"}, "ep2", 1))

            # Encode
            encoder = BehaviorEncoderV2()
            symbols = encoder.encode_all(ledger.replay())

            # Should have BOUNDARY between episodes
            assert BOUNDARY in symbols
            # Episode 1 symbols + BOUNDARY + Episode 2 symbols
            assert len(symbols) == 6 + 1 + 2  # (R,E,E,E,E,R) + B + (O,E)
            # Episode 2 symbol
            assert symbols == [
                "READ_FILE", "EDIT_FILE", "RUN_TEST", "ERROR", "REPAIR", "RUN_TEST",
                "BOUNDARY",
                "OBSERVE", "EDIT_FILE",
            ]

    def test_cssr_can_process_encoded_symbols(self):
        """CSSR should accept the new alphabet without errors."""
        from cssr import CSSR, CSSRConfig
        from causal_harness.encoding.symbols import TOOL_TO_SYMBOL

        # Build a minimal multi-episode symbol sequence
        symbols = ["OBSERVE", "PLAN", "EDIT_FILE", "VERIFY", "COMPLETE", "BOUNDARY",
                    "OBSERVE", "EDIT_FILE", "RUN_TEST", "ERROR", "REPAIR", "VERIFY",
                    "COMPLETE", "BOUNDARY",
                    "READ_FILE", "PLAN", "EDIT_FILE", "COMPLETE", "BOUNDARY",
                    "OBSERVE", "OBSERVE", "PLAN", "RUN_TEST", "COMPLETE"]

        config = CSSRConfig(L_max=3, alpha=0.05, min_count=2)
        cssr = CSSR(config).fit(symbols)
        m = cssr.metrics

        assert m["alphabet_size"] >= 5
        assert m["n_states"] >= 1
        # The boundary symbol creates a clear structural breakpoint
        # that CSSR should detect as a recurrent pattern
