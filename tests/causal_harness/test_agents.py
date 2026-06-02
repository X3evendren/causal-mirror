"""Tests for multi-agent organization."""
from causal_harness.agents.roles import (
    AgentRole, RoleDefinition, RoleRegistry, ROLE_DEFINITIONS,
)
from causal_harness.agents.blackboard import (
    Blackboard, BlackboardEntry, EntryType,
)
from causal_harness.agents.arbitration import (
    Arbitrator, ArbitrationResult, ArbitrationOutcome,
)
from causal_harness.agents.credit import CreditAssigner, CreditRecord


class TestRoles:
    def test_all_roles_defined(self):
        assert len(ROLE_DEFINITIONS) == 5
        assert AgentRole.PLANNER in ROLE_DEFINITIONS
        assert AgentRole.REFEREE in ROLE_DEFINITIONS

    def test_executor_can_edit(self):
        role_def = ROLE_DEFINITIONS[AgentRole.EXECUTOR]
        assert role_def.may_use("edit_file")
        assert role_def.may_use("shell")

    def test_planner_cannot_execute(self):
        role_def = ROLE_DEFINITIONS[AgentRole.PLANNER]
        assert not role_def.may_use("edit_file")
        assert role_def.may_use("read_file")

    def test_verifier_can_disagree(self):
        role_def = ROLE_DEFINITIONS[AgentRole.VERIFIER]
        assert role_def.can_disagree
        assert role_def.requires_evidence

    def test_registry_enforces_single_assignment(self):
        reg = RoleRegistry()
        reg.register(AgentRole.EXECUTOR, "agent_1")
        with __import__('pytest').raises(ValueError):
            reg.register(AgentRole.EXECUTOR, "agent_2")

    def test_registry_tool_access(self):
        reg = RoleRegistry()
        assert reg.check_tool_access(AgentRole.PLANNER, "read_file")
        assert not reg.check_tool_access(AgentRole.PLANNER, "edit_file")

    def test_required_roles_simple(self):
        reg = RoleRegistry()
        roles = reg.required_roles_for_task("simple")
        assert AgentRole.EXECUTOR in roles
        assert AgentRole.REFEREE not in roles

    def test_required_roles_complex(self):
        reg = RoleRegistry()
        roles = reg.required_roles_for_task("complex")
        assert len(roles) == 5


class TestBlackboard:
    def test_post_and_query(self):
        bb = Blackboard()
        bb.post(BlackboardEntry(
            entry_id="", entry_type=EntryType.PLAN,
            role=AgentRole.PLANNER, content="Do X then Y",
        ))
        results = bb.query(entry_type=EntryType.PLAN)
        assert len(results) == 1

    def test_query_by_role(self):
        bb = Blackboard()
        bb.post(BlackboardEntry("", EntryType.PLAN, AgentRole.PLANNER, "Plan"))
        bb.post(BlackboardEntry("", EntryType.RESEARCH, AgentRole.RESEARCHER, "Research"))
        results = bb.query(role=AgentRole.PLANNER)
        assert len(results) == 1

    def test_thread_traces_references(self):
        bb = Blackboard()
        plan = BlackboardEntry("", EntryType.PLAN, AgentRole.PLANNER, "Plan")
        bb.post(plan)
        result = BlackboardEntry(
            "", EntryType.ACTION_RESULT, AgentRole.EXECUTOR,
            "Done", references=[plan.entry_id],
        )
        bb.post(result)
        obj = BlackboardEntry(
            "", EntryType.OBJECTION, AgentRole.VERIFIER,
            "Wrong", references=[result.entry_id],
        )
        bb.post(obj)

        thread = bb.get_thread(result.entry_id)
        assert len(thread) >= 2  # result + objection

    def test_objections_pending(self):
        bb = Blackboard()
        result = BlackboardEntry("", EntryType.ACTION_RESULT, AgentRole.EXECUTOR, "Done")
        bb.post(result)
        obj = BlackboardEntry(
            "", EntryType.OBJECTION, AgentRole.VERIFIER,
            "Wrong", references=[result.entry_id],
        )
        bb.post(obj)
        pending = bb.objections_pending()
        assert len(pending) == 1


class TestArbitrator:
    def test_verifier_with_evidence_wins(self):
        arb = Arbitrator()
        exec_entry = BlackboardEntry(
            "", EntryType.ACTION_RESULT, AgentRole.EXECUTOR,
            "Tests pass", evidence=[],
        )
        verif_entry = BlackboardEntry(
            "", EntryType.OBJECTION, AgentRole.VERIFIER,
            "Tests fail", evidence=["/path/to/test_output.txt", "test:test_foo"],
        )
        result = arb.resolve_disagreement(verif_entry, exec_entry)
        assert result.outcome == ArbitrationOutcome.ACCEPT
        assert verif_entry.entry_id in result.accepted_entries

    def test_no_evidence_requests_more(self):
        arb = Arbitrator()
        a = BlackboardEntry("", EntryType.ACTION_RESULT, AgentRole.EXECUTOR, "OK")
        b = BlackboardEntry("", EntryType.ACTION_RESULT, AgentRole.PLANNER, "Not OK")
        result = arb.resolve_disagreement(a, b)
        assert result.outcome == ArbitrationOutcome.MORE_EVIDENCE

    def test_no_pending_objections_accepts(self):
        arb = Arbitrator()
        bb = Blackboard()
        result = arb.resolve(bb)
        assert result.outcome == ArbitrationOutcome.ACCEPT


class TestCreditAssigner:
    def test_positive_credit_for_risk_reduction_success(self):
        ca = CreditAssigner()
        cr = ca.record("ep1", "executor", "EDIT_FILE", 0.8, 0.3, outcome=0)
        assert cr.contribution == 0.5  # risk dropped 0.5, success

    def test_negative_credit_for_risk_reduction_failure(self):
        ca = CreditAssigner()
        cr = ca.record("ep1", "executor", "EDIT_FILE", 0.8, 0.3, outcome=1)
        assert cr.contribution == -0.5  # risk dropped but episode still failed

    def test_role_totals_accumulate(self):
        ca = CreditAssigner()
        ca.record("ep1", "verifier", "VERIFY", 0.6, 0.2, outcome=0)
        ca.record("ep1", "verifier", "VERIFY", 0.4, 0.1, outcome=0)
        assert ca.role_credit("verifier") == 0.7  # 0.4 + 0.3

    def test_top_contributors(self):
        ca = CreditAssigner()
        ca.record("e1", "executor", "EDIT", 0.5, 0.1, 0)
        ca.record("e1", "researcher", "SEARCH", 0.6, 0.5, 0)
        ca.record("e1", "planner", "PLAN", 0.3, 0.4, 1)
        top = ca.top_contributors(3)
        assert top[0][1] >= top[-1][1]  # sorted descending

    def test_episode_summary(self):
        ca = CreditAssigner()
        ca.record("ep1", "executor", "EDIT", 0.8, 0.3, 0)
        ca.record("ep1", "verifier", "VERIFY", 0.3, 0.1, 0)
        summary = ca.episode_summary("ep1")
        assert summary["total_actions"] == 2
        assert summary["net_credit"] > 0
