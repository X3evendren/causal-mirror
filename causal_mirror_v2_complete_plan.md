# Causal Mirror V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans or an equivalent task-by-task execution workflow. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild Causal Mirror from an offline self-report prototype into a causally instrumented agent harness that can predict action consequences, preserve useful memory, sense uncertainty, calibrate self-evaluation, execute robustly, coordinate agents, enforce safety, and evaluate long-horizon reliability.

**Architecture:** Keep the existing CSSR/CM-DGM research code as the causal-analysis kernel, reuse nanobot as the mature harness substrate, and add a new control layer that turns causal state estimates into runtime decisions. The central design change is moving CSSR from "after-the-fact report generation" into the online control loop.

**Tech Stack:** Python 3.11+, numpy/scipy/statsmodels, pydantic or dataclasses for event schemas, JSONL/SQLite for local ledgers, pytest for verification, nanobot hooks/tools/memory as integration substrate.

---

## 1. Current Diagnosis

### 1.1 Existing Assets Worth Keeping

- `cssr/`: a usable CSSR pipeline with suffix trie, splitting, determinization, epsilon-machine assembly, metrics, and visualization.
- `agent/`: a minimal self-modeling loop: task solving, behavior encoding, CSSR analysis, and natural-language self-report generation.
- `cm_dgm/`: an early population/evolution experiment that compares causal-metric fitness against accuracy-only fitness.
- `nanobot-main/`: a mature agent harness with loop state machine, hooks, tool registry, subagents, memory, filesystem/tool safety, shell guardrails, web tools, and session persistence.

### 1.2 Main Architectural Problem

The project currently has two separated systems:

```text
Causal Mirror: behavior -> CSSR -> report -> next prompt
Nanobot: user/task -> agent loop -> tools -> memory/sessions
```

The intended final system should be:

```text
agent loop -> event ledger -> causal state model -> policy controller -> safer next action
```

The difference is important. A report is advice; a controller changes runtime behavior.

### 1.3 Critical Technical Risks

1. CSSR metrics appear unreliable on agent traces. Existing result files show `n_states > 1` while `C_mu`, `E`, and `predictive_information` remain `0.0`. Until fixed, causal fitness and self-report claims are not trustworthy.
2. Behavior encoding is too lossy. A 16-symbol action-quality alphabet erases task type, tool context, verification type, error class, and recovery path.
3. Episodes are likely concatenated too aggressively. Mixing multiple tasks into one raw sequence can create artificial transitions between unrelated problems.
4. Current feedback path is prompt-level, not control-level. It may improve by being "more instruction", not by using causal structure.
5. CM-DGM is premature until the causal state model demonstrably predicts failures or useful interventions.
6. Evaluation is too short-horizon. Accuracy over a few rounds does not measure reliability, recovery, tool safety, or long-term identity.

---

## 2. Product Scope

This project should not try to solve all eight agent deficits at once. The V2 MVP should solve five directly and lay foundations for the remaining three.

### Direct MVP Targets

1. **Causal world model:** represent actions with preconditions, expected effects, observed effects, and rollback options.
2. **Active perception:** choose whether to inspect, search, run tests, ask the user, or proceed based on uncertainty and expected information gain.
3. **Calibrated self-evaluation:** estimate risk before and after actions using causal state, tool history, and verification results.
4. **Robust execution:** add verification gates, dry-run checks, transaction logs, rollback recommendations, and loop detection.
5. **Evaluation system:** measure long-horizon reliability, not just benchmark accuracy.

### Phase-2 Targets

6. **Long-term memory and identity continuity:** structured memory beyond raw text.
7. **Safety for high-privilege action:** risk-tiered authorization and sandbox-aware policies.

### Phase-3 Target

8. **Multi-agent organization:** planner/executor/verifier/researcher/referee roles with causal credit assignment.

---

## 3. Target Architecture

```text
                 user / scheduled task / benchmark
                              |
                              v
                    Nanobot Agent Loop
                              |
                              v
                    Causal Mirror Hook
                              |
          +-------------------+-------------------+
          |                   |                   |
          v                   v                   v
    Event Ledger       World Model Store     Memory Store
          |                   |                   |
          v                   v                   v
   Behavior Encoder -> Causal State Model -> Policy Controller
                              |
                              v
          action constraints / verification / sensing / escalation
                              |
                              v
                     next agent iteration
```

### 3.1 Event Ledger

The event ledger is the source of truth. It records observable behavior, not hidden chain-of-thought.

Minimum event types:

- `turn_started`
- `goal_loaded`
- `observation_added`
- `action_proposed`
- `tool_called`
- `tool_result`
- `verification_run`
- `error_detected`
- `repair_attempted`
- `user_approval_requested`
- `state_transition`
- `turn_completed`

Each event must include:

- `event_id`
- `episode_id`
- `turn_id`
- `timestamp`
- `actor`
- `event_type`
- `payload`
- `risk_before`
- `risk_after`
- `outcome`

### 3.2 Behavior Encoder V2

Replace the current flat 16-symbol alphabet with a two-layer representation.

Layer A: symbolic sequence for CSSR:

```text
OBSERVE
PLAN
READ_FILE
EDIT_FILE
RUN_TEST
VERIFY
ERROR
REPAIR
ASK_USER
ESCALATE
COMPLETE
```

Layer B: structured side-channel features:

```json
{
  "tool": "exec",
  "target_type": "test",
  "error_class": "assertion_failure",
  "reversible": true,
  "confidence": 0.72,
  "verification_depth": 2,
  "task_domain": "coding",
  "changed_files": 3
}
```

CSSR receives Layer A. Risk models and reports use both layers.

### 3.3 Causal State Model

The causal state model should output actionable quantities:

- current causal state
- state transition probabilities
- `P(failure | state, action)`
- `P(repair_success | state, repair_action)`
- expected information gain for sensing actions
- loop/rut probability
- confidence calibration error

CSSR remains one estimator, but the interface should allow alternatives:

- CSSR epsilon-machine
- finite-order Markov model baseline
- logistic risk model baseline
- hand-written policy baseline

This prevents overfitting the whole project to one algorithm.

### 3.4 Policy Controller

The controller translates causal state into runtime constraints.

Initial policies:

- If `P(failure | state, action) > 0.60`, require a verification action before execution.
- If the same repair class fails twice, switch repair strategy.
- If information gain from inspection exceeds action value, inspect first.
- If action is high impact and rollback is unavailable, request user approval.
- If loop probability exceeds threshold, summarize state and re-plan.
- If confidence is low and task ambiguity is high, ask the user instead of guessing.

### 3.5 World Model Store

The world model is not a full simulator. It is an action-effect registry.

Each action schema contains:

```yaml
name: edit_file
preconditions:
  - file has been read recently
effects:
  - file content may change
observability:
  - git diff
  - tests
rollback:
  - reverse patch
risk_tier: medium
verification:
  - run targeted tests
  - inspect diff
```

### 3.6 Memory Store

Memory should be structured into four stores:

- `identity`: stable project/agent operating principles
- `preferences`: user preferences and constraints
- `project_state`: current files, decisions, open risks, completed milestones
- `skills`: reusable procedures learned from successful episodes

Raw logs stay in the ledger. Long-term memory stores distilled, typed facts.

### 3.7 Safety Governor

Safety policy should be independent from the LLM.

Risk tiers:

- `low`: read/search/list/summarize
- `medium`: write files, run tests, create branches, install local deps
- `high`: network calls, credential access, broad filesystem edits, sending messages
- `critical`: destructive operations, external side effects, production deploys

High and critical actions require explicit approval or preconfigured policy.

---

## 4. Proposed File Structure

Create a new package instead of mixing research code, experiments, and harness integration.

```text
causal_harness/
  events/
    schema.py
    ledger.py
    episode.py
  encoding/
    symbols.py
    encoder.py
    side_channel.py
  models/
    base.py
    cssr_model.py
    markov_baseline.py
    risk_model.py
  controller/
    policy.py
    active_sensing.py
    verification.py
    loop_detection.py
  world/
    action_schema.py
    registry.py
    defaults.yaml
  memory/
    schema.py
    store.py
    consolidation.py
  safety/
    risk.py
    governor.py
    approvals.py
  eval/
    scenario.py
    runner.py
    metrics.py
    reports.py
integrations/
  nanobot/
    causal_mirror_hook.py
    event_adapters.py
    policy_adapter.py
tests/
  causal_harness/
    ...
```

Keep the existing `cssr/` package, but harden it before using it in the control loop.

---

## 5. Implementation Roadmap

### Phase 0: Stabilize The Research Kernel

Goal: make causal metrics trustworthy.

- [ ] Add regression tests proving that multi-state recurrent machines produce non-zero `C_mu`.
- [ ] Add tests for Even Process, Golden Mean Process, IID, and second-order Markov with explicit expected ranges.
- [ ] Add tests for transition matrix row normalization and non-degenerate steady-state distributions.
- [ ] Add tests proving episode boundaries do not create artificial transitions.
- [ ] Inspect `cssr/machine.py`, `cssr/determinization.py`, and `cssr/splitting.py` for degenerate transitions, empty emission distributions, and state reindexing bugs.
- [ ] Change CM-DGM fitness so it refuses to use causal bonuses when metrics fail validation.

Acceptance criteria:

- Known-process tests pass.
- `n_states > 1` with recurrent non-degenerate transitions yields `C_mu > 0`.
- Existing result anomalies are explained by tests or fixed code.

### Phase 1: Build The Event Ledger

Goal: capture real agent behavior in a replayable format.

- [ ] Create event schemas.
- [ ] Create append-only JSONL ledger.
- [ ] Add episode boundaries.
- [ ] Add adapters for tool calls, tool results, errors, verification, approvals, and final outcomes.
- [ ] Add replay loader that reconstructs episodes from the ledger.
- [ ] Add tests for append, replay, corrupted record handling, and schema migration.

Acceptance criteria:

- A full agent turn can be replayed from ledger events.
- The ledger records enough context to classify action, outcome, and verification without reading conversation history.

### Phase 2: Integrate With Nanobot Through Hooks

Goal: observe live harness behavior without changing the core loop first.

- [ ] Implement `CausalMirrorHook` using nanobot's `AgentHook` lifecycle.
- [ ] Record `before_iteration`, `before_execute_tools`, `after_iteration`, and finalization events.
- [ ] Map nanobot tool calls into action symbols.
- [ ] Map tool errors into stable error classes.
- [ ] Save per-session causal mirror logs.

Acceptance criteria:

- Running nanobot on a task produces an event ledger.
- No policy intervention yet; this phase is observe-only.
- Existing nanobot tests continue to pass.

### Phase 3: Add Risk Prediction And Calibrated Self-Evaluation

Goal: estimate when the agent is likely to fail.

- [ ] Train/evaluate baseline risk predictors from ledger episodes.
- [ ] Implement CSSR-backed causal state prediction.
- [ ] Compare CSSR against Markov and logistic baselines.
- [ ] Add calibration metrics: Brier score, ECE, false-safe rate, false-alarm rate.
- [ ] Generate machine-readable self-evaluation, not just prose.

Acceptance criteria:

- Risk predictor beats simple frequency baseline on held-out episodes.
- Self-evaluation contains a numeric risk estimate and recommended verification action.

### Phase 4: Turn Prediction Into Control

Goal: modify runtime behavior through a policy controller.

- [ ] Implement policy rules for verification, sensing, approval, loop breaking, and repair switching.
- [ ] Add an intervention queue that injects policy messages or tool constraints into the next iteration.
- [ ] Add active sensing: choose read/search/test/ask based on uncertainty and expected information gain.
- [ ] Add loop detection using repeated state-action patterns.
- [ ] Add opt-out config so each policy can be disabled during experiments.

Acceptance criteria:

- High-risk actions trigger verification or approval.
- Repeated failed repairs trigger strategy switching.
- Active sensing reduces avoidable failures in benchmark scenarios.

### Phase 5: Structured Memory And Identity Continuity

Goal: stop treating memory as a bag of text.

- [ ] Define typed memory schemas for identity, preferences, project state, and skills.
- [ ] Build consolidation from ledger episodes into typed memory.
- [ ] Add memory provenance: every memory item links back to source events.
- [ ] Add decay/update rules for stale project state.
- [ ] Add retrieval rules that prefer causal relevance over semantic similarity alone.

Acceptance criteria:

- The agent can recover project state after context loss.
- Memory retrieval has measurable precision/recall on synthetic and real project tasks.
- Stale or contradicted memories are superseded, not blindly accumulated.

### Phase 6: Robust Execution And Safety Governor

Goal: make high-impact actions safer.

- [ ] Define action risk tiers.
- [ ] Add precondition checks for file edits, shell commands, web/network operations, and external messaging.
- [ ] Add transaction records for reversible actions.
- [ ] Add rollback recommendations for failed edits.
- [ ] Add approval gates for high/critical actions.
- [ ] Add prompt-injection and untrusted-content labels to event payloads.

Acceptance criteria:

- Dangerous actions cannot be executed without approval or policy allowance.
- Failed file-edit tasks produce enough data to recover or revert manually.
- Prompt-injection tests do not bypass policy.

### Phase 7: Multi-Agent Organization

Goal: add role separation without creating uncontrolled chatter.

- [ ] Define fixed roles: planner, executor, verifier, researcher, referee.
- [ ] Use a blackboard/event ledger, not free-form agent-to-agent conversation.
- [ ] Add arbitration rules for conflicting recommendations.
- [ ] Add causal credit assignment: which role/action reduced or increased failure risk.
- [ ] Add anti-collusion checks: verifier must use independent evidence.

Acceptance criteria:

- Multi-agent mode improves complex task reliability over single-agent mode on selected scenarios.
- Verifier catches injected or accidental executor mistakes.
- Role messages are traceable through the event ledger.

### Phase 8: Evaluation System

Goal: measure what current benchmarks miss.

- [ ] Build scenario suites for coding, research, file operations, web tasks, memory recall, safety, and multi-turn planning.
- [ ] Track longitudinal metrics over days/weeks.
- [ ] Add intervention A/B tests: observe-only vs policy-control vs prompt-report.
- [ ] Add reliability metrics: recovery rate, silent failure rate, unnecessary escalation rate, loop rate, rollback success, calibration error.
- [ ] Generate experiment reports with causal claims separated from correlations.

Acceptance criteria:

- Every claimed improvement has a baseline and ablation.
- The system can show where CSSR helps, where it does not, and where simpler baselines are enough.

---

## 6. Experimental Design

### 6.1 Required Baselines

Do not compare only "Causal Mirror vs naked agent." Use:

- naked agent
- generic prompt feedback
- random-length matched report
- shuffled causal report
- Markov-risk controller
- CSSR-risk controller
- human-written policy rules

### 6.2 Core Metrics

- task success rate
- silent failure rate
- recovery rate after first error
- verification precision
- verification recall
- Brier score for risk prediction
- expected calibration error
- number of tool calls per success
- context tokens per success
- high-risk action approval correctness
- memory retrieval precision/recall
- multi-agent disagreement resolution accuracy

### 6.3 Causal Claim Standard

A result may be called "causal" only when:

1. the intervention is defined before the run,
2. the control group is matched,
3. confounds like report length and extra attention are controlled,
4. logs show the intervention changed a concrete action,
5. the changed action plausibly caused the outcome difference.

---

## 7. Go/No-Go Gates

### Gate A: CSSR Validity

Proceed only if CSSR metrics are valid on known processes and non-degenerate agent traces.

### Gate B: Predictive Usefulness

Proceed only if causal/risk states predict failures better than simple baselines.

### Gate C: Control Usefulness

Proceed only if interventions reduce silent failures or improve recovery without excessive tool cost.

### Gate D: Safety

Proceed only if high-risk actions are blocked, logged, or approved according to policy.

### Gate E: Multi-Agent

Proceed only if single-agent control has stabilized. Multi-agent organization should not be added before the ledger/controller foundation works.

---

## 8. Recommended Build Order

1. Fix CSSR and metric validation.
2. Build event ledger.
3. Integrate observe-only nanobot hook.
4. Build behavior encoder V2.
5. Build risk model baselines.
6. Add CSSR-backed causal state model.
7. Add policy controller in dry-run mode.
8. Enable policy interventions one at a time.
9. Add structured memory.
10. Add safety governor.
11. Add long-horizon evaluation suite.
12. Add multi-agent roles only after single-agent reliability is measurable.

---

## 9. Definition Of Done For V2 MVP

V2 MVP is complete when:

- A nanobot task produces a replayable event ledger.
- The system can infer a current causal/risk state from recent events.
- The system can predict failure risk with measured calibration.
- The policy controller can force verification or ask for more information.
- The controller's interventions improve at least one reliability metric against matched baselines.
- Memory stores project state and user preferences as typed, source-linked records.
- High-risk actions are governed by explicit risk tier and approval policy.
- Evaluation reports include ablations and avoid unsupported causal claims.

---

## 10. Strategic Recommendation

Do not fully rewrite everything from scratch. Keep:

- `cssr/` after hardening,
- `nanobot-main/` as the harness substrate,
- existing experiment data as diagnostic evidence.

Do rewrite:

- the behavior encoding layer,
- the self-report feedback loop,
- the CM-DGM causal fitness gate,
- the evaluation framework.

The project should become a causal control harness, not just a self-reflection demo.
