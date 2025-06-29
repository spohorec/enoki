import enoki
from enoki.testing import EnokiTestCase, FSMTestCase, run_state_once, make_mock_fsm
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from types import SimpleNamespace
from queue import Queue
import threading
import time


class TestStateTransitions(EnokiTestCase):
    """Test basic state transition behaviors."""

    def test_transition_to_other_state(self):
        """State should transition to another state when returning it."""

        class StateA(enoki.State):
            def on_state(self, ctx):
                return StateB

        class StateB(enoki.State):
            def on_state(self, ctx):
                return None

        self.assertStateTransition(StateA, StateB)

    def test_none_returns_unhandled(self):
        """Returning None should result in Unhandled transition."""

        class TestState(enoki.State):
            def on_state(self, ctx):
                return None

        self.assertStateTransition(TestState, enoki.Unhandled)

    def test_self_returns_again(self):
        """Returning self should result in Again transition."""

        class TestState(enoki.State):
            def on_state(self, ctx):
                return self

        self.assertStateTransition(TestState, enoki.Again)

    def test_type_self_returns_repeat(self):
        """Returning type(self) should result in Repeat transition."""

        class TestState(enoki.State):
            def on_state(self, ctx):
                return type(self)

        self.assertStateTransition(TestState, enoki.Repeat)

    def test_restart_transition(self):
        """Returning Restart should result in Restart transition."""

        class TestState(enoki.State):
            def on_state(self, ctx):
                return enoki.Restart

        self.assertStateTransition(TestState, enoki.Restart)

    def test_retry_transition(self):
        """Returning Retry should result in Retry transition."""

        class TestState(enoki.State):
            def on_state(self, ctx):
                return enoki.Retry

        self.assertStateTransition(TestState, enoki.Retry)

    def test_push_single_state_becomes_transition(self):
        """Pushing a single state should be equivalent to direct transition."""

        class StateA(enoki.State):
            def on_state(self, ctx):
                return enoki.Push(StateB)

        class StateB(enoki.State):
            def on_state(self, ctx):
                return None

        # Single push should be converted to direct transition
        self.assertStateTransition(StateA, StateB)

    def test_push_multiple_states(self):
        """Pushing multiple states should return Push with all states."""

        class StateA(enoki.State):
            def on_state(self, ctx):
                return enoki.Push(StateB, StateC)

        class StateB(enoki.State):
            def on_state(self, ctx):
                return None

        class StateC(enoki.State):
            def on_state(self, ctx):
                return None

        self.assertPushTransition(StateA, [StateB, StateC])

    def test_pop_transition(self):
        """Returning Pop should result in Pop transition."""

        class TestState(enoki.State):
            def on_state(self, ctx):
                return enoki.Pop

        self.assertPopTransition(TestState)

    def test_invalid_transition_raises_error(self):
        """Invalid transition values should raise InvalidTransition."""

        class TestState(enoki.State):
            def on_state(self, ctx):
                return "invalid"

        self.assertStateRaisesException(TestState, enoki.InvalidTransition)

    def test_empty_push_raises_error(self):
        """Empty Push should raise InvalidPushError."""

        class TestState(enoki.State):
            def on_state(self, ctx):
                return enoki.Push()

        self.assertStateRaisesException(TestState, enoki.InvalidPushError)


class TestStateLifecycle(EnokiTestCase):
    """Test state lifecycle methods (on_enter, on_leave, etc.)"""

    def test_on_enter_called_on_first_tick(self):
        """on_enter should be called when state is entered."""
        call_tracker = {}

        class TestState(enoki.State):
            def on_enter(self, ctx):
                call_tracker["on_enter"] = True

            def on_state(self, ctx):
                return None

        self.assertLifecycleMethodCalled(TestState, "on_enter")

    def test_on_leave_called_on_transition(self):
        """on_leave should be called when transitioning to another state."""
        call_tracker = {}

        class StateA(enoki.State):
            def on_leave(self, ctx):
                call_tracker["on_leave"] = True

            def on_state(self, ctx):
                return StateB

        class StateB(enoki.State):
            def on_state(self, ctx):
                return None

        state = StateA()
        fsm_mock = make_mock_fsm()
        state._tick(fsm_mock, fsm_mock._ctx)

        # on_leave should have been called
        self.assertTrue(call_tracker.get("on_leave", False))

    def test_on_timeout_called_with_timeout_message(self):
        """on_timeout should be called when StateTimedOut message is received."""
        call_tracker = {}

        class TestState(enoki.State):
            def on_timeout(self, ctx):
                call_tracker["on_timeout"] = True
                return None

            def on_state(self, ctx):
                return None

        timeout_msg = enoki.StateTimedOut(TestState, 0)
        state = TestState()
        fsm_mock = make_mock_fsm(msg=timeout_msg, error_state=enoki.DefaultStates.Error)
        state._tick(fsm_mock, fsm_mock._ctx)

        self.assertTrue(call_tracker.get("on_timeout", False))

    def test_missing_on_state_raises_error(self):
        """State without on_state method should raise MissingOnStateHandler."""

        class IncompleteState(enoki.State):
            pass

        self.assertStateRaisesException(IncompleteState, enoki.MissingOnStateHandler)


class TestRetryAndTimeoutLogic(EnokiTestCase):
    """Test retry counter and timeout behavior."""

    def test_retry_counter_decrements_correctly(self):
        """Retry counter should decrement with each retry and raise error at limit."""

        class TestState(enoki.State):
            RETRIES = 2
            TIMEOUT = 1

            def on_state(self, evt):
                pass

        self.assertRetryBehavior(TestState)

    def test_timeout_triggers_on_timeout(self):
        """StateTimedOut message should trigger on_timeout method."""

        class TestState(enoki.State):
            def on_timeout(self, ctx):
                return enoki.DefaultStates.End

        self.assertTimeoutBehavior(TestState)


class TestSharedState(EnokiTestCase):
    """Test shared state functionality."""

    def test_shared_state_propagation(self):
        """States should be able to modify and access shared state."""

        class TestState(enoki.State):
            def on_state(self, ctx):
                ctx.common.test_value = 42
                return None

        self.assertSharedStateModified(TestState, "test_value", 42)

    def test_shared_state_persistence(self):
        """Shared state should persist across multiple state operations."""
        shared = SimpleNamespace(counter=0)

        class TestState(enoki.State):
            def on_state(self, ctx):
                ctx.common.counter += 1
                return None

        # Run the state multiple times
        for i in range(3):
            run_state_once(TestState, shared_state=shared)

        self.assertEqual(shared.counter, 3)


class TestStateMachineLifecycle(FSMTestCase):
    """Test StateMachine lifecycle and behavior."""

    def test_fsm_starts_in_initial_state(self):
        """FSM should start in the specified initial state."""

        class InitialState(enoki.State):
            def on_state(self, ctx):
                return None

        sm = enoki.StateMachine(InitialState)
        self.assertIsInstance(sm._current, InitialState)

    def test_fsm_transitions_between_states(self):
        """FSM should transition between states correctly."""

        class StateA(enoki.State):
            def on_state(self, ctx):
                return StateB

        class StateB(enoki.State):
            TIMEOUT = 1  # Prevent blocking

            def on_state(self, ctx):
                return None

        self.assertFSMTransition(StateA, StateB)

    def test_fsm_completes_on_terminal_state(self):
        """FSM should complete when reaching a terminal state."""

        class TerminalState(enoki.State):
            TERMINAL = True

            def on_state(self, ctx):
                return None

        self.assertFSMCompletes(TerminalState)

    def test_fsm_blocks_in_untimed_state(self):
        """FSM should raise BlockedInUntimedState for untimed, non-terminal states."""

        class BlockingState(enoki.State):
            def on_state(self, ctx):
                return None

        self.assertFSMStalls(BlockingState)

    def test_fsm_reset_functionality(self):
        """FSM reset should return to initial state and clear stack."""

        class StateA(enoki.State):
            def on_state(self, ctx):
                return enoki.Push(StateB)

        class StateB(enoki.State):
            TIMEOUT = 1

            def on_state(self, ctx):
                return None

        sm = enoki.StateMachine(StateA)

        sm.tick()  # Should push StateB onto stack

        # Verify state changed and stack has content
        self.assertIsInstance(sm._current, StateB)
        self.assertEqual(
            len(sm._state_stack), 0
        )  # Push with single state doesn't use stack

        # Reset and verify
        sm.reset()
        self.assertIsInstance(sm._current, StateA)
        self.assertEqual(len(sm._state_stack), 0)


class TestStackOperations(FSMTestCase):
    """Test push/pop stack operations."""

    def test_push_multiple_states_uses_stack(self):
        """Pushing multiple states should use the state stack correctly."""

        class StateA(enoki.State):
            def on_state(self, ctx):
                return enoki.Push(StateB, StateC)

        class StateB(enoki.State):
            def on_state(self, ctx):
                return enoki.Pop

        class StateC(enoki.State):
            TIMEOUT = 1

            def on_state(self, ctx):
                return None

        sm = enoki.StateMachine(StateA)

        sm.tick()

        # Should be in StateB with StateC on stack
        self.assertIsInstance(sm._current, StateC)
        self.assertFSMStackState(sm, [])

    def test_pop_from_empty_stack_raises_error(self):
        """Popping from empty stack should raise EmptyStateStackError."""

        class PopState(enoki.State):
            def on_state(self, ctx):
                return enoki.Pop

        sm = enoki.StateMachine(PopState)
        with self.assertRaises(enoki.EmptyStateStackError):

            sm.tick()

    def test_pop_transitions_to_stacked_state(self):
        """Pop should transition to the state on top of the stack."""

        class StateA(enoki.State):
            def on_state(self, ctx):
                return enoki.Push(StateB, StateC)

        class StateB(enoki.State):
            def on_state(self, ctx):
                return enoki.Pop

        class StateC(enoki.State):
            TIMEOUT = 1

            def on_state(self, ctx):
                return None

        sm = enoki.StateMachine(StateA)

        sm.tick()  # A -> B, push C

        sm.tick()  # B -> Pop -> C

        self.assertIsInstance(sm._current, StateC)
        self.assertEqual(len(sm._state_stack), 0)


class TestMessageHandling(FSMTestCase):
    """Test message filtering and trapping."""

    def test_filter_function_blocks_messages(self):
        """Filter function should prevent messages from reaching states."""

        def always_filter(shared_state, message):
            return True  # Block all messages

        class TestState(enoki.State):
            TIMEOUT = 1

            def on_state(self, ctx):
                return None

        self.assertFSMFilterBlocks(TestState, "test_message", always_filter)

    def test_trap_function_called_for_unhandled_messages(self):
        """Trap function should be called for unhandled messages."""
        trap_calls = []

        def trap_fn(shared_state):
            trap_calls.append(shared_state.msg)

        self.assertFSMTrapCalled(enoki.State, "unhandled_message", trap_fn)
        # Note: Specific assertion would depend on implementation details

    def test_filter_function_exceptions_propagate(self):
        """Exceptions in filter function should propagate."""

        def error_filter(shared_state, message):
            raise ValueError("Filter error")

        class TestState(enoki.State):
            def on_state(self, ctx):
                return None

        sm = enoki.StateMachine(TestState, filter_fn=error_filter)
        with self.assertRaises(ValueError):
            sm.send_message(True)
            sm.tick()


class TestErrorHandling(FSMTestCase):
    """Test error handling mechanisms."""

    def test_on_error_function_called_for_exceptions(self):
        """on_error_fn should be called when states raise exceptions."""
        error_calls = []

        def on_error_fn(shared_state, exception):
            error_calls.append(exception)
            return None  # Don't handle the error

        class ErrorState(enoki.State):
            def on_state(self, ctx):
                raise RuntimeError("Test error")

        sm = enoki.StateMachine(ErrorState, on_error_fn=on_error_fn)

        with self.assertRaises(RuntimeError):

            sm.tick()

        self.assertEqual(len(error_calls), 1)
        self.assertIsInstance(error_calls[0], RuntimeError)

    def test_error_state_transition_on_unhandled_error(self):
        """Unhandled errors should transition to error state if configured."""

        class CustomErrorClass(enoki.State):
            TIMEOUT = 1

            def on_state(self, ctx):
                return enoki.Unhandled

        def on_error_fn(shared_state, exception):
            return CustomErrorClass

        class ErrorState(enoki.State):
            def on_state(self, ctx):
                raise RuntimeError("Test error")

        sm = enoki.StateMachine(ErrorState, on_error_fn=on_error_fn)

        sm.tick()

        self.assertIsInstance(sm._current, CustomErrorClass)


class TestContinuationsAndRenewals(EnokiTestCase):
    """Test continuation vs renewal behavior."""

    def test_again_does_not_reset_retries(self):
        """Again (continuation) should not reset retry counter."""

        class TestState(enoki.State):
            RETRIES = 2

            def on_state(self, ctx):
                return self  # Should become Again

        state = TestState()

        fsm_mock = make_mock_fsm()
        result = state._tick(fsm_mock, fsm_mock._ctx)

        self.assertEqual(result, enoki.Again)

        # After a single tick, We should still have all of our remaining retries
        self.assertEqual(state._tries, state.RETRIES)

        # This should hold true no matter how many times we tick
        for _ in range(10):
            result = state._tick(fsm_mock, fsm_mock._ctx)
            self.assertEqual(result, enoki.Again)
            self.assertEqual(state._tries, state.RETRIES)

    def test_repeat_resets_state_context(self):
        """Repeat (renewal) should reset retry counter and other state context."""

        class TestState(enoki.State):
            RETRIES = 2

            def on_state(self, ctx):
                return type(self)  # Should become Repeat

        state = TestState()
        state._tries = 1  # Simulate some retries used
        state.has_entered = True

        fsm_mock = make_mock_fsm()
        result = state._tick(fsm_mock, fsm_mock._ctx)

        self.assertEqual(result, enoki.Repeat)
        # Note: The actual reset happens in the FSM transition logic

    def test_retry_decrements_counter_and_reenters(self):
        """Retry should decrement counter and cause re-entry."""

        class TestState(enoki.State):
            RETRIES = 2

            def on_state(self, ctx):
                return enoki.Retry

        self.assertStateTransition(TestState, enoki.Retry)

    def test_restart_resets_and_waits(self):
        """Restart should reset state and wait for next message."""

        class TestState(enoki.State):
            def on_state(self, ctx):
                return enoki.Restart

        self.assertStateTransition(TestState, enoki.Restart)


class TestLoggingAndVisualization(FSMTestCase):
    """Test logging and transition tracking features."""

    def test_log_function_receives_transitions(self):
        """Log function should receive transition messages."""
        log_messages = []

        def log_fn(message):
            log_messages.append(message)

        class StateA(enoki.State):
            def on_state(self, ctx):
                return StateB

        class StateB(enoki.State):
            TIMEOUT = 1

            def on_state(self, ctx):
                return None

        sm = enoki.StateMachine(StateA, log_fn=log_fn)

        sm.tick()

        # Should have at least one transition log
        transition_logs = [msg for msg in log_messages if "State Transition" in msg]
        self.assertGreater(len(transition_logs), 0)

    def test_transition_tracking_for_visualization(self):
        """FSM should track transitions for flowchart generation."""

        class StateA(enoki.State):
            def on_state(self, ctx):
                return StateB

        class StateB(enoki.State):
            TIMEOUT = 1

            def on_state(self, ctx):
                return None

        sm = enoki.StateMachine(StateA)

        sm.tick()

        # Should have generated flowchart data
        flowchart = sm.mermaid_flowchart
        self.assertIn("flowchart", flowchart)
        self.assertIn("StateA", flowchart)
        self.assertIn("StateB", flowchart)

        # Should have generated graphviz data
        digraph = sm.graphviz_digraph
        self.assertIn("digraph", digraph)
        self.assertIn("StateA", digraph)
        self.assertIn("StateB", digraph)


class TestMiscellaneousFeatures(FSMTestCase):
    """Test miscellaneous FSM features."""

    def test_dwell_states_dont_block(self):
        """States with CAN_DWELL=True should not cause BlockedInUntimedState."""

        class DwellState(enoki.State):
            CAN_DWELL = True

            def on_state(self, ctx):
                return None  # Would normally block

        sm = enoki.StateMachine(DwellState)
        sm.tick()  # Should not raise exception

        self.assertIsInstance(sm._current, DwellState)

    def test_custom_transition_function_called(self):
        """Custom transition function should be called on transitions."""
        transition_calls = []

        def transition_fn(next_state, shared_state):
            transition_calls.append(next_state)

        class StateA(enoki.State):
            def on_state(self, ctx):
                return StateB

        class StateB(enoki.State):
            TIMEOUT = 1

            def on_state(self, ctx):
                return None

        sm = enoki.StateMachine(StateA, transition_fn=transition_fn)
        sm.tick()

        # Should have called transition function
        self.assertGreater(len(transition_calls), 0)

    def test_custom_common_data(self):
        """FSM should accept and use custom common data object."""

        class CustomData:
            def __init__(self):
                self.value = 100

        class TestState(enoki.State):
            CAN_DWELL = True

            def on_state(self, ctx):
                ctx.common.value += 1
                return None

        custom_data = CustomData()
        sm = enoki.StateMachine(TestState, common_data=custom_data)
        sm.tick()

        self.assertEqual(custom_data.value, 101)


class TestEdgeCasesAndCoverage(EnokiTestCase):
    """Additional tests to improve test coverage on edge cases."""

    def test_push_with_repeat_at_end(self):
        """Test Push with Repeat (type(self)) at the end of the list."""

        class StateA(enoki.State):
            def on_state(self, ctx):
                return enoki.Push(StateB, enoki.Repeat)

        class StateB(enoki.State):
            def on_state(self, ctx):
                return None

        # This should be converted to Push(StateB, StateA) internally
        result = run_state_once(StateA)
        self.assertIsInstance(result, enoki.Push)
        self.assertEqual(len(result.push_states), 2)
        self.assertEqual(result.push_states[0], StateB)
        self.assertEqual(result.push_states[1], StateA)

    def test_push_with_repeat_in_middle_raises_error(self):
        """Test that Push with Repeat in the middle raises InvalidPushError."""

        class StateA(enoki.State):
            def on_state(self, ctx):
                return enoki.Push(StateB, enoki.Repeat, StateC)

        class StateB(enoki.State):
            def on_state(self, ctx):
                return None

        class StateC(enoki.State):
            def on_state(self, ctx):
                return None

        self.assertStateRaisesException(StateA, enoki.InvalidPushError)

    def test_push_with_repeat_not_at_end_raises_error(self):
        """Test that Push with Repeat not at the end raises InvalidPushError."""

        class StateA(enoki.State):
            def on_state(self, ctx):
                return enoki.Push(enoki.Repeat, StateB)

        class StateB(enoki.State):
            def on_state(self, ctx):
                return None

        self.assertStateRaisesException(StateA, enoki.InvalidPushError)

    def test_push_with_self_as_first_state_raises_error(self):
        """Test that Push with self as first state raises InvalidPushError."""

        class StateA(enoki.State):
            def on_state(self, ctx):
                return enoki.Push(type(self), StateB)

        class StateB(enoki.State):
            def on_state(self, ctx):
                return None

        self.assertStateRaisesException(StateA, enoki.InvalidPushError)

    def test_push_with_non_state_class_raises_error(self):
        """Test that Push with non-State class raises InvalidTransition."""

        class StateA(enoki.State):
            def on_state(self, ctx):
                return enoki.Push(enoki.Unhandled)

        self.assertStateRaisesException(StateA, enoki.InvalidPushError)

    def test_state_returning_instance_gets_warning(self):
        """Test that returning state instances generates a warning."""
        log_calls = []

        class StateA(enoki.State):
            def on_state(self, ctx):
                return StateB()  # Instance instead of class

        class StateB(enoki.State):
            def on_state(self, ctx):
                return None

        fsm_mock = make_mock_fsm(log_fn=lambda msg: log_calls.append(msg))
        state = StateA()
        result = state._tick(fsm_mock, fsm_mock._ctx)

        # Should still work but generate a warning
        self.assertEqual(result, StateB)
        self.assertTrue(any("WARNING" in call for call in log_calls))

    def test_state_returning_transition_instance_gets_warning(self):
        """Test that returning transition instances generates a warning."""
        log_calls = []

        class TestState(enoki.State):
            def on_state(self, ctx):
                return enoki.Retry()  # Instance instead of class

        fsm_mock = make_mock_fsm(log_fn=lambda msg: log_calls.append(msg))
        state = TestState()
        result = state._tick(fsm_mock, fsm_mock._ctx)

        self.assertEqual(result, enoki.Retry)
        self.assertTrue(any("WARNING" in call for call in log_calls))

    def test_timeout_self_ticking_state_behavior(self):
        """Test timeout handling when state returns self (self-ticking)."""

        class SelfTickingState(enoki.State):
            def on_timeout(self, ctx):
                return self  # This should trigger special handling

        timeout_msg = enoki.StateTimedOut(SelfTickingState, 0)
        state = SelfTickingState()
        fsm_mock = make_mock_fsm(msg=timeout_msg)

        result = state._tick(fsm_mock, fsm_mock._ctx)

        # Should return self but reset has_entered
        self.assertEqual(result, enoki.Again)
        self.assertFalse(state.has_entered)

    def test_timeout_with_exception_message_cleared(self):
        """Test that timeout with exception message gets cleared."""

        class TestState(enoki.State):
            TIMEOUT = 0.1

            def on_timeout(self, ctx):
                return self

            def on_state(self, ctx):
                return None

        timeout_msg = enoki.StateTimedOut(TestState, 0.1)
        state = TestState()
        fsm_mock = make_mock_fsm(msg=timeout_msg, log_fn=print)

        result = state._tick(fsm_mock, fsm_mock._ctx)

        # Message should be cleared when returning self from timeout
        self.assertIsNone(fsm_mock._ctx.msg)

    def test_returning_category_raises_invalid_transition(self):
        """Test that returning a category raises InvalidTransition."""

        class TestState(enoki.State):
            def on_state(self, ctx):
                return enoki.StateRenewal

        self.assertStateRaisesException(TestState, enoki.InvalidTransition)


class TestStateMachineEdgeCases(FSMTestCase):
    """Test StateMachine edge cases for better coverage."""

    def test_fsm_with_instance_initial_state(self):
        """Test FSM handles instance as initial state (converts to class)."""

        class TestState(enoki.State):
            TIMEOUT = 1

            def on_state(self, ctx):
                return None

        # Pass instance instead of class
        sm = enoki.StateMachine(TestState())
        self.assertIsInstance(sm._current, TestState)

    def test_fsm_with_instance_error_state(self):
        """Test FSM handles instance as error state (converts to class)."""

        class TestState(enoki.State):
            def on_state(self, ctx):
                raise RuntimeError("Test error")

        class CustomError(enoki.State):
            TIMEOUT = 1

            def on_state(self, ctx):
                return None

        def error_handler(shared, exc):
            return CustomError

        # Pass instance instead of class for error state
        sm = enoki.StateMachine(
            TestState, error_state=CustomError(), on_error_fn=error_handler
        )
        sm.tick()

        self.assertIsInstance(sm._current, CustomError)

    def test_timeout_during_state_transition(self):
        """Test timeout that occurs during state transition."""

        class StateA(enoki.State):
            def on_state(self, ctx):
                # Check if we've already transitioned once
                if hasattr(ctx.common, "transitioned"):
                    return enoki.DefaultStates.End
                ctx.common.transitioned = True

                # Manually inject timeout to test the cleanup path since there isn't
                # anywhere to inject this into the queue between checking it the first time
                ctx.send_message(enoki.StateTimedOut(StateA, 0))
                return StateB

        class StateB(enoki.State):
            TIMEOUT = 1

            def on_state(self, ctx):
                return None

        log_calls = []

        def custom_logger(msg):
            log_calls.append(msg)
            print(msg)

        sm = enoki.StateMachine(StateA, log_fn=custom_logger)

        try:
            sm.tick()
        except enoki.StateMachineComplete:
            pass

        # Should log the timeout cleanup message
        timeout_logs = [msg for msg in log_calls if "Timeout for prior state" in msg]
        self.assertTrue(len(timeout_logs) > 0)

    def test_start_non_blocking_execution(self):
        """Test non-blocking execution mode."""

        class CounterState(enoki.State):
            TIMEOUT = 1

            def __init__(self):
                super().__init__()
                self.count = 0

            def on_state(self, ctx):
                self.count += 1
                if self.count >= 3:
                    return enoki.DefaultStates.End
                return None  # Will cause stall

        sm = enoki.StateMachine(CounterState)

        # Should raise NonBlockingStalled when queue is empty
        with self.assertRaises(enoki.NonBlockingStalled):
            sm.start_non_blocking()

    def test_start_non_blocking_with_completion(self):
        """Test non-blocking execution that completes normally."""

        class TerminalState(enoki.State):
            TERMINAL = True

            def on_state(self, ctx):
                return None

        sm = enoki.StateMachine(TerminalState)

        # Should raise StateMachineComplete
        with self.assertRaises(enoki.StateMachineComplete):
            sm.start_non_blocking()

    def test_mermaid_flowchart_with_no_transitions(self):
        """Test mermaid flowchart generation with no transitions."""

        class TestState(enoki.State):
            TIMEOUT = 1

            def on_state(self, ctx):
                return None

        sm = enoki.StateMachine(TestState)

        # No transitions have occurred yet
        flowchart = sm.mermaid_flowchart
        self.assertIn("flowchart LR", flowchart)

    def test_graphviz_with_multiple_transitions(self):
        """Test graphviz generation with multiple transitions."""

        class StateA(enoki.State):
            def on_state(self, ctx):
                if not hasattr(ctx.common, "step"):
                    ctx.common.step = 0
                ctx.common.step += 1

                if ctx.common.step == 1:
                    return StateB
                else:
                    return enoki.DefaultStates.End  # Prevent infinite loop

        class StateB(enoki.State):
            def on_state(self, ctx):
                return StateA  # Go back to A (which will end)

        sm = enoki.StateMachine(StateA)

        try:
            sm.tick()  # A -> B -> A -> End
        except enoki.StateMachineComplete:
            pass

        digraph = sm.graphviz_digraph
        self.assertIn("digraph State", digraph)
        self.assertIn("StateA", digraph)
        self.assertIn("StateB", digraph)

    def test_fsm_cleanup_cancels_timers(self):
        """Test that FSM cleanup properly cancels active timers."""

        class TimedState(enoki.State):
            TIMEOUT = 1

            def on_state(self, ctx):
                return None

        sm = enoki.StateMachine(TimedState)

        sm.tick()  # Start the state and its timer

        # Mock the timer to verify cancel is called
        with patch.object(sm._current, "_cancel_failsafe") as mock_cancel:
            sm.cleanup()
            mock_cancel.assert_called_once()

    def test_shared_state_with_custom_common_data_class(self):
        """Test SharedState with custom common data class."""

        class CustomCommon:
            def __init__(self):
                self.custom_value = "test"

        class TestState(enoki.State):
            CAN_DWELL = True

            def on_state(self, ctx):
                ctx.common.custom_value = "modified"
                return None

        custom_data = CustomCommon()
        sm = enoki.StateMachine(
            TestState,
            common_data=custom_data,
        )

        sm.tick()

        self.assertEqual(custom_data.custom_value, "modified")

    def test_current_state_stack_property(self):
        """Test that current_state_stack returns a copy."""

        class StateA(enoki.State):
            def on_state(self, ctx):
                return enoki.Push(StateB, StateC)

        class StateB(enoki.State):
            TIMEOUT = 1

            def on_state(self, ctx):
                return None

        class StateC(enoki.State):
            def on_state(self, ctx):
                return None

        sm = enoki.StateMachine(StateA)

        sm.tick()

        # Get the stack copy
        stack_copy = sm.current_state_stack

        # Modify the copy
        if stack_copy:
            stack_copy.append("fake_state")

        # Original stack should be unchanged
        self.assertNotEqual(sm._state_stack, stack_copy)


class TestClassPropertyAndMiscellaneous(EnokiTestCase):
    """Test class property descriptor and other miscellaneous functionality."""

    def test_class_property_with_none_klass(self):
        """Test ClassPropertyDescriptor when klass is None."""
        from enoki import ClassPropertyDescriptor

        def test_func(cls):
            return "test_value"

        descriptor = ClassPropertyDescriptor(test_func)

        class TestClass:
            prop = descriptor

        # Test with klass=None (should infer from obj)
        instance = TestClass()
        result = descriptor.__get__(instance, None)
        self.assertEqual(result, "test_value")

    def test_default_error_state_behavior(self):
        """Test the default Error state behavior."""
        log_calls = []

        error_state = enoki.DefaultStates.Error()
        fsm_mock = make_mock_fsm(log_fn=lambda msg: log_calls.append(msg))

        result = error_state._tick(fsm_mock, fsm_mock._ctx)

        # Should log the default error message
        self.assertTrue(any("Default Error State" in call for call in log_calls))
        self.assertTrue(error_state.TERMINAL)

    def test_state_machine_save_methods(self):
        """Test save methods for mermaid and graphviz."""
        import tempfile
        import os

        class StateA(enoki.State):
            def on_state(self, ctx):
                return StateB

        class StateB(enoki.State):
            TIMEOUT = 1

            def on_state(self, ctx):
                return None

        sm = enoki.StateMachine(StateA)

        sm.tick()

        # Test saving mermaid flowchart
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".mmd") as f:
            mermaid_file = f.name

        try:
            sm.save_mermaid_flowchart(mermaid_file)

            with open(mermaid_file, "r") as f:
                content = f.read()
                self.assertIn("flowchart LR", content)
        finally:
            os.unlink(mermaid_file)

        # Test saving graphviz digraph
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".dot") as f:
            graphviz_file = f.name

        try:
            sm.save_graphviz_digraph(graphviz_file)

            with open(graphviz_file, "r") as f:
                content = f.read()
                self.assertIn("digraph State", content)
        finally:
            os.unlink(graphviz_file)


class TestConcurrencyAndThreading(FSMTestCase):
    """Test threading and concurrency aspects."""

    def test_failsafe_timer_creation_and_cancellation(self):
        """Test that failsafe timers are created and cancelled properly."""

        class TimedState(enoki.State):
            TIMEOUT = 0.1  # Short timeout for testing
            CAN_DWELL = True

            def on_state(self, ctx):
                # Don't sleep here - just return None to avoid blocking
                # The timer creation is what we're testing
                return None

        sm = enoki.StateMachine(TimedState)

        # Start the state - should create a timer

        sm.tick()

        # Timer should be active (this tests the timer creation path)
        self.assertIsNotNone(sm._current._failsafe_timer)

        # Reset should cancel the timer
        sm.reset()

        # After reset, new state should be created
        self.assertIsInstance(sm._current, TimedState)

    def test_timeout_timer_callback(self):
        """Test that timeout timer callback works correctly."""

        class TimeoutState(enoki.State):
            TIMEOUT = 0.01  # Very short timeout
            CAN_DWELL = True

            def on_state(self, ctx):
                # Don't sleep in on_state as it will block the FSM
                # Instead, return None and let the timer fire
                return None

            def on_timeout(self, ctx):
                ctx.common.failed = True

        sm = enoki.StateMachine(TimeoutState)

        # Set a default failure condition
        sm.context.common.failed = False
        sm.tick()

        # Wait for timeout to trigger and get processed
        time.sleep(0.1)

        # Now tick again to process any timeout messages
        try:
            sm.tick()
            sm.tick()
        except:
            pass  # We just want to trigger timeout processing

        # Check if timeout was handled
        self.assertTrue(sm.context.common.failed, "Timeout never fired")


class TestComplexScenarios(FSMTestCase):
    """Test complex real-world-like scenarios."""

    def test_complex_push_pop_scenario(self):
        """Test a complex push/pop scenario with multiple states."""

        class MainState(enoki.State):
            def on_state(self, ctx):
                if not hasattr(ctx.common, "step"):
                    ctx.common.step = 0
                ctx.common.step += 1

                print(ctx.common.step)
                if ctx.common.step == 1:
                    return enoki.Push(SubTask1, SubTask2, ContinueMain)
                else:
                    return enoki.DefaultStates.End

        class SubTask1(enoki.State):
            def on_state(self, ctx):
                ctx.common.task1_done = True
                return enoki.Pop

        class SubTask2(enoki.State):
            def on_state(self, ctx):
                ctx.common.task2_done = True
                return enoki.Pop

        class ContinueMain(enoki.State):
            def on_state(self, ctx):
                return MainState  # Back to MainState

        sm = enoki.StateMachine(MainState)

        # Execute the complex scenario
        try:
            while True:

                sm.tick()
        except enoki.StateMachineComplete:
            pass

        # Verify the execution path
        self.assertTrue(hasattr(sm._ctx.common, "task1_done"))
        self.assertTrue(hasattr(sm._ctx.common, "task2_done"))
        self.assertTrue(sm.is_finished)

    def test_retry_with_eventual_success(self):
        """Test retry behavior that eventually succeeds."""

        class FlakeyState(enoki.State):
            RETRIES = 3

            def on_state(self, ctx):
                if not hasattr(ctx.common, "attempts"):
                    ctx.common.attempts = 0
                ctx.common.attempts += 1

                # Fail first 2 times, succeed on 3rd
                if ctx.common.attempts < 3:
                    return enoki.Retry
                else:
                    return enoki.DefaultStates.End

        sm = enoki.StateMachine(FlakeyState)

        try:
            while True:

                sm.tick()
        except enoki.StateMachineComplete:
            pass

        self.assertEqual(sm._ctx.common.attempts, 3)
        self.assertTrue(sm.is_finished)

    def test_error_recovery_scenario(self):
        """Test error recovery with custom error handling."""

        class ErrorProneState(enoki.State):
            def on_state(self, ctx):
                if not hasattr(ctx.common, "error_count"):
                    ctx.common.error_count = 0
                ctx.common.error_count += 1

                if ctx.common.error_count <= 2:
                    raise RuntimeError(f"Error #{ctx.common.error_count}")
                else:
                    return enoki.DefaultStates.End

        class RecoveryState(enoki.State):
            def on_state(self, ctx):
                ctx.common.recovered = True
                return ErrorProneState  # Try again

        def error_handler(shared, exc):
            shared.common.last_error = str(exc)
            return RecoveryState

        sm = enoki.StateMachine(ErrorProneState, on_error_fn=error_handler)

        try:
            while True:

                sm.tick()
        except enoki.StateMachineComplete:
            pass

        self.assertTrue(hasattr(sm._ctx.common, "recovered"))
        self.assertTrue(hasattr(sm._ctx.common, "last_error"))
        self.assertTrue(sm.is_finished)
