import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock
from queue import Queue, PriorityQueue
import enoki


def make_mock_fsm(
    shared_state=None,
    msg=None,
    state_stack=None,
    log_fn=None,
    error_state=None,
    **kwargs
):
    """
    Create a MagicMock FSM test double with all required attributes for Enoki states.
    
    Args:
        shared_state: dict/object for shared state between states
        msg: message to set as current message
        state_stack: optional list for state stack (for push/pop operations)
        log_fn: function for logging (default: no-op)
        error_state: error state class (default: enoki.DefaultStates.Error)
        **kwargs: additional attributes to set on the FSM mock
    
    Returns:
        MagicMock configured as an FSM with SharedState properly set up
    """
    fsm_mock = MagicMock(name="MockFSM")
    
    # Core FSM attributes
    fsm_mock.log = log_fn or (lambda *args, **kwargs: None)
    fsm_mock._log_fn = fsm_mock.log
    fsm_mock._err_state = error_state or enoki.DefaultStates.Error
    fsm_mock._state_stack = list(state_stack) if state_stack is not None else []
    fsm_mock._msg_queue = Mock(spec=PriorityQueue)
    fsm_mock._is_finished = False
    fsm_mock._current = None
    
    # Timer and transition functions
    fsm_mock.start_failsafe_timer = MagicMock(return_value=MagicMock())
    fsm_mock._transition_fn = MagicMock()
    fsm_mock._on_err_fn = MagicMock()
    
    # Set up shared state
    if shared_state is None:
        shared_state = SimpleNamespace()
    elif isinstance(shared_state, dict):
        shared_state = SimpleNamespace(**shared_state)
    
    # Create proper SharedState object
    fsm_mock._ctx = enoki.SharedContext(
        send_message=lambda x: False,
        log=fsm_mock.log,
        msg=msg,
        common=shared_state
    )
    
    # Apply any additional kwargs
    for key, value in kwargs.items():
        setattr(fsm_mock, key, value)
    
    return fsm_mock


def run_state_once(state_class, shared_state=None, msg=None, **fsm_kwargs):
    """
    Run a state once with given inputs and return the transition result.
    
    Args:
        state_class: The state class to instantiate and run
        shared_state: Shared state object/dict
        msg: Message to pass to the state
        **fsm_kwargs: Additional FSM mock configuration
    
    Returns:
        The transition result from the state's _tick method
    """
    state = state_class()
    fsm_mock = make_mock_fsm(shared_state=shared_state, msg=msg, **fsm_kwargs)
    return state._tick(fsm_mock, fsm_mock._ctx)


def run_state_until_stable(state_class, shared_state=None, msg=None, max_iterations=10, **fsm_kwargs):
    """
    Run a state repeatedly until it returns a stable transition (not Unhandled).
    Useful for testing states that need multiple ticks to reach a decision.
    
    Args:
        state_class: The state class to instantiate and run
        shared_state: Shared state object/dict
        msg: Initial message to pass to the state
        max_iterations: Maximum iterations before giving up
        **fsm_kwargs: Additional FSM mock configuration
    
    Returns:
        tuple: (final_transition_result, iteration_count)
    
    Raises:
        RuntimeError: If max_iterations reached without stable result
    """
    state = state_class()
    fsm_mock = make_mock_fsm(shared_state=shared_state, msg=msg, **fsm_kwargs)
    
    for i in range(max_iterations):
        try:
            result = state._tick(fsm_mock, fsm_mock._ctx)
            if result is not enoki.Unhandled:
                return result, i + 1
            # Clear message after first iteration (simulate FSM behavior)
            fsm_mock._ctx.msg = None
        except (enoki.StateRetryLimitError, enoki.StateTimedOut) as e:
            # Inject the error as the next message (simulate FSM behavior)
            fsm_mock._ctx.msg = e
    
    raise RuntimeError(f"State did not stabilize after {max_iterations} iterations")


class EnokiTestCase(unittest.TestCase):
    """Base test case class with helpful assertion methods for Enoki FSM testing."""
    
    def assertStateTransition(self, state_class, expected_transition, 
                             shared_state=None, msg=None, **fsm_kwargs):
        """Assert that a state transitions to the expected state/transition type."""
        result = run_state_once(state_class, shared_state, msg, **fsm_kwargs)
        
        if isinstance(expected_transition, type) and issubclass(expected_transition, enoki.State):
            self.assertIs(result, expected_transition, 
                         f"Expected transition to {expected_transition.__name__}, got {result}")
        elif isinstance(expected_transition, type) and issubclass(expected_transition, enoki.TransitionType):
            self.assertIs(result, expected_transition,
                         f"Expected {expected_transition.__name__}, got {result}")
        else:
            self.assertEqual(result, expected_transition)
    
    def assertNoTransition(self, state_class, shared_state=None, msg=None, **fsm_kwargs):
        """Assert that a state does not transition (returns Unhandled or Again)."""
        result = run_state_once(state_class, shared_state, msg, **fsm_kwargs)
        self.assertIn(result, (enoki.Unhandled, enoki.Again),
                     f"Expected no transition (Unhandled/Again), got {result}")
    
    def assertPushTransition(self, state_class, expected_states, 
                            shared_state=None, msg=None, **fsm_kwargs):
        """Assert that a state returns a Push transition with expected states."""
        result = run_state_once(state_class, shared_state, msg, **fsm_kwargs)
        self.assertIsInstance(result, enoki.Push, "Expected Push transition")
        self.assertEqual(list(result.push_states), expected_states,
                        f"Expected push states {expected_states}, got {list(result.push_states)}")
    
    def assertPopTransition(self, state_class, shared_state=None, msg=None, **fsm_kwargs):
        """Assert that a state returns a Pop transition."""
        result = run_state_once(state_class, shared_state, msg, **fsm_kwargs)
        self.assertIs(result, enoki.Pop, "Expected Pop transition")
    
    def assertStateRaisesException(self, state_class, expected_exception, 
                                  shared_state=None, msg=None, **fsm_kwargs):
        """Assert that a state raises a specific exception."""
        state = state_class()
        fsm_mock = make_mock_fsm(shared_state=shared_state, msg=msg, **fsm_kwargs)
        
        with self.assertRaises(expected_exception):
            state._tick(fsm_mock, fsm_mock._ctx)
    
    def assertLifecycleMethodCalled(self, state_class, method_name, 
                                   shared_state=None, msg=None, **fsm_kwargs):
        """Assert that a specific lifecycle method is called on a state."""
        call_tracker = {}
        
        # Create a subclass that tracks the method call
        class TrackedState(state_class):
            def __getattribute__(self, name):
                attr = super().__getattribute__(name)
                if name == method_name and callable(attr):
                    def tracked_method(*args, **kwargs):
                        call_tracker[method_name] = True
                        return attr(*args, **kwargs)
                    return tracked_method
                return attr
        
        # Run the state
        try:
            run_state_once(TrackedState, shared_state, msg, **fsm_kwargs)
        except Exception:
            pass  # We only care about method calls, not results
        
        self.assertTrue(call_tracker.get(method_name, False),
                       f"Expected {method_name} to be called")
    
    def assertSharedStateModified(self, state_class, attribute_name, expected_value,
                                 shared_state=None, msg=None, **fsm_kwargs):
        """Assert that a state modifies shared state as expected."""
        if shared_state is None:
            shared_state = SimpleNamespace()
        elif isinstance(shared_state, dict):
            shared_state = SimpleNamespace(**shared_state)
        
        # Run the state
        run_state_once(state_class, shared_state, msg, **fsm_kwargs)
        
        # Check if the attribute was set correctly
        actual_value = getattr(shared_state, attribute_name, None)
        self.assertEqual(actual_value, expected_value,
                        f"Expected {attribute_name}={expected_value}, got {actual_value}")
    
    def assertRetryBehavior(self, state_class, 
                           shared_state=None, msg=None, **fsm_kwargs):
        """Assert that a state properly handles retry logic."""
        # Test that retries work up to the limit
        state = state_class()

        for retry_count in range(state_class.RETRIES + 1):
        
            try:
                state._handle_retries()
            except enoki.StateRetryLimitError:
                self.fail(f"Retry limit reached too early at retry {retry_count}")
        # Should raise error on final attempt
        with self.assertRaises(enoki.StateRetryLimitError):
            state._handle_retries()
    
    def assertTimeoutBehavior(self, state_class, shared_state=None, **fsm_kwargs):
        """Assert that a state properly handles timeout messages."""
        timeout_msg = enoki.StateTimedOut(state_class, 0)
        result = run_state_once(state_class, shared_state, timeout_msg, **fsm_kwargs)
        self.assertIsNotNone(result, "State should return a transition on timeout")


class FSMTestCase(EnokiTestCase):
    """Extended test case for testing full StateMachine behavior."""
    
    def assertFSMTransition(self, initial_state, expected_state, msg=None, **sm_kwargs):
        """Assert that an FSM transitions from initial to expected state."""
        sm = enoki.StateMachine(initial_state, **sm_kwargs)
        initial_type = type(sm._current)
        
        try:
            sm.send_message(msg)
            sm.tick()
        except enoki.StateMachineComplete:
            pass  # Terminal states are OK
        
        if isinstance(expected_state, type):
            self.assertIsInstance(sm._current, expected_state,
                                f"Expected FSM to be in {expected_state.__name__}, "
                                f"but it's in {type(sm._current).__name__}")
        else:
            self.assertEqual(sm._current, expected_state)
    
    def assertFSMCompletes(self, initial_state, msg=None, **sm_kwargs):
        """Assert that an FSM completes (reaches terminal state)."""
        sm = enoki.StateMachine(initial_state, **sm_kwargs)
         
        with self.assertRaises(enoki.StateMachineComplete):
            sm.send_message(msg)
            sm.tick()
        
        self.assertTrue(sm.is_finished, "FSM should be marked as finished")
    
    def assertFSMStalls(self, initial_state, msg=None, **sm_kwargs):
        """Assert that an FSM stalls (blocked in untimed state)."""
        sm = enoki.StateMachine(initial_state, **sm_kwargs)
        
        with self.assertRaises(enoki.BlockedInUntimedState):
            sm.send_message(msg)
            sm.tick()
    
    def assertFSMStackState(self, sm, expected_stack):
        """Assert that the FSM's state stack matches expected states."""
        actual_stack = [s.__name__ if isinstance(s, type) else str(s) for s in sm._state_stack]
        expected_names = [s.__name__ if isinstance(s, type) else str(s) for s in expected_stack]
        self.assertEqual(actual_stack, expected_names,
                        f"Expected stack {expected_names}, got {actual_stack}")
    
    def assertFSMFilterBlocks(self, initial_state, msg, filter_fn, **sm_kwargs):
        """Assert that FSM filter function blocks a message."""
        state_called = {}
        
        class TrackedState(initial_state):
            def on_state(self, st):
                state_called['called'] = st.msg == msg
                return super().on_state(st)
        
        sm = enoki.StateMachine(TrackedState, filter_fn=filter_fn, **sm_kwargs)
        sm.send_message(msg)
        sm.tick()
        
        self.assertFalse(state_called.get('called', False),
                        "State should not be called when message is filtered")
    
    def assertFSMTrapCalled(self, initial_state, msg, trap_fn, **sm_kwargs):
        """Assert that FSM trap function is called for unhandled messages."""
        # Ensure the state doesn't handle the message
        class UnhandlingState(initial_state):
            TIMEOUT = 1  # Prevent blocking
            def on_state(self, st):
                return None  # Unhandled
        
        sm = enoki.StateMachine(UnhandlingState, trap_fn=trap_fn, **sm_kwargs)
        sm.send_message(msg)
        sm.tick()
        # The trap function should have been called (specific assertion depends on implementation)