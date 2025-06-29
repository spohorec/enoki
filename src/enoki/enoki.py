#!/usr/bin/env python3
"""
Enoki provides a framework for creating and managing finite state machines (FSMs).
It includes classes for defining states, transitions, and managing shared state between
different states. The library supports features such as state retries, timeouts,
pushdown automata (state stack), and error handling.

Key Components:
- `State`: Base class for defining states. Users can extend this class to create custom states.
- `StateMachine`: Manages state transitions, shared state, and message queues.
- `SharedState`: Facilitates sharing information between states and the FSM.
- `TransitionType`: Base class for defining transition types (e.g., Push, Pop, Repeat, Retry).
- `DefaultStates`: Contains default states like `End`.
- `GenericCommon`: A container for shared state when no custom shared state is provided.

Features:
1. State Lifecycle:
    - `on_enter`: Called when entering a state.
    - `on_leave`: Called when leaving a state.
    - `on_state`: Main handler for state logic (required for all states).
    - `on_fail`: Called when a state exceeds its retry limit.
    - `on_timeout`: Called when a state exceeds its timeout duration.

2. Transition Types:
    - `Push`: Pushes states onto a stack and transitions to the first state.
    - `Pop`: Pops the top state off the stack and transitions to it.
    - `Repeat`: Re-enters the same state from the beginning.
    - `Retry`: Restarts the current state without transitioning.

3. Error Handling:
    - Custom handlers for various error conditions (e.g., Retry limits, timeouts).
    - Default error state for handling unhandled exceptions.

4. Timeout and Retry:
    - States can define `TIMEOUT` and `RETRIES` class variables for automatic timeout and retry handling.

5. Non-blocking Execution:
    - Supports non-blocking state machine execution with message queues.

6. Visualization:
    - Generates a Graphviz-compatible directed graph representation of state transitions.

Usage:
- Define custom states by inheriting from the `State` class and implementing required methods.
- Instantiate a `StateMachine` with initial, final, and error states.
- Use the `tick` method to process messages and transition between states.

Exceptions:
- `StateRetryLimitError`: Raised when a state exceeds its retry limit.
- `MissingOnStateHandler`: Raised when a state does not define an `on_state` handler.
- `EmptyStateStackError`: Raised when attempting to pop from an empty state stack.
- `NoPushedStatesError`: Raised when no states are provided to a `Push` transition.
- `NonBlockingStalled`: Raised when a non-blocking FSM stalls.
- `BlockedInUntimedState`: Raised when the FSM is blocked in a state without a timeout.

"""

import collections
from dataclasses import dataclass, field
from datetime import datetime
from queue import PriorityQueue, Empty
from threading import Timer
from typing import Any, Union, Optional, Type


class StateMachineComplete(Exception):
    pass


class MissingOnStateHandler(Exception):
    pass


class InvalidPushError(Exception):
    pass


class EmptyStateStackError(Exception):
    pass


class NoPushedStatesError(Exception):
    pass


class NonBlockingStalled(Exception):
    pass


class InvalidTransition(Exception):
    pass


class StateRetryLimitError(Exception):
    pass


class BlockedInUntimedState(Exception):
    def __init__(self, state):
        super().__init__(f"Blocking on state without a timer: {state.name}")
        self.state = state


class ClassPropertyDescriptor:
    def __init__(self, fget):
        self.fget = fget

    def __get__(self, obj, klass=None):
        if klass is None:
            klass = type(obj)
        return self.fget(klass)


@dataclass
class EnokiInternalMessage:
    state: "State"


@dataclass
class StateTimedOut(EnokiInternalMessage):
    timeout: int


def classproperty(func):
    return ClassPropertyDescriptor(func)


@dataclass(order=True)
class PrioritizedMessage:
    priority: int
    item: Any = field(compare=False)


# TODO: Maybe change AWAITS to IMMEDIATE?
@dataclass
class TransitionType:

    # A transition that awaits will require backing out to the FSM loop to await
    # an event or message
    AWAITS: bool = False

    @classproperty
    def name(cls):
        return cls.__name__

    @classproperty
    def base_name(cls):
        if hasattr(cls, "__bases__"):
            return cls.__bases__[0].name
        return cls.name


# Transition Categories


@dataclass
class StateTransition(TransitionType):
    """Represents a transition to a different state"""

    pass


@dataclass
class StateContinuation(TransitionType):
    """Represents staying in the same state."""

    pass


@dataclass
class StateRenewal(TransitionType):
    """Represents a transition back into the current state"""


# TODO: Rethink this name?
@dataclass
class Again(StateContinuation):
    """Executes the current state immediately from on_state without affecting
    the retry counter.

    NOTE: This is synonymous with returning `self`
    """

    pass


@dataclass
class Unhandled(StateContinuation):
    """Executes the current state from on_state without affecting the retry
    counter on the next event or message.

    NOTE: This is synonymous with returning `None`
    """

    AWAITS = True


@dataclass
class Retry(StateContinuation):
    """Retries the current state, decrementing it's
    retry counter and immediately beginning execution from on_enter."""

    pass


@dataclass
class Restart(StateRenewal):
    """Restarts the current state, resetting
    the retry counter, and awaiting a message or event to begin from
    on_enter."""

    AWAITS = True


@dataclass
class Repeat(StateRenewal):
    """Repeats the current state. The retry
    counter is reset, and the state immediately begins execution from  on_enter.

    NOTE: This is synonymous with returning `type(self)` OR the constructor for
    the current state (i.e it acts like any other state transition aside from NOT firing on_exit)
    """

    pass


class State(StateTransition):
    """
    The base class for all user-defined states in the Enoki FSM library.

    States are the workhorses of the FSM. To define a state, subclass State and implement at least the `on_state` method.
    Optionally, you may override `on_enter`, `on_leave`, `on_fail`, and `on_timeout` for custom entry/exit/failure/timeout behavior.

    ---
    **State Lifecycle Methods:**
      - `on_enter(shared)`: Called once when entering the state. Use for setup. Does not affect transitions.
      - `on_leave(shared)`: Called once when leaving the state. Use for cleanup. Does not affect transitions.
      - `on_state(shared)`: Main handler for state logic. Must be implemented. Controls transitions.
      - `on_fail(shared)`: Called when retry limit is reached. May return a new state or None.
      - `on_timeout(shared)`: Called when timeout is reached. May return a new state, self, or None.

    ---
    **Transition Return Values (from `on_state`, `on_fail`, `on_timeout`):**
      - `None`: No transition; FSM remains in current state and waits for next message ("wait").
      - `self`: Immediate continuation (Again); FSM re-enters current state from `on_state` (does not reset retries/timer).
      - `type(self)`: Renewal (Repeat); FSM restarts current state from `on_enter` (resets retries/timer).
      - `Restart`: FSM resets retry counter, re-enters state from `on_enter`, and waits for next message.
      - `Retry`: FSM decrements retry counter, re-enters state from `on_enter` immediately.
      - Another state class: FSM transitions to that state, calling `on_leave` and then new state's `on_enter`/`on_state`.
      - `Push(StateA, StateB, ...)`: FSM pushes states onto stack, transitions to first; supports pushdown automata.
      - `Pop`: FSM pops the top state from stack and transitions to it.

    ---
    **Timeouts and Retries:**
      - Set `TIMEOUT` (seconds) to enable automatic timeout and `on_timeout` handling. This is REQUIRED for states that might wait on a specific message
      - Set `RETRIES` (int) to enable retry limit and `on_fail` handling.
      - Returning self/type(self) from `on_timeout` resets timer/retries as appropriate.
      - Retry/timeout logic is handled automatically by the FSM.

    ---
    **Stack/Pushdown Automata:**
      - Use `Push` and `Pop` to implement stack-based state transitions.
      - Pushing a single state is equivalent to a direct transition.
      - Popping with an empty stack raises `EmptyStateStackError`.

    ---
    **Error Handling:**
      - Unhandled transitions raise `InvalidTransition`.
      - Missing `on_state` raises `MissingOnStateHandler`.
      - Retry/timeout errors raise `StateRetryLimitError`/`StateTimedOut`.
      - FSM can be configured with custom error states and error handlers.

    ---
    **Edge Cases and Best Practices:**
      - Continuations (Again) do not reset retries/timers; renewals (Repeat) do.
      - Shared state (`SharedState`) is passed to all methods for communication between states.
      - All transitions should return state classes, not instances (except Push).
      - Use `TERMINAL = True` to mark a state as terminal.

    See the Enoki documentation and test suite for more usage examples and edge case handling.
    """

    TIMEOUT = None
    RETRIES = None
    TERMINAL = False
    CAN_DWELL = False

    def __init__(self):
        self._tries = None
        self._failsafe_timer = None
        self._reset()

    def _reset(self):
        self._reset_retries()
        self._cancel_failsafe()
        self.has_entered = False

    def _reset_retries(self):
        if self.RETRIES is not None:
            self._tries = self.RETRIES + 1

    def on_state(self, ctx: "SharedContext"):
        """This method MUST be overriden by state implementations"""
        raise MissingOnStateHandler(f"State {self.name} has no on_state handler")

    def on_enter(self, ctx: "SharedContext"):
        pass

    def on_leave(self, ctx: "SharedContext"):
        pass

    def on_fail(self, ctx: "SharedContext"):
        pass

    def on_timeout(self, ctx: "SharedContext"):
        return self

    def _cancel_failsafe(self):
        if self._failsafe_timer:
            self._failsafe_timer.cancel()
            self._failsafe_timer = None

    def _start_failsafe(self, fsm: "StateMachine"):
        self._failsafe_timer = fsm._start_failsafe_timer(self.TIMEOUT)
        self._failsafe_timer.start()

    def _handle_retries(self):
        if self._tries is None:
            return
        elif self._tries == 0:
            raise StateRetryLimitError()
        else:
            self._tries -= 1

    def _maybe_failsafe_timer(self, fsm: "StateMachine"):
        if self.TIMEOUT:
            self._cancel_failsafe()
            self._start_failsafe(fsm)

    def _handle_timeout(self, fsm: "StateMachine", ctx:"SharedContext"):
        fsm.log(f"{self} in _handle_timeout")

        # TODO: Document this behavior w/r/t self timed states
        result = self.on_timeout(ctx) or fsm._err_state

        # If we return ourselves from a timeout, this means that this
        # is a 'self ticking' state and we need to check our retries
        # and reset our timer
        if result is not None and result.name == self.name:
            self.has_entered = False
            # Make sure that our timer is cancelled (in case out of order)
            self._cancel_failsafe()
            if isinstance(ctx.msg, StateTimedOut):
                ctx.msg = None
            # Force a tick
            fsm.send_message(None)

        return result

    def _tick(self, fsm: "StateMachine", ctx: "SharedContext") -> TransitionType:
        result = None

        if isinstance(ctx.msg, StateTimedOut):
            if ctx.msg.state.name != self.name:
                fsm.log(
                    f"Timeout for prior state {ctx.msg.state.name} fired during {self.name}, ignoring"
                )
            else:
                fsm.log(f"This timeout is for {self}, proceeding")
                result = self._handle_timeout(fsm, ctx)
        else:
            try:
                if not self.has_entered:
                    self._on_enter(fsm, ctx)
                result = self.on_state(ctx)
            except StateRetryLimitError:
                fsm.log(
                    f"State: {self.name} exceeded its maximum retry limit of {self.RETRIES} (retries {self._tries})"
                )
                result = self.on_fail(ctx) or fsm._err_state

        sanitized_result = self._sanitize_state_result(fsm, ctx, result)

        # If we're leaving for another state, we need to call on_leave
        if isinstance(sanitized_result, Push) or issubclass(
            sanitized_result, StateTransition
        ):
            self._on_leave(ctx)

        return sanitized_result

    def _on_enter(self, fsm: "StateMachine", ctx: "SharedContext"):
        self._handle_retries()

        self.on_enter(ctx)

        self._maybe_failsafe_timer(fsm)
        self.has_entered = True

    # TODO: Add warning for aliases except None

    def _sanitize_state_result(self, fsm: "StateMachine", ctx: "SharedContext", result: Any):
        # sanitize our return values
        match result:
            case None:
                # We alias None to Unhandled
                return Unhandled
            case s if s is self:
                # We alias 'self' to Again
                return Again
            case type() if result is type(self):
                # we alias type(self) to Repeat
                return Repeat
            case type() if result in (
                StateContinuation,
                StateTransition,
                StateRenewal,
                TransitionType,
            ):
                # We don't allow direct use of our categories or the base class
                raise InvalidTransition(
                    f"{result} is not a valid transition. Direct use of base base classes / categories is not allowed"
                )
            case type() if issubclass(
                result, (StateContinuation, StateRenewal, State, Pop)
            ):
                # Continuations/Renewals/Pop/States are passed on
                return result
            case t if isinstance(t, (StateContinuation, StateRenewal, Pop, State)):
                # Returning instances of Continuation/Renewal subtypes, and all
                # transitions aside from Push are accepted, we just buck them
                # back down to a constructor for consistency and warn the user
                fsm.log(
                    f"WARNING: {self.name} returned an instance of {type(t)}, please return a constructor (return State instead of State()) "
                )
                return type(t)
            case p if isinstance(p, Push):
                # Push is returned as an instance since we need the state it
                # contains, HOWEVER: there are some edge cases that we'll handle
                # here

                # We don't allow an empty push
                if not p.push_states:
                    raise InvalidPushError(
                        f"{self.name} attempted a push with no states provided"
                    )

                # We only support direct state transitions in Push
                # therefore we need to sanitize all of our substates, and then
                # verify that everything contained therein is valid
                sanitized_states = [
                    self._sanitize_state_result(fsm, ctx, x) for x in p.push_states
                ]

                # Now we need to ensure that:
                #
                # - Any repeats are transformed to type(self)
                # - All of the elements in the sanitized list are States
                # - type(self) ONLY shows up as the last element in the list

                # Repeat/type(self) allows us to do circular patterns like
                # Push(Foo, Bar, Baz, enoki.Repeat), coming back to the current
                # state after doing other things. We only allow this when
                # type(self) exists at the end of a list of > 1 states to avoid
                # accidental inescapable loops
                for idx, val in enumerate(sanitized_states):
                    if issubclass(val, Repeat):
                        sanitized_states[idx] = type(self)

                    if not issubclass(sanitized_states[idx], State):
                        raise InvalidPushError(
                            f"{self.name} attempted to push {val.name}. Only pushing States/Repeat is supported"
                        )

                    if issubclass(sanitized_states[idx], type(self)):
                        is_last = idx == len(sanitized_states) - 1
                        if idx == 0:
                            raise InvalidPushError(
                                f"{self.name} attempted to Push itself as the first state in a push stack"
                            )

                        if not is_last:
                            raise InvalidPushError(
                                f"{self.name} attempted to Push itself in a position that was not last in a series of pushed states"
                            )

                # A push of a single state is just an alias for a transition, so
                # we can avoid the push mechanism altogether
                if len(sanitized_states) == 1:
                    return sanitized_states[0]

                p.push_states = sanitized_states
                return p
            case _:
                raise InvalidTransition(f"{t} is not a valid transition")

    def _on_leave(self, ctx):
        self.on_leave(ctx)
        self._reset()


@dataclass(init=False)
class Push(StateTransition):
    """
    Represents a transition type where one or more states are pushed onto the
    stack.

    Pushing a single state is synonymous with a transition.

    Args:
        *push_states (List[Type[State]]): A variable number of state classes to
        be pushed onto the stack.
    """

    push_states: list[type[State]]

    def __init__(self, *push_states: list[type[State]]):
        self.push_states = push_states


@dataclass
class Pop(StateTransition):
    """Pops the top state off the stack and transitions to it."""

    pass


# End is a default state that any FSM can use
class DefaultStates:
    class End(State):
        TERMINAL = True

        def on_state(self, ctx):
            pass

    class Error(State):
        TERMINAL = True

        def on_state(self, ctx):
            ctx.log("Default Error State")
            pass


# TODO: Think about just making this optional and None, this feels silly
@dataclass
class GenericCommon:
    """This is an empty container class to hold any carry-over state in
    between FSM states should the user not provide a state container.

    """

    pass


# TODO: Rethink this name
@dataclass
class SharedContext:
    """SharedContesxt is passed to each state to allow states to share
    information downstream and access critical functionality like sending
    messages. The shared state object contains a reference to a common state
    object (SharedState.common), which may be user supplied during state machine
    instantiation.  It also contains the first message in the message queue (if
    any).

    """
    send_message: callable
    log: callable
    common: Any
    msg: Optional[Any] = None


# TODO: Test _do_nothing works as I hope it will...
_do_nothing = lambda x: None


class StateMachine:
    MESSAGE_PRIORITY = 2
    INTERNAL_MESSAGE_PRIORITY = 1
    ERROR_PRIORITY = 0
    """The StateMachine object is responsible for managing state
    transitions and bookkeeping shared state. On instantiation, the
    user must supply an initial state (which MUST be a subclass of enoki.State)

    Additionally, the user MAY supply a Queue object (msg_queue) which will be
    used to pass messages into states (and relay information about timeouts and
    retry failures between the states and FSM). If no msg_queue is provided
    a default queue.Queue().empty() is used.

    The user MAY supply filter and trap functions. filter allows the
    user to pre-screen messages that may be important to the state
    machine machine, but might not be necessary to transition a state.
    trap will capture any unhandled message and can be used to raise
    exceptions or log notification messages.

    The state machine will raise an error if it stops in a state that has
    neither a timeout nor is the final state unless it is included in an
    iterable called dwell_states.

    (For a visual overview of the data flow, see enoki_data_flow.png)

    Finally, the user MAY supply a common_data class instance. This
    will be passed into each state and can be used to propagate
    information between states. If no common_data class is provided,
    an empty 'GenericCommon' will be provided (which is simply an empty class)

    """

    def __init__(
        self,
        initial_state,
        error_state=DefaultStates.Error,
        filter_fn=None,
        trap_fn=None,
        on_error_fn=None,
        log_fn=print,
        transition_fn=None,
        common_data=None,
    ):

        # We want to make sure that initial/final/default_err states
        # are descriptors, not instances
        if isinstance(initial_state, State):
            initial_state = type(initial_state)

        if isinstance(error_state, State):
            error_state = type(error_state)

        self._initial_st = initial_state
        self._err_state = error_state
        self._msg_queue = PriorityQueue()

        self._log_fn = log_fn or _do_nothing
        self._transition_fn = transition_fn or (lambda x, y: None)
        self._on_err_fn = on_error_fn or (lambda x, y: None)

        # Used for pushdown states
        self._state_stack = []

        self._current = None
        self._finished = False

        self.reset_transitions()

        self._last_trans_time = datetime.now()

        # The filter and trap functions are used to filter messages
        # (for example, common messages that apply to the process
        # rather than an individual state) and trap unhandled messages
        # (so that one could, for example, raise an exception)
        self._filter_fn = filter_fn or (lambda x, y: None)
        self._trap_fn = trap_fn or _do_nothing

        self._ctx = SharedContext(
            send_message=self.send_message,
            log=self.log,
            common=common_data or GenericCommon()
        )

        self.reset()
        
    @property
    def context(self) -> SharedContext:
        return self._ctx

    def log(self, msg):
        self._log_fn(msg)

    def _start_failsafe_timer(self, duration):
        def _wrap_timeout(state, timeout):
            self._send_message_internal(StateTimedOut(state, timeout))

        return Timer(
            duration,
            lambda x, y: _wrap_timeout(x, y),
            args=[self._current, duration],
        )

    def reset_transitions(self):
        # We store transitions and times separately since we don't
        # want slightly different times to affect the set of actual transitions
        self._transition_id = 0 
        self._transitions = set()
        self._transition_times = collections.defaultdict(list)

    @property
    def mermaid_flowchart(self):
        result = "flowchart LR\n"
        clusters = collections.defaultdict(set)
        transitions = ""
        cluster_transitions = set()
        for trans_tup in self._transitions:
            (first_base, first, second_base, second) = trans_tup

            clusters[first_base].add(first)
            clusters[second_base].add(second)
            deltas = self._transition_times[trans_tup]
            min_delta = min(deltas, key=lambda x: x[1])[1]
            max_delta = max(deltas, key=lambda x: x[1])[1]
            mean_delta = sum([x[1] for x in deltas]) / len(deltas)
            delta_str = f"{min_delta:.2f} - {max_delta:.2f} ({mean_delta:.2f})"
            transitions += f'  {first} -->|"{delta_str}"| {second}\n'

        # Add clusters
        for cname, cluster in clusters.items():
            if cname == "None":
                continue
            result += f"  subgraph {cname}\n"
            for node in cluster:
                result += f"    {node}\n"
            result += "  end\n"

        result += transitions
        result += "\n"
        return result

    def save_mermaid_flowchart(self, filename):
        with open(filename, "w") as f:
            f.write(self.mermaid_flowchart)

    @property
    def graphviz_digraph(self):
        result = "digraph State {\n\trankdir=LR;\n\tnodesep=0.5;\n"
        clusters = collections.defaultdict(set)
        transitions = ""
        cluster_transitions = set()
        for trans_tup in self._transitions:
            (first_base, first, second_base, second) = trans_tup
            clusters[first_base].add(first)
            clusters[second_base].add(second)
            cluster_transitions.add(f"{first_base}->{second_base}")
            trans_deltas = self._transition_times[trans_tup]
            trans_deltas_strs = [f"{t_id}: {time:.2}" for t_id, time in trans_deltas]
            transitions += '{}->{} [ label="{}" ];\n'.format(
                first, second, "\n".join(trans_deltas_strs)
            )

        for cname, cluster in clusters.items():
            result += f"\tsubgraph cluster_{cname} {{\n"
            result += f'\t\tlabel="{cname}"'
            for node in cluster:
                result += f"\t\t{node};\n"
            result += "\tcolor=black;\n"
            result += "\t}\n\n"

        result += transitions
        result += "}\n"
        return result

    def save_graphviz_digraph(self, filename):
        with open(filename, "w") as f:
            f.write(self.graphviz_digraph)

    def reset(self):
        self._is_finished = False
        self.clear_state_stack()
        self._transition(self._initial_st)

    def clear_state_stack(self):
        self._state_stack = []

    @property
    def current_state_stack(self):
        """Returns a copy of the current state stack."""
        return self._state_stack.copy()

    def cleanup(self):
        if self._current:
            self._current._cancel_failsafe()

    @property
    def is_finished(self):
        return self._is_finished

    def start_non_blocking(self):
        # Still need while loop for getting errors pushed into queue
        while True:
            try:
                # Still check messages for RetryLimitException
                self.tick()
            except StateMachineComplete:
                raise

            # TODO: Test this, seems wrong
            if self._msg_queue.empty():
                raise NonBlockingStalled(
                    f"Non-blocking state machine stalled in {self._current.name}"
                )

    def _log_transition(self, from_st: Optional[State], to_st: State):
        """
        Logs a state transition, including the transition details and timing.

        Args:
            from_st: The state transitioning from. Can be None if no prior state exists.
            to_st: The state transitioning to.
        """
        # Calculate time deltas for each transition
        trans_time = datetime.now()
        trans_delta = (trans_time - self._last_trans_time).total_seconds()
        self._last_trans_time = trans_time

        if from_st:
            from_name = from_st.name
            from_base = from_st.base_name
        else:
            from_name = "None"
            from_base = "None"

        to_name = to_st.name
        to_base = to_st.base_name
        trans_tup = (from_base, from_name, to_base, to_name)

        self._transitions.add(trans_tup)
        self._transition_times[trans_tup].append((self._transition_id, trans_delta))
        self._transition_id += 1

        self._log_fn(f"State Transition: {from_name} -> {to_name}")

    def _transition(self, trans_state):
        next_state = None
        continuation = False
        renewal = False
        retry = False
        match trans_state:
            case p if isinstance(p, Push):
                # If the next state is a Push, save the push states on the
                # state stack and transition to the next state

                # Push the states on the stack in reverse order, keeping
                # the first state for the transition
                for state in reversed(trans_state.push_states[1:]):
                    self._state_stack.append(state)
                next_state = trans_state.push_states[0]
            case type() if issubclass(trans_state, Pop):
                # For a pop, try to pull the top state off of the stack.
                # Otherwise, just transition to the state provided
                if not self._state_stack:
                    raise EmptyStateStackError("No states on stack!")
                next_state = self._state_stack.pop()
            case type() if issubclass(trans_state, State):
                # For a State, we're just going to transition
                next_state = trans_state
            case type() if issubclass(trans_state, (Again, Unhandled)):
                continuation = True
            case type() if issubclass(trans_state, Retry):
                continuation = True
                retry = True
                self._current.has_entered = False
            case type() if issubclass(trans_state, StateRenewal):
                renewal = True

        if renewal or continuation:
            next_state = type(self._current)

        transition = type(self._current) != next_state

        self._log_transition(self._current, next_state)
        self._transition_fn(next_state, self._ctx)

        # If we are preempting another state (as in the case of an error), or we
        # haven't called on_exit (as is the case in Renewals and Retries), we
        # need to make sure we've cleared out the failsafe timer.
        if self._current and (renewal or retry or transition):
            self._current._cancel_failsafe()

        # Continuations re-use the same state context (to keep track of retries)
        if not continuation:
            self._current = next_state()

    def send_message(self, message=None):
        """Sends a message to the finite state machine"""

        # TODO: Return filtered or not
        # If this is a filtered message, no reason to give it to the state
        # machine
        try:
            if message and self._filter_fn(self._ctx, message):
                return
            # Messages are appended to the right and handled FIFO style
            self._send_message_internal(message)
        except Exception as e:
            # Exceptions are given high priority to ensure we handle them first
            self._send_message_internal(e)

    def _send_message_internal(self, message):
        match message:
            case _ if isinstance(message, EnokiInternalMessage):
                self._msg_queue.put(
                    PrioritizedMessage(self.INTERNAL_MESSAGE_PRIORITY, message)
                )
            case _ if isinstance(message, Exception):
                self._msg_queue.put(PrioritizedMessage(self.ERROR_PRIORITY, message))
            case _:
                # standard message
                self._msg_queue.put(PrioritizedMessage(self.MESSAGE_PRIORITY, message))

    def tick(self, timeout: Optional[int] = 0):
        # Grab our next message
        # Timeout: 0 = Non blocking (default)
        # Timeout: None = Blocks forever
        # Timeout: >0 = Blocks for specific amount

        match timeout:
            case 0:
                block = False
            case None:
                block = True
            case x if x > 0:
                block = True
            case _:
                raise ValueError("Tick timeout must be None, or >=0")

        fsm_busy = True
        while fsm_busy:
            try:
                self._ctx.msg = self._msg_queue.get(block, timeout).item
            except Empty:
                self._ctx.msg = None

            try:
                self._is_finished = self._current.TERMINAL

                if isinstance(self._ctx.msg, Exception):
                    raise self._ctx.msg

                next_transition = self._current._tick(self, self._ctx)

                if next_transition is Unhandled and self._ctx.msg:
                    # Trap unhandled messages
                    self._trap_fn(self._ctx)

                # Clear the message
                self._ctx.msg = None

                # Transition to our next state
                self._transition(next_transition)

                fsm_busy = (not self._msg_queue.empty()) or (not next_transition.AWAITS)
            except Exception as e:
                # While it's true that 'Pokemon errors' are typically
                # in poor taste, this allows the user to selectively
                # handle error cases, and throw any error that isn't
                # explicitely handled
                next_transition = None
                if self._on_err_fn:
                    next_transition = self._on_err_fn(self._ctx, e)
                if next_transition:
                    self._transition(next_transition)
                else:
                    raise e

        if self.is_finished:
            raise StateMachineComplete

        # If the state machine hasn't finished and the current state doesn't
        # have a timeout or isn't marked as can dwell at the end of a tick an
        # exception is raised to indicate that the state machine is stalled.
        if (
            self._msg_queue.empty()
            and self._current.TIMEOUT is None
            and not self._current.CAN_DWELL
        ):
            raise BlockedInUntimedState(self._current)
