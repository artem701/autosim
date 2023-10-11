from helpers import remove_by_identity
from helpers import coalesce
import logging
from enum import Enum, auto


class Event:
    def __init__(self):
        self.sender = None


class Listener:
    def inputEvents(self) -> set[type(Event)]:
        """List of events accepted by listner. None is a wildcard for any event.

        Returns:
            list[Event]: List of accepted events or None.
        """
        return None

    def accept(self, event: Event) -> list[Event]:
        """Accept event and return produced events.

        Args:
            event (Event): Event that happened recently.

        Returns:
            list[Event]: Events produced while handling this event.
        """
        return None


class Iteration(Event):
    """Service event, produced each loop iteration. Cannot be created by any objects but Event Loop.
    """
    pass


class AddListener(Event):
    """Service event, needed to create new listneres by current listeners.
    """

    def __init__(self, listener: Listener):
        self.listener = listener


class RemoveListener(Event):
    """Service event, needed to create new listneres by current listeners.
    """

    def __init__(self, listener: Listener):
        self.listener = listener


class Terminate(Event):
    """Terminate Event Loop.
    """

    def __init__(self, immediate: bool):
        """Terminate event constructor.

        Args:
            immediate (bool): If True, current iteration is finished, then in the next iteration events produced
            before Terminate are executed. If False then second iteration is executed fully as well.
        """
        self.immediate = immediate


class EventLoop(Listener):

    ListenersSet = set[Listener]
    ListenersQueue = list[Listener]
    EventsQueue = list[Event]
    EventsMap = dict[type(Event), ListenersQueue]

    class _ListnerAction(Enum):
        ADD = auto()
        REMOVE = auto()

    def __init__(self):
        self._listners = EventLoop.ListenersSet()
        self._listners_actions = []
        self._queue = EventLoop.EventsQueue()
        self._map = EventLoop.EventsMap()

        self._terminate_flag = False
        self._terminate_immediate_flag = False

        self.subscribe(self)

    def inputEvents(self) -> set[type(Event)]:
        return {Terminate, AddListener, RemoveListener}

    def accept(self, event: Event) -> list[Event]:
        if isinstance(event, AddListener):
            self._listners_actions.append(
                (event.listener, EventLoop._ListnerAction.ADD))

        if isinstance(event, RemoveListener):
            self._listners_actions.append(
                (event.listener, EventLoop._ListnerAction.REMOVE))

        if isinstance(event, Terminate):
            self._terminate_flag = True
            self._terminate_immediate_flag = self._terminate_immediate_flag or event.immediate

    def subscribe(self, listener: Listener):
        for l in self._listners:
            if l is listener:
                return
        inputEvents = listener.inputEvents()
        if inputEvents is None or len(inputEvents) == 0:
            return
        self._listners.add(listener)
        for evt in inputEvents:
            self._map.setdefault(
                evt, EventLoop.ListenersQueue()).append(listener)

    def unsubscribe(self, listener: Listener):
        if listener not in self._listners:
            return
        self._listners.remove(listener)
        for _, ls in self._map:
            remove_by_identity(ls, listener)

    def put(self, event: Event):
        """Put event into queue. May use for environment preparation before run.
        """
        self._queue.append(event)

    def iterate(self) -> bool:
        """Make a single Event Loop interation.

        Returns:
            bool: Termination indicator.
        """
        next_queue = EventLoop.EventsQueue()
        self._queue = EventLoop.EventsQueue([Iteration()]) + self._queue
        # Dispatch events
        for e in self._queue:
            ls = self._map.setdefault(type(e), [])
            if len(ls) == 0:
                logging.warning(f"Unhandeled event {type(e)}")
            for l in self._map.setdefault(type(e), []):
                for ev in coalesce(l.accept(e), []):
                    ev.sender = l
                    next_queue.append(ev)
                if self._terminate_immediate_flag:
                    return
        # Use new queue
        self._queue = next_queue
        # Handle listners changes
        for la in self._listners_actions:
            if la == EventLoop._ListnerAction.ADD:
                self.subscribe(l)
            elif la == EventLoop._ListnerAction.REMOVE:
                self.unsubscribe(l)
            else:
                raise ValueError(f"Unhandeled listner action {la}")

    def loop(self):
        while not self._terminate_flag:
            self.iterate()
