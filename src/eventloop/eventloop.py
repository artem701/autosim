from helpers import remove_by_identity, to_array
import logging
from enum import Enum, auto


class Event:
    def __init__(self):
        self.sender = None


class Listener:
    def input_events(self) -> set[type(Event)]:
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

    def input_events(self) -> set[type(Event)]:
        return {Terminate, AddListener, RemoveListener}

    def accept(self, event: Event) -> list[Event]:
        logging.debug(f"rcv {event}")
        if isinstance(event, AddListener):
            self._listners_actions.append(
                (event.listener, EventLoop._ListnerAction.ADD))

        if isinstance(event, RemoveListener):
            self._listners_actions.append(
                (event.listener, EventLoop._ListnerAction.REMOVE))

        if isinstance(event, Terminate):
            self._terminate_flag = True
            self._terminate_immediate_flag = self._terminate_immediate_flag or event.immediate

    def subscribe(self, listener: Listener) -> bool:
        logging.debug(f"subscribe {listener}...")
        for l in self._listners:
            if l is listener:
                raise RuntimeError(f"{listener} is already subscribed!")

        input_events = to_array(listener.input_events())
        if len(input_events) == 0:
            raise RuntimeError(f"{listener} has no iput events!")

        self._listners.add(listener)
        for evt in input_events:
            self._map.setdefault(
                evt, EventLoop.ListenersQueue()).append(listener)
        logging.debug(f"subscribed to events {input_events}.")
        return True

    def unsubscribe(self, listener: Listener):
        if listener not in self._listners:
            return
        self._listners.remove(listener)
        for ls in self._map.values():
            remove_by_identity(ls, listener)

    def put(self, event: Event):
        """Put event into queue. May use for environment preparation before run.
        """
        self._queue.append(event)

    def loop(self):
        while self.iterate():
            pass

    def iterate(self):
        """Make a single Event Loop interation.
        """
        logging.debug('= = = ITERATION START = = =')
        self._queue = EventLoop.EventsQueue([Iteration()]) + self._queue
        while not self._terminate_immediate_flag and len(self._queue) > 0:
            self._queue = self._handle_events()
        self._handle_listeners_actions()
        logging.debug('= = =  ITERATION END  = = =')
        return not self._terminate_flag

    def _handle_events(self) -> EventsQueue:
        """Handle available events in the queue. Return queue with new events.
        """
        new_events = EventLoop.EventsQueue()
        for e in self._queue:
            new_events += self._handle_event(e)
            if self._terminate_immediate_flag:
                break
        return new_events

    def _handle_event(self, event) -> EventsQueue:
        """Handle one event. Return produced events.
        """
        new_events = EventLoop.EventsQueue()
        listeners = self._map.setdefault(type(event), [])

        logging.debug(f"handle event {event}")

        if len(listeners) == 0:
            logging.warning(f"Unhandeled event {type(event)}")

        # For each listner which accepts this event
        for l in listeners:
            for ev in to_array(l.accept(event)):
                ev.sender = l
                new_events.append(ev)

            if self._terminate_immediate_flag:
                break

        return new_events

    def _handle_listeners_actions(self):
        for listener, action in self._listners_actions:
            logging.debug(f"handle {action.name} {listener}")
            if action == EventLoop._ListnerAction.ADD:
                self.subscribe(listener)
            elif action == EventLoop._ListnerAction.REMOVE:
                self.unsubscribe(listener)
            else:
                raise ValueError(f"Unhandeled listner action {action.name}")
        self._listners_actions = []
