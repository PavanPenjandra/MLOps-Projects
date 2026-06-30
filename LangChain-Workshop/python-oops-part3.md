### Module 7 — `@property`, `@classmethod`, `@staticmethod` (Hours 19–21)
**Three decorators, three completely different jobs — they are not variations on a theme**

#### 7.1 Concept Lecture — What each one actually is (1h)

**Why this matters for LangGraph:** every LangChain base class uses all three. `name` and `description` on `BaseTool` are `@property` — read-only access to private fields. `from_dict()` factory methods are `@classmethod`. `validate_model_name()` helpers are `@staticmethod`. Once you can read which one a method uses, you instantly know its relationship to instances vs. the class.

**The three in one diagram:**

```
                    STATIC METHOD           CLASS METHOD            INSTANCE METHOD
                    (@staticmethod)         (@classmethod)          (@property / plain)
                    ───────────────         ──────────────          ───────────────────
Receives:           nothing special         cls (the class)         self (the instance)
Called on:          class OR instance       class OR instance        instance only
Knows about:        neither class           the class               both class (via type)
                    nor instance            (not the instance)      and instance
Purpose:            utility function        factory / registry      behavior or attribute
                    that lives nearby                               per object

Example:
@staticmethod                  @classmethod                  @property
def validate(name: str) -> bool:  def from_dict(cls, d):         def name(self):
    return bool(name.strip())      return cls(**d)                return self._name
```

**`@property` — a method that pretends to be an attribute:**
```python
class ResearchAgent:
    def __init__(self, name: str):
        self._name = name    # private by convention: _name, not name

    @property
    def name(self) -> str:
        return self._name    # called when someone reads agent.name -- no ()

    @name.setter
    def name(self, value: str):
        if not value.strip():
            raise ValueError("name cannot be blank")
        self._name = value.strip()   # called when someone writes agent.name = "x"

    @name.deleter
    def name(self):
        raise AttributeError("name cannot be deleted")

a = ResearchAgent("ev_agent")
print(a.name)          # "ev_agent" -- reads via getter, no ()
a.name = "  ev_v2  "  # writes via setter, strips whitespace automatically
print(a.name)          # "ev_v2"
try:
    del a.name
except AttributeError as e:
    print("deleter guard:", e)
```

**`@classmethod` — the factory pattern:**
```python
class ResearchAgent:
    def __init__(self, name: str, model: str, brief: str):
        self._name = name
        self.model = model
        self.brief = brief

    @classmethod
    def from_dict(cls, d: dict) -> "ResearchAgent":
        return cls(d["name"], d["model"], d["brief"])

    @classmethod
    def ev_researcher(cls) -> "ResearchAgent":
        return cls("ev_agent", "gpt-4o", "EV charging infrastructure")

config = {"name": "market_agent", "model": "claude-3", "brief": "Solar market"}
agent1 = ResearchAgent.from_dict(config)      # factory from dict
agent2 = ResearchAgent.ev_researcher()         # named preset

print(agent1._name, agent1.model, agent1.brief)
print(agent2._name, agent2.model, agent2.brief)
```

**Why `cls` instead of hardcoding the class name:** if a subclass inherits `from_dict()`, calling `SubClass.from_dict(d)` correctly returns a `SubClass` instance — not a `ResearchAgent`. This is the whole point of `cls`. Hardcoding `ResearchAgent(...)` inside the method would break subclass factories silently.

**`@staticmethod` — a utility function that lives inside the class for organization:**
```python
class ResearchAgent:
    SUPPORTED_MODELS = ["gpt-4o", "gpt-4", "claude-3", "claude-2"]

    @staticmethod
    def validate_model(model_name: str) -> bool:
        return model_name in ResearchAgent.SUPPORTED_MODELS

    @staticmethod
    def format_brief(brief: str, max_length: int = 200) -> str:
        return brief.strip()[:max_length]

print(ResearchAgent.validate_model("gpt-4o"))     # True
print(ResearchAgent.validate_model("llama-3"))    # False
print(ResearchAgent.format_brief("  EV Market  "))   # "EV Market"
```

A `@staticmethod` is just a regular function grouped under the class — it receives neither `self` nor `cls`. If it doesn't need access to the class or any instance, it's a static method. If it uses the class itself (to create instances, access class attributes), it should be a `@classmethod` instead.

#### 7.2 Guided Coding Drill — A `ResearchAgent` using all three (1h)

```python
class ResearchAgent:
    _registry: dict = {}

    def __init__(self, name: str, model: str, brief: str = ""):
        if not ResearchAgent.validate_model(model):
            raise ValueError(f"Unsupported model: {model!r}")
        self._name = name.strip()
        self._model = model
        self._brief = brief
        ResearchAgent._registry[self._name] = self

    @property
    def name(self) -> str:
        return self._name

    @property
    def model(self) -> str:
        return self._model

    @model.setter
    def model(self, value: str):
        if not ResearchAgent.validate_model(value):
            raise ValueError(f"Unsupported model: {value!r}")
        self._model = value

    @property
    def brief(self) -> str:
        return self._brief

    @brief.setter
    def brief(self, value: str):
        self._brief = ResearchAgent.format_brief(value)

    @classmethod
    def from_dict(cls, d: dict) -> "ResearchAgent":
        return cls(d["name"], d["model"], d.get("brief", ""))

    @classmethod
    def get_by_name(cls, name: str) -> "ResearchAgent":
        if name not in cls._registry:
            raise KeyError(f"No agent named {name!r}")
        return cls._registry[name]

    @classmethod
    def all_agents(cls) -> list:
        return list(cls._registry.values())

    @staticmethod
    def validate_model(model: str) -> bool:
        return isinstance(model, str) and model.strip().startswith(("gpt", "claude"))

    @staticmethod
    def format_brief(brief: str, max_len: int = 300) -> str:
        return " ".join(brief.strip().split())[:max_len]

    def __repr__(self):
        return f"ResearchAgent(name={self._name!r}, model={self._model!r})"

a1 = ResearchAgent("ev_agent", "gpt-4o", "  EV market  analysis  ")
a2 = ResearchAgent.from_dict({"name": "solar_agent", "model": "claude-3-5-sonnet"})

print(a1.brief)            # 'EV market analysis' -- stripped and normalized
a2.model = "gpt-4"         # setter validates the new value
print(a2.model)

print(ResearchAgent.get_by_name("ev_agent"))
print(ResearchAgent.all_agents())

try:
    bad = ResearchAgent("x", "llama-3")
except ValueError as e:
    print("validation error:", e)
```

**Drill:** add a `@classmethod reset_registry(cls)` that clears the `_registry` dict — useful in tests when you need a clean slate — and a `@staticmethod briefing_score(brief) -> float` that returns a quality score for a brief (e.g. word count / 10, capped at 1.0).

#### 7.3 Homework/Challenge — Subclass-aware factories (1h)

Prove that `@classmethod` factories work correctly for subclasses:
1. Create a `SpecializedAgent(ResearchAgent)` with one extra attribute (`specialty: str`)
2. Override `from_dict(cls, d)` in the subclass to also read `d["specialty"]`
3. Confirm that `SpecializedAgent.from_dict(d)` returns a `SpecializedAgent`, not a `ResearchAgent`
4. Confirm that `ResearchAgent.from_dict(d)` still returns a plain `ResearchAgent`

**Challenge:** add a `@classmethod from_json(cls, json_str: str)` to `ResearchAgent` that parses a JSON string and delegates to `from_dict` — but raises a clear `ValueError` (not the raw `json.JSONDecodeError`) if the JSON is malformed.

---

### Module 8 — Composition, Mixins, and Design Patterns (Hours 22–24)
**HAS-A beats IS-A most of the time — here's when, why, and how**

#### 8.1 Concept Lecture — Composition vs. inheritance: not a style preference, a design decision (1h)

**Why this matters for LangGraph:** LangChain's tool system is not a deep inheritance tree — it's a flat, composable architecture. A `ResilientSearchTool` is not "a RetryingTool that is a LoggingTool that is a BaseTool" (deep inheritance). It IS-A `BaseTool` (one level of inheritance for the contract), and it HAS-A `Logger` and HAS-A `RetryPolicy` (two composed behaviors). Understanding this is what lets you extend LangChain cleanly instead of fighting it.

**IS-A vs. HAS-A decision guide:**
```
Ask yourself: "Is this relationship always true for every instance?"

"A SearchTool IS-A BaseTool"       → inheritance -- always true, non-optional
"A SearchTool HAS-A RateLimiter"   → composition -- optional, swappable per instance
"A ChatAgent HAS-A Memory"         → composition -- the agent is not A memory
"A ChatAgent HAS-A ToolBox"        → composition -- you can swap the toolbox
"A ChatAgent IS-A Runnable"        → Protocol -- it has the right methods
```

**Composition in code — the `ChatAgent HAS-A Memory, HAS-A ToolBox`:**
```python
class Memory:
    """Sliding-window conversation memory."""
    def __init__(self, maxsize=5):
        from collections import deque
        self._store = deque(maxlen=maxsize)

    def add(self, turn: dict): self._store.append(turn)
    def get_all(self) -> list: return list(self._store)
    def __len__(self): return len(self._store)

class ToolBox:
    def __init__(self):
        self._tools = {}

    def add(self, tool): self._tools[tool.name] = tool
    def run(self, name, **kwargs): return self._tools[name].run(**kwargs)
    def has(self, name): return name in self._tools
    def __repr__(self): return f"ToolBox(tools={list(self._tools)})"

class ChatAgent:
    """
    ChatAgent COMPOSES Memory and ToolBox.
    It does NOT inherit from either -- it uses them.
    """
    def __init__(self, name: str, memory_size: int = 5):
        self.name = name
        self.memory = Memory(maxsize=memory_size)   # HAS-A Memory
        self.tools = ToolBox()                        # HAS-A ToolBox

    def add_tool(self, tool): self.tools.add(tool)

    def chat(self, user_message: str) -> str:
        tool_hint = "search" if "?" in user_message else None
        if tool_hint and self.tools.has(tool_hint):
            response = self.tools.run(tool_hint, query=user_message)
        else:
            response = f"[{self.name}]: I processed '{user_message}'"
        self.memory.add({"user": user_message, "ai": str(response)})
        return str(response)

    def __repr__(self):
        return f"ChatAgent(name={self.name!r}, memory={len(self.memory)}/{5}, {self.tools})"
```

**Mixins — adding behaviors without deep inheritance:**

A **mixin** is a class that provides one specific behavior (logging, retrying, rate-limiting) and is designed to be combined with other mixins via multiple inheritance. The key: a mixin should **never** have its own `__init__` that conflicts, and it should **always** call `super()` cooperatively.

```python
class LogMixin:
    """Adds logging to any class. No __init__ needed."""
    def log(self, msg: str, level: str = "INFO"):
        print(f"[{level}] [{self.__class__.__name__}] {msg}")


class RetryMixin:
    """Adds retry behavior. Must be combined with a class that has self.log()."""
    MAX_RETRIES: int = 3

    def run_with_retry(self, fn, *args, **kwargs):
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                self.log(f"attempt {attempt}/{self.MAX_RETRIES} failed: {e}", "WARN")
                if attempt == self.MAX_RETRIES:
                    raise


class ResilientSearchTool(LogMixin, RetryMixin, BaseTool):
    """IS-A BaseTool. HAS behaviors from LogMixin and RetryMixin."""

    def __init__(self, max_results: int = 5):
        super().__init__("resilient_search", "Web search with retry and logging")
        self.max_results = max_results
        self._call_number = 0

    def _raw_search(self, query: str) -> str:
        self._call_number += 1
        if self._call_number < 2:   # simulated flakiness
            raise ConnectionError("transient network failure")
        return f"Top {self.max_results} results for '{query}'"

    def run(self, query: str, **kwargs) -> str:
        self.log(f"searching for: {query!r}")
        result = self.run_with_retry(self._raw_search, query)   # from RetryMixin
        self.log(f"success: {result}")
        return result
```

**Why `LogMixin` works without `__init__`:** it only adds methods. `self.__class__.__name__` works because `self` is always the final concrete object — even inside a mixin method. This is the same reason `super()` in mixins cooperates correctly through the MRO.

#### 8.2 Guided Coding Drill — Build a production-grade `ResilientSearchTool` (1h)

```python
tool = ResilientSearchTool(max_results=3)
print(repr(tool))
result = tool.run("EV charging infrastructure")
print(result)
print("MRO:", [c.__name__ for c in ResilientSearchTool.__mro__])
```

**Expected output (verified):**
```
[INFO] [ResilientSearchTool] searching for: 'EV charging infrastructure'
[WARN] [ResilientSearchTool] attempt 1/3 failed: transient network failure
[INFO] [ResilientSearchTool] success: Top 3 results for 'EV charging infrastructure'
Top 3 results for 'EV charging infrastructure'
MRO: ['ResilientSearchTool', 'LogMixin', 'RetryMixin', 'BaseTool', 'object']
```

**Drill:** add a third mixin `CacheMixin` that stores results keyed by `(fn_name, args_hash)` in a dict, and returns the cached result on a cache hit instead of calling the function again. The mixin should override `run_with_retry` (if `RetryMixin` is present) or `run()` (if not), using `super()` correctly so it plays nicely in the MRO chain.

#### 8.3 Homework/Challenge — The `Agent` composition pattern (1h)

Build a `BaseAgent` using composition of components, with the following design:
1. `BaseAgent` composes `Memory`, `ToolBox`, and a list of `step_functions` (plain callables)
2. `run_pipeline(brief)` iterates through steps, threading a state dict through each
3. `BaseAgent` does **not** inherit from `Memory`, `ToolBox`, or any step function class

Then demonstrate the "swap the component" advantage: create two agents with the same steps but different memory sizes — prove they're completely independent without any code change in the step functions.

**Challenge:** add `__enter__` / `__exit__` to `BaseAgent` so it can be used as a context manager: `with BaseAgent("ev", memory_size=10) as agent: ...` — print "agent initialized" on enter and "agent cleaned up" on exit (simulate closing a connection).

---

### Module 9 — `dataclasses`, `__slots__`, and Descriptors (Hours 25–27)
**Three tools for specific problems — not replacements for each other**

#### 9.1 Concept Lecture — `@dataclass`: when your class is primarily data (1h)

**Why this matters for LangGraph:** `ToolResult`, `CheckpointData`, `Message` — any class whose job is mainly holding structured data with equality and a good repr — should be a `@dataclass`. The decorator writes `__init__`, `__repr__`, and `__eq__` for you, correctly, every time.

```python
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ToolResult:
    tool_name: str
    output: str
    success: bool = True
    error: str = ""
    metadata: dict = field(default_factory=dict)   # NEVER use metadata: dict = {}!
                                                    # That's the mutable-default-arg bug -- field() fixes it

    def __post_init__(self):
        """Called automatically after __init__ -- validate/transform here."""
        if not self.tool_name.strip():
            raise ValueError("tool_name cannot be blank")
        if not self.success and not self.error:
            self.error = "Unknown error"

r1 = ToolResult("search", "found 3 results")
r2 = ToolResult("search", "found 3 results")
print(r1)              # ToolResult(tool_name='search', output='found 3 results', success=True, error='', metadata={})
print(r1 == r2)        # True -- __eq__ generated automatically

r3 = ToolResult("calc", "", success=False)
print(r3.error)        # "Unknown error" -- set by __post_init__
```

**`frozen=True` for immutable dataclasses:**
```python
@dataclass(frozen=True)
class AgentConfig:
    model: str
    temperature: float = 0.7
    max_tokens: int = 1000

config = AgentConfig("gpt-4o")
print(config)
try:
    config.model = "claude"
except Exception as e:
    print("frozen guard:", type(e).__name__, e)
# FrozenInstanceError: cannot assign to field 'model'
```

**`order=True` for sortable dataclasses:**
```python
@dataclass(order=True)
class SearchResult:
    score: float
    content: str

results = [SearchResult(0.7, "b"), SearchResult(0.9, "a"), SearchResult(0.5, "c")]
print(sorted(results))   # sorted by score automatically via generated __lt__, __gt__, etc.
```

**`__slots__` — trading flexibility for memory efficiency:**
```python
class Finding:
    __slots__ = ("specialist", "content", "confidence")
    def __init__(self, specialist, content, confidence):
        self.specialist = specialist
        self.content = content
        self.confidence = confidence

f = Finding("web_researcher", "EV market grew 20%", 0.9)
print(f.specialist, f.confidence)
try:
    f.unexpected = "nope"    # AttributeError: slots prevent arbitrary attribute creation
except AttributeError as e:
    print("slots guard:", e)
```

**Why `__slots__` matters:** a standard Python object stores its instance attributes in a `__dict__` (an actual dictionary). For a class that creates thousands of instances (e.g., `Finding` objects for every chunk retrieved from a vector store), that per-instance dict has real memory overhead. `__slots__` replaces the per-instance dict with a fixed, pre-allocated structure — typically 40–50% less memory per instance, with slightly faster attribute access.

When to use `__slots__`:
- You know all attribute names at class definition time (can't add new ones later)
- You're creating many instances (thousands+)
- You need the memory saving or the "no arbitrary attributes" guarantee

When **not** to use `__slots__`:
- When you need `__dict__` (e.g., for dynamic attribute tricks, or mixing with classes that don't use `__slots__`)
- For most everyday classes — the optimization is real but usually unnecessary

#### 9.2 Guided Coding Drill — Descriptors: reusable field-level logic (1h)

**The concept.** A descriptor is any object that implements `__get__`, `__set__`, or `__delete__` — and when placed as a class attribute, it intercepts every access to that attribute on any instance. This is how `@property` works internally — `property` is itself a descriptor class.

**Why this matters for LangGraph:** if you find yourself writing the same validation logic in `@property` setters across three different classes, you want a descriptor. LangChain's Pydantic-based fields are, at the language level, descriptors — this is the mechanism you'd reach for if you ever needed field-level validation without Pydantic.

```python
class Validated:
    """A reusable descriptor that validates string fields."""

    def __set_name__(self, owner, name):
        self.name = name              # the attribute name on the class
        self.storage_name = f"_{name}"   # where we actually store the value

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self               # accessed on the CLASS, not an instance
        return getattr(obj, self.storage_name, None)

    def __set__(self, obj, value):
        if not isinstance(value, str):
            raise TypeError(f"{self.name!r} must be a string, got {type(value).__name__!r}")
        if not value.strip():
            raise ValueError(f"{self.name!r} cannot be blank")
        setattr(obj, self.storage_name, value.strip())


class NonNegativeFloat:
    """A reusable descriptor for non-negative float fields."""

    def __set_name__(self, owner, name):
        self.name = name
        self.storage_name = f"_{name}"

    def __get__(self, obj, objtype=None):
        if obj is None: return self
        return getattr(obj, self.storage_name, 0.0)

    def __set__(self, obj, value):
        value = float(value)
        if value < 0:
            raise ValueError(f"{self.name!r} must be non-negative, got {value}")
        setattr(obj, self.storage_name, value)


class Finding:
    """Uses two reusable descriptors instead of two property pairs."""
    specialist = Validated()
    content    = Validated()
    confidence = NonNegativeFloat()

    def __init__(self, specialist, content, confidence):
        self.specialist = specialist    # goes through Validated.__set__
        self.content = content          # goes through Validated.__set__
        self.confidence = confidence    # goes through NonNegativeFloat.__set__

    def __repr__(self):
        return (f"Finding(specialist={self.specialist!r}, "
                f"confidence={self.confidence})")


f = Finding("web_researcher", "EV market grew 20%", 0.9)
print(f)
try: Finding("", "content", 0.5)
except ValueError as e: print("blank guard:", e)
try: Finding("name", "content", -0.1)
except ValueError as e: print("negative guard:", e)
try: Finding(123, "content", 0.5)
except TypeError as e: print("type guard:", e)
print("descriptor on class:", type(Finding.specialist))
```

**Verified output:**
```
Finding(specialist='web_researcher', confidence=0.9)
blank guard: 'specialist' cannot be blank
negative guard: 'confidence' must be non-negative, got -0.1
type guard: 'specialist' must be a string, got 'int'
descriptor on class: <class '__main__.Validated'>
```

**The `__set_name__` magic:** called automatically when the class is defined, it tells each descriptor what attribute name it was assigned to — so `specialist = Validated()` causes Python to call `Validated.__set_name__(Finding, "specialist")`, letting the descriptor know its own name without you having to pass it explicitly.

#### 9.3 Homework/Challenge — A `TypedField` descriptor with default values (1h)

Build a general `TypedField(expected_type, default=None, validator=None)` descriptor:
1. `expected_type`: raises `TypeError` if the value is the wrong type
2. `default`: returned by `__get__` if the field hasn't been set yet
3. `validator`: an optional callable `(value) -> None` that raises `ValueError` if the value is invalid

```python
import re

class AgentConfig:
    name    = TypedField(str, validator=lambda v: None if v.strip() else (_ for _ in ()).throw(ValueError("blank")))
    model   = TypedField(str, default="gpt-4o")
    temp    = TypedField(float, default=0.7, validator=lambda v: None if 0 <= v <= 2 else (_ for _ in ()).throw(ValueError("out of range")))
    max_tok = TypedField(int, default=1000)

    def __init__(self, name, **kwargs):
        self.name = name
        for k, v in kwargs.items():
            setattr(self, k, v)
```

**Challenge:** make `TypedField` also work as a class that can be used as a type annotation alongside `@dataclass`, the same way `dataclasses.field()` works — implementing `__class_getitem__` so you can write `TypedField[str]` as the annotation.

---

### Module 10 — Capstone: Build a Complete Agent Class Hierarchy (Hours 28–30)
**Everything assembled — no library imports, no shortcuts, the real shape**

#### The Brief

You are building a minimal, production-shaped agentic system in pure Python — no LangChain, no LangGraph, no Pydantic, no `asyncio`. Every class you write should be immediately recognizable to a LangChain developer as "yes, that's the right shape." By the end of Hour 30, the LangGraph/LangChain source code should read like a more feature-complete version of what you just built.

#### 10.1 Session 1 (1h): `ToolResult`, `BaseTool`, `ToolBox`

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

@dataclass
class ToolResult:
    tool_name: str
    output: str
    success: bool = True
    error: str = ""
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.tool_name.strip():
            raise ValueError("tool_name cannot be blank")
        if not self.success and not self.error:
            self.error = "Unknown error occurred"

@runtime_checkable
class Runnable(Protocol):
    def invoke(self, state: dict) -> dict: ...

class BaseTool(ABC):
    def __init__(self, name: str, description: str):
        if not name.strip():
            raise ValueError("Tool name cannot be blank")
        self._name = name.strip()
        self._description = description.strip()
        self._call_count = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @abstractmethod
    def run(self, *args, **kwargs) -> ToolResult:
        pass

    def invoke(self, state: dict) -> dict:
        self._call_count += 1
        try:
            filtered = {k: v for k, v in state.items()
                        if k not in ("history", "thread_id", "error")}
            result = self.run(**filtered)
            return {"tool_result": result, "last_error": None}
        except Exception as e:
            err = ToolResult(self._name, "", success=False, error=str(e))
            return {"tool_result": err, "last_error": str(e)}

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self._name!r}, calls={self._call_count})"


class ToolBox:
    def __init__(self):
        self._tools: dict[str, BaseTool] = {}

    def add(self, tool: BaseTool) -> "ToolBox":
        if not isinstance(tool, BaseTool):
            raise TypeError(f"Expected BaseTool, got {type(tool).__name__}")
        self._tools[tool.name] = tool
        return self

    def __len__(self):      return len(self._tools)
    def __contains__(self, name): return name in self._tools
    def __iter__(self):     return iter(self._tools.values())
    def __getitem__(self, name):
        if name not in self._tools:
            raise KeyError(f"No tool {name!r}. Available: {list(self._tools)}")
        return self._tools[name]
    def __call__(self, name, *args, **kwargs):
        return self[name].run(*args, **kwargs)
    def __repr__(self):
        return f"ToolBox(tools={list(self._tools)})"


class CalculatorTool(BaseTool):
    def __init__(self):
        super().__init__("calculator", "Performs add or multiply on two numbers")

    def run(self, a, b, operation="add", **kwargs) -> ToolResult:
        ops = {"add": lambda: a + b, "multiply": lambda: a * b}
        if operation not in ops:
            raise ValueError(f"Unknown operation {operation!r}")
        return ToolResult(self.name, str(ops[operation]()))


class SearchTool(BaseTool):
    def __init__(self, max_results: int = 5):
        super().__init__("search", "Returns top N results for a query")
        self._max_results = max_results

    def run(self, query: str, **kwargs) -> ToolResult:
        return ToolResult(
            self.name,
            f"Top {self._max_results} results for '{query}'",
            metadata={"query": query, "max_results": self._max_results}
        )
```

#### 10.2 Session 2 (1h): `Memory`, `BaseAgent`, `SequentialChain`

```python
from collections import deque
import copy

class Memory:
    def __init__(self, maxsize: int = 5):
        self._store: deque = deque(maxlen=maxsize)
        self._maxsize = maxsize

    def add(self, turn: dict):
        if not isinstance(turn, dict):
            raise TypeError(f"Memory stores dicts, got {type(turn).__name__}")
        self._store.append(copy.copy(turn))   # shallow copy each turn -- safe for string-valued turns

    def get_all(self) -> list[dict]:
        return list(self._store)

    def clear(self):
        self._store.clear()

    def __len__(self):      return len(self._store)
    def __iter__(self):     return iter(self._store)
    def __getitem__(self, i): return list(self._store)[i]
    def __repr__(self):
        return f"Memory(stored={len(self)}/{self._maxsize})"


class BaseAgent:
    def __init__(self, name: str, memory_size: int = 5):
        self._name = name
        self.memory = Memory(maxsize=memory_size)
        self.tools = ToolBox()
        self._state: dict = {}

    @property
    def name(self) -> str: return self._name

    def add_tool(self, tool: BaseTool) -> "BaseAgent":
        self.tools.add(tool)
        return self   # enables chaining: agent.add_tool(a).add_tool(b)

    def update_state(self, update: dict) -> None:
        self._state = {**self._state, **update}

    def get_state(self) -> dict:
        return dict(self._state)

    def reset(self) -> None:
        self._state = {}
        self.memory.clear()

    def run_tool(self, tool_name: str, **kwargs) -> ToolResult:
        result = self.tools(tool_name, **kwargs)
        self.update_state({"last_tool": tool_name, "last_result": result})
        return result

    def __enter__(self):
        print(f"[{self._name}] initialized")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f"[{self._name}] cleaned up")
        return False

    def __repr__(self):
        return (f"BaseAgent(name={self._name!r}, "
                f"memory={len(self.memory)}, tools={self.tools})")


class SequentialChain:
    def __init__(self, steps: list):
        for s in steps:
            if not callable(s):
                raise TypeError(f"All steps must be callable, got {type(s).__name__}")
        self.steps = steps
        self._run_count = 0

    def __call__(self, initial_state: dict) -> dict:
        self._run_count += 1
        state = dict(initial_state)
        for step in self.steps:
            update = step(state)
            if not isinstance(update, dict):
                raise TypeError(f"Step {step!r} must return a dict, got {type(update).__name__}")
            state = {**state, **update}
        return state

    def __len__(self): return len(self.steps)
    def __repr__(self): return f"SequentialChain(steps={len(self)}, runs={self._run_count})"
```

#### 10.3 Session 3 (1h): Wire it all together and verify

```python
# ──── node functions (the LangGraph node shape: state -> dict) ────
def research_node(state: dict) -> dict:
    query = state.get("brief", "general research")
    result = state["_agent"].run_tool("search", query=query)
    return {"findings": result.output, "research_done": True}

def analysis_node(state: dict) -> dict:
    findings = state.get("findings", "")
    result = state["_agent"].run_tool("calculator", a=len(findings), b=2, operation="multiply")
    return {"analysis_score": result.output}

def report_node(state: dict) -> dict:
    return {
        "report": (
            f"BRIEF: {state.get('brief')}\n"
            f"FINDINGS: {state.get('findings')}\n"
            f"SCORE: {state.get('analysis_score')}"
        )
    }

# ──── assemble ────
with BaseAgent("ev_researcher", memory_size=10) as agent:
    agent.add_tool(SearchTool(max_results=5)).add_tool(CalculatorTool())

    chain = SequentialChain([research_node, analysis_node, report_node])

    initial_state = {"brief": "EV charging infrastructure in India", "_agent": agent}
    final_state = chain(initial_state)

    agent.memory.add({"input": initial_state["brief"], "output": final_state["report"]})

    print(final_state["report"])
    print()
    print("agent:", agent)
    print("chain:", chain)
    print("memory entries:", len(agent.memory))
    print("search calls:", agent.tools["search"]._call_count)
    print("calc calls:", agent.tools["calculator"]._call_count)
```

**Capstone acceptance checklist:**
- [ ] `BaseTool` is an ABC; instantiating it without `run()` raises `TypeError`
- [ ] `ToolBox` supports `len()`, `in`, `[]`, `for`, and `()` -- all implemented
- [ ] `ToolResult` is a `@dataclass` with `__post_init__` validation
- [ ] `Memory` has a hard cap via `deque(maxlen=N)` -- verified by adding N+1 entries
- [ ] `BaseAgent` is a context manager -- `with` block logs enter and exit
- [ ] `SequentialChain` is callable (`chain(state)`) and threads state correctly
- [ ] All concrete `BaseTool` subclasses satisfy `isinstance(tool, Runnable)` via Protocol
- [ ] `agent.add_tool(a).add_tool(b)` works (method chaining via `return self`)
- [ ] The full `research → analysis → report` chain produces a non-empty report
- [ ] **Recording:** explain the design of each class using only the three terms from the intro: "IS-A (inheritance)", "HAS-A (composition)", and "behaves-like (Protocol)" — no other jargon

---
