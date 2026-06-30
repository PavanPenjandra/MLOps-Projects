### Module 3 — Dunder Methods I: Representation and Comparison (Hours 7–9)
**`__repr__`, `__str__`, `__eq__`, `__hash__` — the four methods every class you write should have**

#### 3.1 Concept Lecture — The dunder protocol: Python's method dispatch system (1h)

**Why this matters for LangGraph:** when you `print(state)`, Python calls `state.__repr__()`. When you check `tool_a == tool_b`, Python calls `tool_a.__eq__(tool_b)`. When you put a tool in a `set()`, Python calls `tool.__hash__()`. These are not optional extras — they are the mechanism behind every natural Python operation your objects will ever participate in. LangChain objects implement all four, which is exactly why they print usefully, compare correctly, and can be deduplicated in sets.

**The naming convention:** any method surrounded by double underscores on both sides (`__name__`) is called a **dunder** (double-underscore) method, or sometimes a "magic method." They're not actually magic — they're regular methods Python calls implicitly in specific situations.

```python
class ToolResult:
    def __init__(self, tool_name, output, success=True):
        self.tool_name = tool_name
        self.output = output
        self.success = success

result = ToolResult("search", "found 3 results")
print(result)   # <__main__.ToolResult object at 0x7f...>  -- useless default
```

**Fix it with `__repr__` and `__str__`:**

```
              __repr__                          __str__
         ┌──────────────────┐             ┌───────────────────┐
         │ Unambiguous repr │             │ Human-readable    │
         │ for DEVELOPERS   │             │ for USERS         │
         │ eval(repr(x))==x │             │ print(x) uses it  │
         │ always implement │             │ falls back to repr │
         └──────────────────┘             └───────────────────┘
```

**Rule of thumb:** always implement `__repr__`. Only implement `__str__` when you want a different, friendlier string for end-user display.

```python
class ToolResult:
    def __init__(self, tool_name, output, success=True):
        self.tool_name = tool_name
        self.output = output
        self.success = success

    def __repr__(self):
        return (
            f"ToolResult(tool_name={self.tool_name!r}, "
            f"output={self.output!r}, success={self.success!r})"
        )
        # Note the !r format spec -- it calls repr() on the value,
        # adding quotes around strings automatically

    def __str__(self):
        status = "✓" if self.success else "✗"
        return f"[{status}] {self.tool_name}: {self.output}"

r = ToolResult("search", "found 3 results")
print(repr(r))   # ToolResult(tool_name='search', output='found 3 results', success=True)
print(str(r))    # [✓] search: found 3 results
print(r)          # [✓] search: found 3 results -- print() calls __str__
```

**`__eq__` and `__hash__` — they must be implemented together:**

```python
class ToolResult:
    # ... (above attributes + __repr__ + __str__)

    def __eq__(self, other):
        if not isinstance(other, ToolResult):
            return NotImplemented   # not False -- NotImplemented lets Python try the other side
        return (self.tool_name == other.tool_name and
                self.output == other.output and
                self.success == other.success)

    def __hash__(self):
        return hash((self.tool_name, self.output, self.success))
        # hash a TUPLE of the fields -- tuples are hashable, the tuple hash is deterministic
```

**The critical rule Python enforces:** if you define `__eq__`, Python *sets `__hash__` to `None`* — making your object unhashable (can't be put in a `set` or used as a `dict` key) unless you also explicitly define `__hash__`. This is intentional: if two objects are "equal" (`__eq__` returns True), they must have the same hash. Failing to implement both breaks that contract.

```python
r1 = ToolResult("search", "found 3")
r2 = ToolResult("search", "found 3")
r3 = ToolResult("calc",   "42")

print(r1 == r2)              # True
print(r1 == r3)              # False
print(hash(r1) == hash(r2))  # True
print(len({r1, r2, r3}))     # 2 -- r1 and r2 are deduplicated in a set
```

**Pitfall:** returning `False` from `__eq__` when comparing against an incompatible type, instead of `NotImplemented`. Returning `False` silently prevents the *other* object from getting a chance to define equality with yours. `NotImplemented` tells Python "I don't know — ask the other side."

#### 3.2 Guided Coding Drill — `__repr__`, `__str__`, `__eq__`, `__hash__` on a State class (1h)

```python
class AgentState:
    def __init__(self, brief, findings=None, approved=False):
        self.brief = brief
        self.findings = findings or []
        self.approved = approved

    def __repr__(self):
        return (
            f"AgentState(brief={self.brief!r}, "
            f"findings={self.findings!r}, approved={self.approved!r})"
        )

    def __str__(self):
        status = "APPROVED" if self.approved else "PENDING"
        n = len(self.findings)
        return f"AgentState[{status}] brief={self.brief!r}, {n} finding(s)"

    def __eq__(self, other):
        if not isinstance(other, AgentState):
            return NotImplemented
        return (self.brief == other.brief and
                self.findings == other.findings and
                self.approved == other.approved)

    def __hash__(self):
        return hash((self.brief, tuple(self.findings), self.approved))
        # findings is a list -- lists aren't hashable -- convert to tuple first

s1 = AgentState("EV market", ["finding a", "finding b"])
s2 = AgentState("EV market", ["finding a", "finding b"])
s3 = AgentState("Solar market")

print(repr(s1))
print(str(s1))
print("s1 == s2:", s1 == s2)
print("s1 == s3:", s1 == s3)
print("dedup in set:", len({s1, s2, s3}))   # 2
print("as dict key:", {s1: "first run", s3: "second run"})
```

**Drill:** add a method `to_dict(self) -> dict` that returns a plain `dict` representation of the state (so it can be serialized to JSON later), and add a `classmethod from_dict(cls, d)` that reconstructs an `AgentState` from that dict. Confirm that `AgentState.from_dict(s1.to_dict()) == s1`.

#### 3.3 Homework/Challenge — Immutable ToolResult (1h)

By default, your `ToolResult` can be mutated after creation: `r.output = "different"` works fine and silently changes a result that was supposed to be a record. Make `ToolResult` *de facto* immutable:
1. Override `__setattr__` to raise `AttributeError` for any attribute assignment *after* `__init__` has finished
2. Override `__delattr__` similarly
3. Keep `__repr__`, `__eq__`, `__hash__`

**Hint:** use an instance flag `self._initialized` (set it via `object.__setattr__(self, '_initialized', False)` in `__init__` to bypass your own guard, then set it to `True` at the very end of `__init__`).

**Challenge:** instead of the flag trick, make `ToolResult` a subclass of `tuple` — store the three fields as the tuple's contents, implement `__new__` instead of `__init__`, and add `@property` methods to expose `tool_name`, `output`, and `success` by position.

---

### Module 4 — Dunder Methods II: Container and Callable Behavior (Hours 10–12)
**`__len__`, `__contains__`, `__getitem__`, `__iter__`, `__call__` — make your objects behave natively**

#### 4.1 Concept Lecture — Writing objects that behave like built-ins (1h)

**Why this matters for LangGraph:** `len(memory)` working on a `Memory` object, `"finding" in results` working on a `ResultSet`, a `Chain` object being callable like `chain(state)` — these are not framework tricks, they're dunder methods. The moment you implement them, your class "speaks Python natively" and any library — including LangChain — can use it with zero friction.

**The four container dunder methods:**

| Method | Triggered by | Example |
|---|---|---|
| `__len__` | `len(obj)` | `len(memory)` — how many turns stored |
| `__contains__` | `x in obj` | `"search" in tool_registry` |
| `__getitem__` | `obj[key]` | `memory[0]` or `state["findings"]` |
| `__iter__` | `for x in obj` | `for tool in toolbox:` |

```python
from collections import deque

class Memory:
    def __init__(self, maxsize=5):
        self._store = deque(maxlen=maxsize)
        self._maxsize = maxsize

    def add(self, turn: dict):
        if not isinstance(turn, dict):
            raise TypeError(f"Expected dict, got {type(turn).__name__}")
        self._store.append(turn)

    def __len__(self):
        return len(self._store)

    def __contains__(self, item):
        return item in self._store    # checks if the dict is in the deque

    def __getitem__(self, index):
        items = list(self._store)
        return items[index]           # supports negative indexing, slicing -- free via list

    def __iter__(self):
        return iter(self._store)      # lets: for turn in memory:

    def __repr__(self):
        return f"Memory(size={len(self)}/{self._maxsize})"

mem = Memory(maxsize=3)
mem.add({"user": "hello", "ai": "hi"})
mem.add({"user": "EV market?", "ai": "Growing 20% YoY"})
mem.add({"user": "Key players?", "ai": "Tesla, BYD"})
mem.add({"user": "4th turn", "ai": "1st evicted"})

print(repr(mem))                 # Memory(size=3/3)
print(len(mem))                  # 3
print(mem[-1])                    # most recent turn
print({"user": "Key players?", "ai": "Tesla, BYD"} in mem)   # True

for turn in mem:
    print(f"  USER: {turn['user']}")
```

**`__call__` — making an instance callable:**

```python
class ChainRunner:
    """A chain of nodes, callable like a plain function."""

    def __init__(self, steps: list):
        self.steps = steps
        self._run_count = 0

    def __call__(self, state: dict) -> dict:
        self._run_count += 1
        for step in self.steps:
            update = step(state)       # each step is also callable
            state = {**state, **update}
        return state

    def __repr__(self):
        return f"ChainRunner(steps={len(self.steps)}, runs={self._run_count})"

def researcher(state): return {"raw": f"Researched: {state['brief']}"}
def formatter(state):  return {"report": f"REPORT: {state['raw']}"}

chain = ChainRunner([researcher, formatter])
print("callable?", callable(chain))     # True
result = chain({"brief": "EV market", "raw": "", "report": ""})
print(result)
print(chain)                             # ChainRunner(steps=2, runs=1)
```

**`callable()` — the built-in that checks if something is callable:** works on functions, classes (calling a class creates an instance), and any object with `__call__`. This is exactly how LangChain's LCEL `|` chain operator works internally — it checks whether each component is callable in the right way, using `isinstance` against `Runnable`.

#### 4.2 Guided Coding Drill — A `ToolBox` container class (1h)

Build a `ToolBox` class that stores tools by name, supports `len()`, `in` checks, indexing by name, and iteration — plus a `__call__` method that runs a tool by name:

```python
class ToolBox:
    def __init__(self):
        self._tools = {}

    def add(self, tool):
        self._tools[tool.name] = tool

    def __len__(self):
        return len(self._tools)

    def __contains__(self, name):
        return name in self._tools

    def __getitem__(self, name):
        if name not in self._tools:
            raise KeyError(f"No tool named {name!r}. Available: {list(self._tools)}")
        return self._tools[name]

    def __iter__(self):
        return iter(self._tools.values())   # iterate over TOOL OBJECTS, not names

    def __call__(self, tool_name, *args, **kwargs):
        return self[tool_name].run(*args, **kwargs)

    def __repr__(self):
        names = list(self._tools.keys())
        return f"ToolBox(tools={names})"

# Using it:
box = ToolBox()
box.add(CalculatorTool())
box.add(SearchTool())

print(repr(box))
print(len(box))                         # 2
print("calculator" in box)              # True
print(box["calculator"].description)    # via __getitem__
print(box("calculator", 3, 7))          # via __call__

for tool in box:                         # via __iter__
    print(f"  {tool.name}: {tool.description}")
```

**Drill:** add a `__delitem__` method so `del box["search"]` removes a tool, and a `keys()` method that returns the tool names — making `ToolBox` feel like a dict from the outside, while hiding the internal storage details.

#### 4.3 Homework/Challenge — A `ResultHistory` with slicing and filtering (1h)

Build a `ResultHistory` class that:
1. Stores `ToolResult` objects (from Module 3)
2. Implements `__len__`, `__getitem__` (with support for both integer index AND slicing — check `isinstance(index, slice)`), `__iter__`, `__contains__`
3. Adds a `filter(fn)` method returning a NEW `ResultHistory` containing only results where `fn(result)` is `True`

**Challenge:** implement `__add__` so two `ResultHistory` objects can be combined with `+`, returning a third new `ResultHistory` containing all results from both — the exact behavior you'd want when merging specialist results in a fan-out pattern.

---

### Module 5 — Inheritance, `super()`, and the MRO (Hours 13–15)
**The machine Python actually runs when you call `super()` is not what you think**

#### 5.1 Concept Lecture — Single inheritance, `super()`, and why it's not "call the parent" (1h)

**Why this matters for LangGraph:** every custom tool, model, and runnable in LangChain starts with `super().__init__(...)`. If you think `super()` means "call my direct parent class," you are one step away from a silent bug when multiple inheritance or mixins enter the picture — which they do, in the LangChain source, constantly.

**The correct mental model for `super()`:**

> `super()` does not call the parent class. It calls **the next class in the MRO** — the Method Resolution Order — a linearized sequence Python computes for every class at definition time.

```python
class A:
    def greet(self): return "A"

class B(A):
    def greet(self): return "B -> " + super().greet()

class C(A):
    def greet(self): return "C -> " + super().greet()

class D(B, C):
    def greet(self): return "D -> " + super().greet()

print([cls.__name__ for cls in D.__mro__])
# ['D', 'B', 'C', 'A', 'object']

print(D().greet())
# D -> B -> C -> A
```

**ASCII trace of the MRO call chain:**
```
D().greet() called
     │
     ▼
D.greet()  -- calls super().greet()
     │        super() for D == next in MRO == B
     ▼
B.greet()  -- calls super().greet()
     │        super() for B (in context of D's MRO) == C, NOT A
     ▼
C.greet()  -- calls super().greet()
     │        super() for C (in context of D's MRO) == A
     ▼
A.greet()  -- returns "A", no super() call here
     │
     ◄──── the return values bubble back: "A"
          C returns "C -> A"
          B returns "B -> C -> A"
          D returns "D -> B -> C -> A"
```

**Why "call the parent" as a mental model breaks:** in `B.greet()`, `super().greet()` calls `C`, not `A` — because Python computed the full MRO of `D` before any method was called, and every `super()` call in the chain uses that same ordering. `super()` is a *cooperative* mechanism — every class in the chain is expected to call `super()` too, forwarding the chain.

**The C3 Linearization algorithm (you don't need to compute it, but you need to know it exists):** Python uses a specific algorithm to compute the MRO, guaranteeing that each class appears exactly once, parents always come after their children, and the local precedence order of parents is preserved. You can always inspect the MRO with `ClassName.__mro__` or `ClassName.mro()`.

**`super()` in `__init__` — the correct pattern:**
```python
class BaseTool:
    def __init__(self, name, description):
        self.name = name
        self.description = description

class SpecializedTool(BaseTool):
    def __init__(self, name, description, max_results=10):
        super().__init__(name, description)    # hands name & description UP the chain
        self.max_results = max_results         # adds its own attribute

t = SpecializedTool("search", "web search", max_results=5)
print(t.name, t.description, t.max_results)
```

**Pitfall:** forgetting `super().__init__()` in a subclass. The object is created (memory allocated, `__new__` has run) but none of the parent's setup happens — `self.name` never gets set, so the first access crashes with `AttributeError` rather than at construction time, which is confusing to debug.

#### 5.2 Guided Coding Drill — A 3-level tool hierarchy with MRO tracing (1h)

```python
class BaseTool:
    def __init__(self, name, description):
        self.name = name
        self.description = description
        print(f"  BaseTool.__init__ called for {name!r}")

    def run(self, *args, **kwargs):
        raise NotImplementedError(f"{type(self).__name__} must implement run()")

    def invoke(self, state: dict) -> dict:
        """The LangGraph-compatible entry point."""
        try:
            result = self.run(**{k: v for k, v in state.items() if k != "history"})
            return {"result": result, "error": None}
        except Exception as e:
            return {"result": None, "error": str(e)}


class SearchBaseTool(BaseTool):
    def __init__(self, name, description, max_results=5):
        super().__init__(name, description)   # calls BaseTool.__init__
        self.max_results = max_results
        print(f"  SearchBaseTool.__init__ called with max_results={max_results}")

    def run(self, query, **kwargs):
        return [f"result_{i} for '{query}'" for i in range(self.max_results)]


class PaginatedSearchTool(SearchBaseTool):
    def __init__(self, max_results=5, page_size=2):
        super().__init__("paginated_search", "Paginates results", max_results)
        self.page_size = page_size
        print(f"  PaginatedSearchTool.__init__ called with page_size={page_size}")

    def run(self, query, page=0, **kwargs):
        all_results = super().run(query)    # calls SearchBaseTool.run
        start = page * self.page_size
        return all_results[start:start + self.page_size]


print("Creating PaginatedSearchTool:")
tool = PaginatedSearchTool(max_results=6, page_size=2)
print("\nMRO:", [c.__name__ for c in PaginatedSearchTool.__mro__])
print("\nPage 0:", tool.run("EV market", page=0))
print("Page 1:", tool.run("EV market", page=1))
```

**Drill:** add a method `full_spec(self)` at the `BaseTool` level that calls `super().full_spec()` if the superclass has it (use `hasattr(super(), 'full_spec')`), otherwise starts the string, letting each subclass add its own line. Confirm that calling `tool.full_spec()` on a 3-level instance produces a correctly accumulated string.

#### 5.3 Homework/Challenge — MRO prediction exercise (1h)

For each of the following class hierarchies, **predict the full MRO and the output of calling `greet()`** before running the code. Then run it and compare:

```python
# A: Diamond with override at D only
class W:
    def greet(self): return "W"

class X(W):
    pass    # no override

class Y(W):
    def greet(self): return "Y->" + super().greet()

class Z(X, Y):
    pass   # no override

# Predict: Z().greet() = ?
# Predict: Z.__mro__ = ?
```

```python
# B: Each class in chain calls super() correctly
class Logging:
    def run(self, *a, **kw):
        print("logging start")
        result = super().run(*a, **kw)
        print("logging end")
        return result

class Retrying:
    def run(self, *a, **kw):
        print("retry wrapper")
        return super().run(*a, **kw)

class ActualWorker:
    def run(self, x):
        return x * 2

class ProductionTool(Logging, Retrying, ActualWorker):
    pass

# Predict: ProductionTool().run(5) output?
```

**Challenge:** the `ProductionTool` above has no `__init__` and `BaseTool` is not in its hierarchy. Add `BaseTool` into the hierarchy while keeping all three mixin behaviors working — without breaking the MRO (Python will raise a `TypeError` if your class hierarchy has a MRO conflict; prove you can resolve it).

---

### Module 6 — Abstract Base Classes and Protocols (Hours 16–18)
**The two ways Python enforces contracts — and why they're both in LangChain**

#### 6.1 Concept Lecture — ABCs enforce at construction; Protocols enforce at check time (1h)

**Why this matters for LangGraph:** `BaseTool` in LangChain uses `ABC` to guarantee that any concrete tool class must implement `run()` — trying to instantiate it without an implementation raises `TypeError` immediately. The `Runnable` interface, on the other hand, is a `Protocol` — you don't inherit from it, you just happen to have the right methods, and Python can check whether you qualify at runtime without any class relationship declared.

**Two completely different mechanisms, two completely different use cases:**

```
Abstract Base Class (ABC)               Protocol
─────────────────────────               ──────────────────────────────
YOU INHERIT from it                     YOU DO NOT inherit from it
├── IS-A relationship enforced          ├── Duck typing ("has-a method")
├── Raises TypeError at instantiation   ├── Checked with isinstance() at
│   if @abstractmethod not implemented  │   runtime (if @runtime_checkable)
└── Use when: "you MUST be one of us"   └── Use when: "you can do this job"

Example in LangChain:                   Example in LangChain:
class CalculatorTool(BaseTool):         class MyChain:
    def run(self, a, b): ...                def invoke(self, state): ...
                                        # no inheritance -- it IS a Runnable
                                        # because it has .invoke()
```

**ABC in code:**
```python
from abc import ABC, abstractmethod

class BaseTool(ABC):
    def __init__(self, name, description):
        self.name = name
        self.description = description

    @abstractmethod
    def run(self, *args, **kwargs):
        """Every concrete tool MUST implement this. ABC enforces it."""
        pass                # body doesn't matter -- it's never called directly

    def invoke(self, state: dict) -> dict:
        """Concrete method -- subclasses inherit this for free."""
        try:
            result = self.run(**state)
            return {"result": result, "error": None}
        except Exception as e:
            return {"result": None, "error": str(e)}

try:
    BaseTool("x", "y")    # TypeError -- can't instantiate without run()
except TypeError as e:
    print("ABC enforcement:", e)

class GoodTool(BaseTool):
    def run(self, query, **kwargs):
        return f"ran: {query}"

t = GoodTool("search", "searches")    # works -- run() is implemented
print(t.invoke({"query": "EV market"}))
```

**Protocol in code:**
```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class Runnable(Protocol):
    def invoke(self, state: dict) -> dict: ...

class LLMNode:
    def invoke(self, state):
        return {"output": f"LLM says: {state.get('input','')}"}

class NotRunnable:
    def execute(self, x): return x     # different method name

print(isinstance(LLMNode(), Runnable))      # True -- has .invoke()
print(isinstance(NotRunnable(), Runnable))  # False -- wrong method name
print(isinstance(GoodTool("t","d"), Runnable))  # True -- inherits invoke() from BaseTool
```

**Why both are used together:** `BaseTool(ABC)` guarantees every tool has `run()` implemented. `Runnable(Protocol)` lets ANY object — tools, LLM wrappers, chains, custom classes — be treated identically as long as they have `invoke()`, regardless of their inheritance hierarchy. This is exactly the design LangChain uses: strong enforcement within the tool family (ABC), flexible interoperability across component types (Protocol).

#### 6.2 Guided Coding Drill — Building a complete `BaseTool(ABC)` + `Runnable` system (1h)

```python
from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable
from dataclasses import dataclass

@dataclass
class ToolResult:
    tool_name: str
    output: str
    success: bool = True
    error: str = ""

@runtime_checkable
class Runnable(Protocol):
    def invoke(self, state: dict) -> dict: ...

class BaseTool(ABC):
    def __init__(self, name: str, description: str):
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name must be a non-empty string")
        self._name = name
        self._description = description

    @property
    def name(self): return self._name

    @property
    def description(self): return self._description

    @abstractmethod
    def run(self, *args, **kwargs) -> ToolResult:
        """Execute the tool and return a ToolResult."""
        pass

    def invoke(self, state: dict) -> dict:
        """Standard Runnable interface -- every BaseTool IS a Runnable."""
        try:
            tool_kwargs = {k: v for k, v in state.items()
                          if k not in ("history", "thread_id")}
            result = self.run(**tool_kwargs)
            return {"tool_result": result, "error": None}
        except Exception as e:
            return {
                "tool_result": ToolResult(self._name, "", success=False, error=str(e)),
                "error": str(e)
            }

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self._name!r})"

class CalculatorTool(BaseTool):
    def __init__(self):
        super().__init__("calculator", "Adds or multiplies two numbers")

    def run(self, a, b, operation="add", **kwargs) -> ToolResult:
        ops = {"add": a + b, "multiply": a * b}
        if operation not in ops:
            raise ValueError(f"Unknown operation: {operation!r}")
        return ToolResult(self.name, str(ops[operation]))

class SearchTool(BaseTool):
    def __init__(self, max_results=5):
        super().__init__("search", "Searches for information")
        self._max_results = max_results

    def run(self, query, **kwargs) -> ToolResult:
        return ToolResult(self.name, f"Top {self._max_results} results for '{query}'")

# Verify the full system
calc = CalculatorTool()
search = SearchTool(max_results=3)

print(isinstance(calc, Runnable))     # True
print(isinstance(search, Runnable))   # True
print(calc.invoke({"a": 3, "b": 7, "operation": "multiply"}))
print(search.invoke({"query": "EV market trends", "history": ["ignored"]}))
print(calc.invoke({"a": 3, "b": 7, "operation": "divide"}))   # handled error
```

**Drill:** add a `ToolRegistry` class that only accepts objects satisfying `isinstance(tool, Runnable)`, raises `TypeError` otherwise, and can batch-invoke all registered tools against the same state dict — returning a list of all results.

#### 6.3 Homework/Challenge — Multiple abstract methods with a partial template (1h)

Real-world ABCs often have multiple abstract methods AND some concrete "template method" methods that call the abstract ones. Build a `BaseResearchSpecialist(ABC)` with:
- `@abstractmethod research(self, brief) -> list[str]` — the raw findings
- `@abstractmethod format_findings(self, findings) -> str` — how to format them
- A concrete `produce_report(self, brief) -> str` that calls both (template method pattern)

```python
class WebResearchSpecialist(BaseResearchSpecialist):
    def research(self, brief):
        return [f"Web finding {i} about {brief}" for i in range(3)]
    def format_findings(self, findings):
        return "\n".join(f"  • {f}" for f in findings)

class FinancialResearchSpecialist(BaseResearchSpecialist):
    def research(self, brief):
        return [f"Financial metric {i} for {brief}" for i in range(2)]
    def format_findings(self, findings):
        return " | ".join(findings)

# Both should produce a report via the inherited produce_report()
```

**Challenge:** add a `@classmethod get_specialist(cls, kind: str)` to `BaseResearchSpecialist` that looks up registered subclasses and instantiates the right one — implement the registration by having each subclass call `BaseResearchSpecialist.register(cls)` in its own class body using a class-level dict.

---
