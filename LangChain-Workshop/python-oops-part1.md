# Python OOP: Zero to Expert in 30 Hours
### For developers who already know OOP — and need to unlearn three things before Python's version will make sense

> You know classes. You know inheritance. You've probably written `public void main(String[] args)` or the JavaScript equivalent of a constructor at least a hundred times. This course is not "what is a class." It is "here are the nine ways Python OOP is *different* from what you know, why each difference exists, and how every single one of those differences shows up directly in LangChain's and LangGraph's source code." By Hour 30 you will not just write Python classes — you will *read* `BaseTool`, `BaseModel`, and `StateGraph` and understand every design decision in them at the language level.

---

## The Three Things to Unlearn First

Before the course starts, three mental models need adjusting — not because they're wrong in general, but because they're wrong for Python specifically:

| What other languages taught you | What Python actually does | Why it matters for LangChain |
|---|---|---|
| "Private means inaccessible" | Python has no truly private attributes — `_x` is a *convention* ("please don't touch"), `__x` is *name-mangling* (harder but not impossible to reach from outside) | Every LangChain base class uses `_x` for internal fields you'll need to understand, not avoid |
| "Interfaces are a separate construct" | Python uses **Protocols** and **duck typing** — if your object has the right methods, it *is* that interface, no declaration required | The `Runnable` interface in LangChain is a Protocol, not a formal Java-style interface |
| "Inheritance is the main reuse tool" | Python strongly prefers **composition** and **mixins** for reuse; deep inheritance trees are a code smell here, not good design | LangChain tools are composed of behaviors (logging, retrying, streaming) glued via mixins, not a 6-level inheritance chain |

---

## 0. How This Course Is Organized

**10 modules, 3 hours each. Every module follows this rhythm — always:**

| Block | Time | What happens |
|---|---|---|
| **Concept Lecture** | 1h | The Python-specific mechanism, with whiteboard-style ASCII diagrams and the LangGraph analogy stated immediately |
| **Guided Coding Drill** | 1h | You build the concept in the context of an agentic system component — a tool, a node, a chain |
| **Homework/Challenge** | 1h | An unguided extension — you extend the drill toward something a real agent project would actually need |

**The capstone:** Module 10 (Hours 28–30) assembles every pattern into a full, clean class hierarchy representing a minimal agentic system — `BaseTool`, `BaseAgent`, `SequentialChain`, `Memory` — in pure Python, no library imports, such that a LangChain developer would look at it and say "yes, this is the right shape."

---

## Part 1 — Detailed Syllabus

| Module | Topic | LangGraph Analogy | Hours |
|---|---|---|---|
| 1 | Python's Object Model — everything is an object | Why `graph` and `node_function` are both first-class objects you can pass around | 1–3 |
| 2 | `__init__`, `self`, instance vs. class attributes | `BaseTool.__init__` stores `name` and `description`; the class attribute `registry` tracks all instances | 4–6 |
| 3 | Dunder Methods I — `__repr__`, `__str__`, `__eq__`, `__hash__` | Why two state dicts with the same content should be equal, and when that breaks things | 7–9 |
| 4 | Dunder Methods II — `__len__`, `__contains__`, `__call__`, `__getitem__` | A `Memory` object you can `len()`, a `Chain` you can call like a function | 10–12 |
| 5 | Inheritance, `super()`, and the MRO | How `CalculatorTool(BaseTool)` gets `invoke()` for free, and why `super()` in multiple inheritance isn't what you expect | 13–15 |
| 6 | Abstract Base Classes and Protocols | Why LangChain enforces `run()` at the ABC level, and why `Runnable` is a Protocol not a class | 16–18 |
| 7 | `@property`, `@classmethod`, `@staticmethod` | A tool's `name` is a computed property with a validation setter; `from_dict()` is a classmethod factory | 19–21 |
| 8 | Composition, Mixins, and the "has-a" design | `ChatAgent` HAS-A `Memory`, doesn't inherit it; `LogMixin` + `RetryMixin` gives a tool two behaviors without a deep hierarchy | 22–24 |
| 9 | `dataclasses`, `__slots__`, and Descriptors | `ToolResult` as a dataclass; `Validated` as a reusable descriptor for field-level type enforcement | 25–27 |
| 10 | Capstone — Build a complete agent class hierarchy | The whole thing assembled: `BaseTool`, `BaseAgent`, `Memory`, `Chain`, all wired together | 28–30 |

**Total: 30.0 hours.**

---

## Five-Week Schedule (6 hours/week)

| Week | Modules | Content |
|---|---|---|
| **Week 1** | 1–2 | Python's object model, `__init__`, `self`, class vs. instance attributes — the foundational corrections |
| **Week 2** | 3–4 | All eight dunder methods — the mechanism behind every "Pythonic" thing LangChain does |
| **Week 3** | 5–6 | Inheritance, MRO, ABC, Protocol — reading and writing the full tool/runnable hierarchy |
| **Week 4** | 7–8 | `@property`/classmethod/staticmethod, composition, mixins — the design patterns in the actual source code |
| **Week 5** | 9–10 | Dataclasses, slots, descriptors, capstone — ship the full hierarchy |

---

## Part 2 — Module-by-Module Content

---

### Module 1 — Python's Object Model (Hours 1–3)
**"Everything is an object" is not a slogan — it's load-bearing architecture**

#### 1.1 Concept Lecture — What "everything is an object" actually means (1h)

**Why this matters for LangGraph, immediately:** in Python, a *function* is an object. A *class* is an object. A *module* is an object. This means you can pass a node function as an argument, store it in a list, and call it later — which is exactly what `StateGraph.add_node(name, fn)` does internally. If you come from a language where functions are "just functions" (not first-class objects), this one fact unlocks everything.

```python
def researcher_node(state):
    return {"raw_findings": f"Researched: {state['brief']}"}

# Functions are objects -- they have attributes
print(type(researcher_node))        # <class 'function'>
print(researcher_node.__name__)      # 'researcher_node'

# They can be passed around like any other value
steps = [researcher_node]           # a list CONTAINING a function
steps[0]({"brief": "EV market"})    # call it
```

**The Python object model — ASCII diagram:**
```
Everything in memory is a PyObject:

  ┌─────────────────────────────────────────────────────┐
  │  PyObject (every value in Python)                   │
  │  ┌───────────┐  ┌───────────┐  ┌──────────────────┐ │
  │  │ type      │  │ reference │  │ value / pointer  │ │
  │  │ (what am  │  │ count     │  │ to actual data   │ │
  │  │  I?)      │  │           │  │                  │ │
  │  └───────────┘  └───────────┘  └──────────────────┘ │
  └─────────────────────────────────────────────────────┘

  "hello"   -->  PyObject(type=str,   value=...)
  42        -->  PyObject(type=int,   value=...)
  [1,2,3]   -->  PyObject(type=list,  value=...)
  my_fn     -->  PyObject(type=function, value=code_object)
  MyClass   -->  PyObject(type=type,  value=class_definition)
```

**`id()` and identity — the tool that proves it:**
```python
x = "LangChain"
y = x
z = "LangChain"

print(id(x), id(y), id(z))     # x and y are SAME object (same id); z may or may not share
print(x is y)                    # True  -- same object
print(x == z)                    # True  -- same VALUE

state_a = {"brief": "EV"}
state_b = state_a               # state_b points at THE SAME DICT
state_b["new_key"] = "added"
print(state_a)                   # {'brief': 'EV', 'new_key': 'added'} -- state_a "changed" too
```

**Attributes and `type()`:** every object carries its type with it, and you can interrogate it at any time:
```python
items = [1, 2, 3]
print(type(items))               # <class 'list'>
print(type(items).__name__)      # 'list'
print(isinstance(items, list))   # True
print(isinstance(items, (list, tuple)))   # True -- isinstance accepts a tuple of types
```

**The LangGraph tie-in, concrete:** `isinstance(node_fn, Runnable)` is how LangGraph validates that what you passed as a node is actually callable in the way it expects. This `isinstance` check works because `Runnable` is a Protocol (Module 6) and Python's type system is fully inspectable at runtime.

**Pitfall:** `type(x) == list` vs. `isinstance(x, list)`. The `type()` check is exact — a `list` subclass fails it. `isinstance()` checks the full inheritance chain, so a subclass of `list` passes correctly. Always use `isinstance()` for type-checking in production code.

**Exercise:** create any 5 Python objects (a string, a function, a class itself, a list, an integer). Print `type(x).__name__`, `id(x)`, and `callable(x)` for each. Notice that the *class object itself* is callable (calling it creates an instance).

#### 1.2 Guided Coding Drill — Object identity and mutation in agent state (1h)

**The drill.** Build a small simulation showing precisely when state mutation is a bug vs. a feature:

```python
import copy

# Scenario 1: TWO names, ONE object -- mutation affects both
state_original = {"findings": ["a"], "brief": "EV market"}
state_alias = state_original          # both names, one dict

state_alias["findings"].append("b")
print("Scenario 1 (alias mutation):")
print("  original:", state_original["findings"])   # ['a', 'b'] -- aliased!
print("  alias:   ", state_alias["findings"])      # ['a', 'b']

# Scenario 2: shallow copy -- top-level keys are new, nested objects still shared
state_shallow = state_original.copy()
state_shallow["brief"] = "NEW brief"               # top-level: independent
state_shallow["findings"].append("c")              # nested list: still shared!
print("\nScenario 2 (shallow copy):")
print("  original brief:", state_original["brief"])    # 'EV market' -- good
print("  original findings:", state_original["findings"]) # ['a', 'b', 'c'] -- LEAKED

# Scenario 3: deep copy -- fully independent
state_deep = copy.deepcopy(state_original)
state_deep["findings"].append("d")
print("\nScenario 3 (deep copy):")
print("  original findings:", state_original["findings"])   # ['a', 'b', 'c'] -- untouched
print("  deep copy findings:", state_deep["findings"])      # ['a', 'b', 'c', 'd']

# The LangGraph-correct pattern: return a NEW dict, never mutate the input
def researcher_node(state):
    new_findings = state["findings"] + ["new insight"]    # + creates a new list
    return {"findings": new_findings}                     # return partial update only

merged = {**state_original, **researcher_node(state_original)}
print("\nCorrect node pattern:", merged["findings"])
print("Original untouched:", state_original["findings"])
```

**Why each scenario matters:** Scenario 1 is "accidental shared state" — the most common beginner bug in any multi-step pipeline. Scenario 2 is "I thought I copied it" — common from developers who learned shallow copy as "the copy method." Scenario 3 is correct but expensive for large state; the `{**state, **update}` pattern at the bottom is what LangGraph actually recommends — return a *partial* update and let the framework merge it, never mutate the input, never deep-copy the entire state on every node call.

#### 1.3 Homework/Challenge — First-class functions as a node registry (1h)

Build a `NodeRegistry` class that stores node functions (plain functions) by name, and can retrieve and execute them:

```python
class NodeRegistry:
    def __init__(self):
        self._nodes = {}          # name -> function (function is an object!)

    def register(self, name, fn):
        if not callable(fn):
            raise TypeError(f"{fn!r} is not callable")
        self._nodes[name] = fn

    def run(self, name, state):
        if name not in self._nodes:
            raise KeyError(f"No node named {name!r}")
        return self._nodes[name](state)

    def list_nodes(self):
        return list(self._nodes.keys())

def researcher(state): return {"raw": f"Researched {state['brief']}"}
def formatter(state): return {"report": f"Formatted: {state['raw']}"}

registry = NodeRegistry()
registry.register("researcher", researcher)
registry.register("formatter", formatter)

state = {"brief": "EV market", "raw": ""}
state = {**state, **registry.run("researcher", state)}
state = {**state, **registry.run("formatter", state)}
print(state)
print("registered nodes:", registry.list_nodes())
```

**Challenge:** add a `run_all(state)` method that runs all registered nodes in insertion order, threading state through each — and add type-checking so `register()` also rejects a function whose first argument is not named `state` (use `inspect.signature(fn).parameters` to check this).

---

### Module 2 — `__init__`, `self`, Instance vs. Class Attributes (Hours 4–6)
**The constructor is not magic — and Python makes you see exactly what it does**

#### 2.1 Concept Lecture — The four things `__init__` actually does (1h)

**Why this matters for LangGraph:** `BaseTool.__init__(self, name, description)` stores two strings as instance attributes. Every custom tool you will ever write calls `super().__init__("my_name", "my_description")`. If you don't deeply understand what that call does and why, you won't know why leaving it out silently breaks things.

**The four things `__init__` does (not three, not five):**

```python
class BaseTool:
    #                     ┌── (2) Python passes the new object here automatically
    def __init__(self,   name,   description):
    #   ▲                  ▲          ▲
    #   │              (3) positional  (3) positional or keyword arg
    # (1) special name:     args passed BY THE CALLER
    #     Python calls this
    #     automatically after
    #     creating the object

        self.name = name           # (4) attach data TO this specific object
        self.description = description
```

1. **`__init__` is called automatically** — you don't call it yourself: `t = BaseTool("x", "y")` triggers it.
2. **`self` is the new object** — Python created it just before calling `__init__` and passes it in. You name this parameter `self` by convention, but Python doesn't care what you call it (don't call it anything else).
3. **The remaining parameters** are what the caller passes.
4. **`self.name = name`** literally writes a new attribute onto the object — Python objects are dict-like containers, and this is adding a key.

**The proof that `self` is just the object itself:**
```python
class Demo:
    def __init__(self, value):
        self.value = value
    def show(self):
        return self.value

d = Demo(42)
print(d.show())          # 42 -- normal call
print(Demo.show(d))      # 42 -- calling the UNBOUND method, passing d explicitly
```
Both lines are identical — `d.show()` is just syntactic sugar for `Demo.show(d)`. Python always passes the instance as the first argument, which is why every method must accept `self`.

**Instance attributes vs. class attributes — ASCII diagram:**
```
                       ┌────────────────────────────────────────┐
       CLASS level     │  BaseTool                              │
                       │  ┌──────────────────────────────────┐  │
                       │  │ registry = {}    (class attr)    │  │
                       │  │ MAX_RETRIES = 3  (class attr)    │  │
                       │  └──────────────────────────────────┘  │
                       └────────────┬───────────────────────────┘
                                    │ (shared by all instances)
              ┌─────────────────────┼──────────────────────────┐
              ▼                     ▼                           ▼
  tool_a (instance)       tool_b (instance)          tool_c (instance)
  name = "search"         name = "calculator"        name = "lookup"
  description = "..."     description = "..."        description = "..."
  (each has its own)      (each has its own)         (each has its own)
```

```python
class BaseTool:
    registry = {}           # CLASS attribute -- ONE dict shared by ALL instances

    def __init__(self, name, description):
        self.name = name                  # INSTANCE attribute -- unique per object
        self.description = description
        BaseTool.registry[name] = self   # register every new tool globally

search = BaseTool("search", "searches the web")
calc   = BaseTool("calculator", "adds numbers")
print(BaseTool.registry.keys())    # dict_keys(['search', 'calculator'])
print(search.registry is calc.registry)   # True -- same dict!
```

**The mutable-class-attribute pitfall (the most important warning in this module):**
```python
class BadTool:
    history = []            # MUTABLE class attribute -- SHARED by every instance

class GoodTool:
    def __init__(self):
        self.history = []   # INSTANCE attribute -- each object gets its OWN list

bad1 = BadTool()
bad2 = BadTool()
bad1.history.append("a")
print(bad2.history)          # ['a'] -- LEAKED to bad2!

good1 = GoodTool()
good2 = GoodTool()
good1.history.append("a")
print(good2.history)         # [] -- correctly isolated
```

**Exercise:** write a `ToolRegistry` class that uses a class-level dict to track every instance ever created, plus a class-level counter for total instances, and an instance-level `call_count` tracking how many times each individual tool has been called.

#### 2.2 Guided Coding Drill — Build `BaseTool` from first principles (1h)

```python
class BaseTool:
    """The foundational tool class -- mirrors what LangChain's BaseTool actually does."""
    
    _registry = {}      # class-level: every tool registered here at creation time
    _count = 0          # class-level: total tools ever created

    def __init__(self, name: str, description: str):
        if not name.strip():
            raise ValueError("Tool name cannot be blank")
        self.name = name
        self.description = description
        self._call_count = 0            # instance-level: calls to THIS tool specifically
        BaseTool._registry[name] = self
        BaseTool._count += 1

    def run(self, *args, **kwargs):
        """Subclasses must override this."""
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement run()"
        )

    def _tracked_run(self, *args, **kwargs):
        """Internal: run + count the call."""
        self._call_count += 1
        return self.run(*args, **kwargs)

    @classmethod
    def get_all(cls):
        return dict(cls._registry)     # return a copy so callers can't mutate the registry

    @classmethod
    def total_created(cls):
        return cls._count

class CalculatorTool(BaseTool):
    def __init__(self):
        super().__init__("calculator", "Adds two numbers")

    def run(self, a, b):
        return a + b

class SearchTool(BaseTool):
    def __init__(self):
        super().__init__("search", "Searches the web")

    def run(self, query):
        return f"3 results for '{query}'"

calc = CalculatorTool()
search = SearchTool()

calc._tracked_run(2, 3)
calc._tracked_run(10, 20)

print("calc calls:", calc._call_count)    # 2
print("search calls:", search._call_count)  # 0
print("total tools:", BaseTool.total_created())     # 2
print("all tools:", list(BaseTool.get_all().keys()))
```

**Drill:** extend `BaseTool` with an instance-level `last_error` attribute (starts as `None`), set it inside `_tracked_run` if `run()` raises — so the caller can always check `tool.last_error` after a call to see if anything went wrong, without having to catch the exception themselves.

#### 2.3 Homework/Challenge — A self-documenting tool (1h)

Extend the `BaseTool` class so each tool instance automatically builds its own docstring-based usage guide:

1. Add an `examples` class attribute (list of dicts) that subclasses can define, e.g. `[{"input": {"a": 2, "b": 3}, "output": 5}]`
2. Add an instance method `usage_guide()` that formats a string combining `self.description`, `self.__class__.__name__`, and all `examples`
3. Add a `__repr__` that returns `ClassName(name='x', calls=N)` — use only `self` attributes, not any globals

**Challenge:** make `BaseTool.__init__` raise a `TypeError` (with a clear message naming the missing attribute) if a subclass defines an `examples` attribute but it's not a `list` — enforcing the contract at construction time, the same way Pydantic would at a field level.

---
