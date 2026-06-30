# Python for LangChain & LangGraph: Zero to Proficient

### A 30-Hour No-Fluff Course for Developers New to Python

> **How to use this document:** Open it on Day 1. Read top to bottom. Every section builds on the last. Code every example. Do every drill. Skip nothing.

---

# TABLE OF CONTENTS

- [Part 1: Detailed Syllabus](#part-1-detailed-syllabus)
  - [Module 1: Python's Object Model & Syntax Fundamentals](#module-1-pythons-object-model--syntax-fundamentals)
  - [Module 2: Strings, Numbers & Basic I/O](#module-2-strings-numbers--basic-io)
  - [Module 3: Lists & Tuples — Sequential Data](#module-3-lists--tuples--sequential-data)
  - [Module 4: Dictionaries — The Heart of LangGraph State](#module-4-dictionaries--the-heart-of-langgraph-state)
  - [Module 5: Sets & Control Flow](#module-5-sets--control-flow)
  - [Module 6: Functions — The Atom of Agent Behavior](#module-6-functions--the-atom-of-agent-behavior)
  - [Module 7: OOP Fundamentals — Building Your Own Tools](#module-7-oop-fundamentals--building-your-own-tools)
  - [Module 8: Modules, Imports & Project Structure](#module-8-modules-imports--project-structure)
  - [Module 9: Error Handling & Context Managers](#module-9-error-handling--context-managers)
  - [Module 10: Capstone — Manual LangChain in Pure Python](#module-10-capstone--manual-langchain-in-pure-python)
- [Part 2: Companion Study Guide](#part-2-companion-study-guide)
  - [Environment Setup](#environment-setup)
  - [Quick Reference Cards](#quick-reference-cards)
  - [LangChain Translation Dictionary](#langchain-translation-dictionary)
  - [Common Pitfalls](#common-pitfalls)
  - [Practice Problem Bank (50 Drills)](#practice-problem-bank)
  - [Bridge Topics Appendix](#bridge-topics-appendix)
- [Part 3: Prerequisite Roadmap](#part-3-prerequisite-roadmap)

---

# PART 1: DETAILED SYLLABUS

---

## MODULE 1: Python's Object Model & Syntax Fundamentals

**Duration: 3 hours** | **Theme: The ground rules before we write a single agent**

---

### Lesson 1.1 — Concept Lecture (1 hour)

**Topic:** Variables, naming, indentation, and Python's "everything is an object" philosophy.

**Why this matters for LangGraph:**
Every piece of state in a LangGraph graph—messages, tool results, memory, routing decisions—lives in a Python object. Before you can manipulate state, you must understand _what a Python object is_ and _how Python names refer to objects_ (not contain them). This is the single most important mental model shift coming from other languages.

---

#### 1.1.1 — Python is Not a Script. It's a Live Object System.

In Python, a "variable" is a _label_ (reference) attached to an object living in memory. The object knows its own type. This differs from statically-typed languages where the variable _is_ the container.

```python
# The integer 42 is an object. x is a label pointing to it.
x = 42
print(type(x))       # <class 'int'>
print(id(x))         # memory address of the object

# y is now a SECOND label pointing to the SAME object
y = x
print(id(x) == id(y))  # True — same object!

# Reassigning y makes it point to a NEW object
y = 99
print(id(x) == id(y))  # False — different objects now
print(x)                # still 42 — x is unaffected
```

**ASCII Memory Diagram:**

```
After x = 42, y = x:

  x ──────┐
           ▼
         [42] (object at 0x10a3b1c)
           ▲
  y ──────┘

After y = 99:

  x ──────► [42] (object at 0x10a3b1c)
  y ──────► [99] (object at 0x10a4d2f)
```

**LangGraph Analogy:**
LangGraph's `State` is a dict of labels pointing to objects. When a node returns `{"messages": new_list}`, it's replacing the label `messages` to point at a new list object. If you mutate the old list in-place, the old reference still points to the modified data. This causes subtle bugs in reducers. You will see this exact issue in Module 4.

---

#### 1.1.2 — Naming Conventions (PEP 8, the law of Python land)

```python
# Variables and functions: snake_case
agent_state = {}
user_message = "Hello"
max_iterations = 10

# Constants: UPPER_SNAKE_CASE
DEFAULT_MODEL = "gpt-4o"
MAX_TOKENS = 4096

# Classes: PascalCase
class AgentNode:
    pass

# "Private" (by convention, not enforced): leading underscore
_internal_counter = 0

# Module-level dunder (double underscore both sides): special Python names
__version__ = "1.0.0"
```

> **Rule:** Python has no `private` keyword. Leading underscores are a _social contract_, not enforced access control. LangChain's source code uses `_run()` for internal tool execution and `run()` for the public API. You'll see this constantly.

---

#### 1.1.3 — Indentation: Python's Block Syntax

Python uses indentation (4 spaces, always) instead of `{}` to define blocks.

```python
# CORRECT
if True:
    print("inside block")    # 4 spaces
    print("still inside")    # same level = same block

# THIS IS A SYNTAX ERROR
if True:
    print("inside")
  print("wrong indent")      # IndentationError!
```

**The rule is simple:** Every block (if, for, while, def, class) must be indented consistently. Everything at the same indent level belongs to the same block.

---

#### 1.1.4 — Basic I/O

```python
# Output
print("Agent started.")
print("Step:", 3)            # comma auto-adds a space
print(f"Model: {DEFAULT_MODEL}")   # f-string (interpolation)

# Input (blocks until user presses Enter)
user_input = input("Enter your query: ")
print(f"You said: {user_input}")

# Multiple values, custom separator
print("A", "B", "C", sep=" | ")   # A | B | C
print("Done!", end="")              # no newline at end
```

---

#### 1.1.5 — Python's Type System at a Glance

```python
# Check type
x = 3.14
print(type(x))           # <class 'float'>
print(isinstance(x, float))  # True
print(isinstance(x, (int, float)))  # True — checks against tuple of types

# Python is DYNAMICALLY typed — type is checked at runtime
x = "now I'm a string"   # totally legal
print(type(x))           # <class 'str'>
```

**LangGraph Analogy:**
LangGraph nodes can return any dict shape — Python won't stop you from returning `{"messages": "oops a string"}` when it expected a list. This is why LangGraph uses Pydantic models for state validation (a Bridge Topic). For now, know that Python's dynamic typing gives you flexibility AND responsibility.

---

### Lesson 1.2 — Guided Coding Drill (1 hour)

**Goal:** Build a simple "agent context" dictionary by hand and prove the object reference model to yourself.

```python
# drill_01_object_model.py
# Run this file: python drill_01_object_model.py

# --- DRILL 1: Labels and Objects ---
print("=== DRILL 1: Labels & Objects ===")

agent_name = "Planner"
session_id = "abc-123"
is_active = True
turn_count = 0

print(f"Agent: {agent_name}")
print(f"Session: {session_id}")
print(f"Active: {is_active}")
print(f"Turns: {turn_count}")
print(f"Name type: {type(agent_name)}")

# --- DRILL 2: Reference aliasing ---
print("\n=== DRILL 2: Reference Aliasing ===")

original_count = 5
backup = original_count   # both point to the same int object
original_count = 10       # original_count now points to new object
print(f"original_count: {original_count}")  # 10
print(f"backup: {backup}")                  # 5 — unchanged!

# --- DRILL 3: Simulating an agent turn counter ---
print("\n=== DRILL 3: Agent Turn Counter ===")

turn_count = 0
MAX_TURNS = 5

print(f"Starting agent. Max turns: {MAX_TURNS}")
turn_count = turn_count + 1
print(f"After turn 1: {turn_count}")
turn_count += 1           # shorthand for turn_count = turn_count + 1
print(f"After turn 2: {turn_count}")

# Prove that += creates a new int object (ints are immutable)
print(f"id before: {id(turn_count)}")
turn_count += 1
print(f"id after:  {id(turn_count)}")   # different id — new object

# --- DRILL 4: Type checking (like LangGraph does at node boundaries) ---
print("\n=== DRILL 4: Type Checking ===")

def check_state_value(value):
    if isinstance(value, str):
        print(f"  String: '{value}'")
    elif isinstance(value, int):
        print(f"  Integer: {value}")
    elif isinstance(value, bool):
        print(f"  Boolean: {value}")
    else:
        print(f"  Unknown type: {type(value)}")

check_state_value("hello from user")
check_state_value(42)
check_state_value(True)
check_state_value(3.14)
```

**Expected output:**

```
=== DRILL 1: Labels & Objects ===
Agent: Planner
Session: abc-123
Active: True
Turns: 0
Name type: <class 'str'>

=== DRILL 2: Reference Aliasing ===
original_count: 10
backup: 5

=== DRILL 3: Agent Turn Counter ===
Starting agent. Max turns: 5
After turn 1: 1
After turn 2: 2
id before: 140234561234xx
id after:  140234561235xx

=== DRILL 4: Type Checking ===
  String: 'hello from user'
  Integer: 42
  Boolean: True
  Unknown type: <class 'float'>
```

> **Note on booleans:** `bool` is actually a subclass of `int` in Python. `isinstance(True, int)` returns `True`. Check for `bool` before `int` if you need to distinguish them.

---

### Lesson 1.3 — Homework Challenge (1 hour)

**Challenge: Agent Registry**

Build a simple "agent registry" using only variables and `print()`. No data structures yet.

**Requirements:**

1. Define 3 "agents" by creating name, role, and turn_limit variables for each (e.g., `agent1_name = "Planner"`, `agent1_role = "planning"`, `agent1_turn_limit = 5`).
2. Print a formatted "registration card" for each agent using f-strings.
3. Simulate 3 turns for agent 1 using `+=`.
4. Print whether agent 1 has exceeded its turn limit using a comparison (`>`).
5. Use `isinstance()` to verify all your turn_limit variables are integers.

**Bonus:** What happens if you accidentally set `agent1_turn_limit = "5"` (a string) instead of `5`? Add code to detect this and print a warning message.

**Solution (attempt it first!):**

```python
# homework_01_solution.py

# Agent definitions
agent1_name = "Planner"
agent1_role = "planning"
agent1_turn_limit = 5

agent2_name = "Researcher"
agent2_role = "research"
agent2_turn_limit = 10

agent3_name = "Executor"
agent3_role = "execution"
agent3_turn_limit = 3

# Registration cards
print("=== AGENT REGISTRY ===")
print(f"[1] {agent1_name} | Role: {agent1_role} | Max Turns: {agent1_turn_limit}")
print(f"[2] {agent2_name} | Role: {agent2_role} | Max Turns: {agent2_turn_limit}")
print(f"[3] {agent3_name} | Role: {agent3_role} | Max Turns: {agent3_turn_limit}")

# Simulate turns for agent 1
turn_count = 0
turn_count += 1
turn_count += 1
turn_count += 1
print(f"\nAgent 1 ran {turn_count} turns.")
print(f"Exceeded limit? {turn_count > agent1_turn_limit}")

# Type validation
print(f"\nagent1_turn_limit is int: {isinstance(agent1_turn_limit, int)}")

# Bonus: bad type detection
bad_limit = "5"   # someone made a mistake
if not isinstance(bad_limit, int):
    print(f"WARNING: turn limit should be int, got {type(bad_limit).__name__}")
```

---

## MODULE 2: Strings, Numbers & Basic I/O

**Duration: 3 hours** | **Theme: The raw material flowing through every LangChain pipeline**

---

### Lesson 2.1 — Concept Lecture (1 hour)

**Why this matters for LangGraph:**
Every LLM interaction is, at its core, a string in and a string out. Prompt templates are string manipulation. Tool results are parsed strings. Message histories are lists of strings and dicts. Master Python's string model and you master 70% of what LangChain's `PromptTemplate` does.

---

#### 2.1.1 — Strings: Immutable Sequences of Unicode Characters

```python
# String creation
s1 = 'single quotes'
s2 = "double quotes"        # identical in Python
s3 = """triple-quoted
string spans
multiple lines"""           # preserves newlines

# Strings are immutable — you cannot change a character in place
s = "hello"
# s[0] = "H"    # TypeError: 'str' object does not support item assignment
s = "Hello"     # you create a NEW string object

# String length
print(len("LangChain"))     # 9

# Indexing (0-based, negative from end)
s = "LangGraph"
print(s[0])     # L
print(s[-1])    # h
print(s[4])     # G

# Slicing [start:stop:step] — returns a new string
print(s[0:4])   # Lang
print(s[4:])    # Graph
print(s[:4])    # Lang
print(s[::2])   # LnGah  (every other character)
print(s[::-1])  # hparGgnaL (reversed)
```

---

#### 2.1.2 — F-Strings: Python's Prompt Templating Primitive

```python
# F-strings (Python 3.6+, use these always)
model_name = "gpt-4o"
temperature = 0.7
max_tokens = 1024

# Basic interpolation
prompt = f"Using model {model_name} with temperature {temperature}."
print(prompt)

# Expressions inside {}
print(f"Half of max_tokens: {max_tokens // 2}")

# Format specifiers
pi = 3.14159265
print(f"Pi to 2 decimal places: {pi:.2f}")
print(f"Score as percentage: {0.8723:.1%}")  # 87.2%

# Multi-line f-string (building a system prompt)
agent_role = "research assistant"
user_name = "Alice"
system_prompt = f"""You are a {agent_role}.
The user's name is {user_name}.
Always be concise and factual."""

print(system_prompt)
```

**LangGraph Analogy:**
`PromptTemplate` from LangChain is a Python class that does exactly this — it takes a template string with `{variable}` placeholders and formats them with `.format_messages()`. Once you build your own in the capstone, you'll realize it's less magic than it looks.

---

#### 2.1.3 — Essential String Methods

````python
text = "  Hello, World! I am an AI agent.  "

# Cleaning
print(text.strip())             # remove leading/trailing whitespace
print(text.lstrip())            # remove leading only
print(text.rstrip())            # remove trailing only

# Case
print(text.lower())
print(text.upper())
print(text.title())             # Title Case

# Searching
print(text.find("AI"))          # index of first occurrence, -1 if not found
print("AI" in text)             # True/False membership test (use this!)
print(text.startswith("  He"))  # True
print(text.endswith("  "))      # True

# Splitting and joining (critical for parsing LLM outputs)
words = text.strip().split(" ")
print(words)
print(len(words))

csv_line = "gpt-4o,0.7,1024,true"
parts = csv_line.split(",")
print(parts)    # ['gpt-4o', '0.7', '1024', 'true']

# Join — inverse of split
path_parts = ["agents", "planner", "tools"]
full_path = "/".join(path_parts)
print(full_path)  # agents/planner/tools

# Replace
response = "The answer is <ANSWER>42</ANSWER>"
cleaned = response.replace("<ANSWER>", "").replace("</ANSWER>", "")
print(cleaned)  # The answer is 42

# Count
text2 = "token token token word"
print(text2.count("token"))  # 3

# Strip specific characters
tag = "```python\ncode here\n```"
print(tag.strip("`"))  # python\ncode here\n
````

---

#### 2.1.4 — Numbers: int, float, and Type Conversion

```python
# Integer operations
a, b = 17, 5
print(a + b)    # 22
print(a - b)    # 12
print(a * b)    # 85
print(a / b)    # 3.4    — always returns float
print(a // b)   # 3      — floor division (integer result)
print(a % b)    # 2      — modulo (remainder)
print(a ** b)   # 1419857 — exponentiation

# Float
temp = 0.1 + 0.2
print(temp)              # 0.30000000000000004 — floating point!
print(round(temp, 2))    # 0.3

# Type conversion (common when parsing LLM outputs)
print(int("42"))         # 42
print(float("3.14"))     # 3.14
print(str(100))          # "100"
print(bool(0))           # False
print(bool(""))          # False — empty string is falsy
print(bool("hello"))     # True

# Truthiness — CRITICAL for LangGraph conditionals
print(bool(None))   # False
print(bool([]))     # False — empty list
print(bool({}))     # False — empty dict
print(bool([0]))    # True  — non-empty list (even if contents are falsy!)
```

**The Falsy Values in Python (memorize this):**

```
False, None, 0, 0.0, "", [], {}, set()
Everything else is truthy.
```

---

#### 2.1.5 — String Parsing Patterns (for LLM Output Parsing)

```python
# Pattern 1: Extract content between tags
def extract_between_tags(text, tag):
    start_tag = f"<{tag}>"
    end_tag = f"</{tag}>"
    start = text.find(start_tag) + len(start_tag)
    end = text.find(end_tag)
    if start == -1 or end == -1:
        return None
    return text[start:end].strip()

llm_response = "<thought>I need to search the web.</thought><action>search</action>"
print(extract_between_tags(llm_response, "thought"))  # I need to search the web.
print(extract_between_tags(llm_response, "action"))   # search

# Pattern 2: Check if response contains a stop signal
STOP_SIGNALS = ["DONE", "FINAL ANSWER", "I cannot help with that"]
def should_stop(response):
    for signal in STOP_SIGNALS:
        if signal.lower() in response.lower():
            return True
    return False

print(should_stop("FINAL ANSWER: Paris is the capital of France."))  # True
print(should_stop("Let me search for more information."))             # False
```

---

### Lesson 2.2 — Guided Coding Drill (1 hour)

```python
# drill_02_strings.py

# --- DRILL 1: Prompt Template Builder ---
print("=== DRILL 1: Manual Prompt Template ===")

def build_prompt(template, **variables):
    """Manually replace {key} placeholders in a template string."""
    result = template
    for key, value in variables.items():
        result = result.replace("{" + key + "}", str(value))
    return result

template = "Answer the following question as a {role}.\n\nQuestion: {question}\n\nBe {style}."
filled = build_prompt(template,
                      role="data scientist",
                      question="What is overfitting?",
                      style="concise")
print(filled)
print()

# --- DRILL 2: Token Counting Approximation ---
print("=== DRILL 2: Token Counter ===")
def approx_token_count(text):
    """Rough token approximation: ~4 chars per token."""
    words = text.split()
    char_count = len(text)
    word_estimate = len(words)
    char_estimate = char_count // 4
    return {"words": word_estimate, "chars": char_count, "approx_tokens": char_estimate}

sample = "LangChain is a framework for building LLM applications."
counts = approx_token_count(sample)
print(f"Words: {counts['words']}, Chars: {counts['chars']}, Approx Tokens: {counts['approx_tokens']}")
print()

# --- DRILL 3: LLM Response Parser ---
print("=== DRILL 3: Response Parser ===")
def parse_structured_response(response):
    """Parse a structured LLM output with labeled sections."""
    result = {}
    lines = response.strip().split("\n")
    for line in lines:
        if ":" in line:
            key, _, value = line.partition(":")
            result[key.strip().lower().replace(" ", "_")] = value.strip()
    return result

llm_output = """
Thought: I should search for recent AI news.
Action: web_search
Action Input: recent AI developments 2024
"""
parsed = parse_structured_response(llm_output)
print(parsed)
# {'thought': 'I should search for recent AI news.',
#  'action': 'web_search',
#  'action_input': 'recent AI developments 2024'}
```

---

### Lesson 2.3 — Homework Challenge (1 hour)

**Challenge: Build a ReAct-Style Output Parser**

The ReAct (Reasoning + Acting) pattern is the backbone of most LangChain agents. The LLM produces output in this format:

```
Thought: [reasoning]
Action: [tool_name]
Action Input: [input to the tool]
```

**Your task:**

1. Write a function `parse_react_output(text)` that extracts `thought`, `action`, and `action_input` from a ReAct-formatted string.
2. Handle cases where a field is missing (return `None` for that field).
3. Write a function `format_react_input(observation)` that takes a tool's result string and formats it as `"Observation: {observation}\n"`.
4. Write a function `is_final_answer(text)` that returns `True` if the text contains `"Final Answer:"`.

**Test your code with these inputs:**

```python
test1 = "Thought: I need to check the weather.\nAction: get_weather\nAction Input: New York"
test2 = "Final Answer: The weather in New York is 72°F and sunny."
test3 = "Thought: I have enough information."  # missing Action
```

---

## MODULE 3: Lists & Tuples — Sequential Data

**Duration: 3 hours** | **Theme: Ordered collections — your message history lives here**

---

### Lesson 3.1 — Concept Lecture (1 hour)

**Why this matters for LangGraph:**
LangGraph's message history (`state["messages"]`) is a Python list. Every new message gets appended. Reducers decide how to merge old and new lists. Knowing list operations cold is non-negotiable.

---

#### 3.1.1 — Lists: Mutable, Ordered, Heterogeneous

```python
# Creation
messages = []                    # empty list
scores = [0.9, 0.75, 0.88]
mixed = [42, "hello", True, None, [1, 2]]   # heterogeneous!

# Length
print(len(messages))    # 0
print(len(scores))      # 3

# Indexing and slicing (same as strings)
items = ["a", "b", "c", "d", "e"]
print(items[0])         # a
print(items[-1])        # e
print(items[1:3])       # ['b', 'c']
print(items[::2])       # ['a', 'c', 'e']

# Membership
print("b" in items)     # True
print("z" in items)     # False
```

---

#### 3.1.2 — List Mutation: The Critical Operations

```python
messages = []

# APPEND — add to end (most common)
messages.append({"role": "user", "content": "Hello"})
messages.append({"role": "assistant", "content": "Hi there!"})
print(messages)

# EXTEND — add all items from another list
new_messages = [
    {"role": "user", "content": "What's 2+2?"},
    {"role": "assistant", "content": "4"}
]
messages.extend(new_messages)
print(f"Total messages: {len(messages)}")

# INSERT — add at specific index
messages.insert(0, {"role": "system", "content": "You are a helpful assistant."})
print(messages[0])  # system message is now first

# REMOVE operations
messages.pop()           # removes and returns last item
messages.pop(0)          # removes and returns item at index 0
# messages.remove(item)  # removes first occurrence by value (rarely used)

# SORT
numbers = [3, 1, 4, 1, 5, 9, 2, 6]
numbers.sort()                         # in-place sort (modifies original)
print(numbers)                         # [1, 1, 2, 3, 4, 5, 6, 9]

sorted_copy = sorted([3, 1, 4])       # returns NEW sorted list
print(sorted_copy)

# Sort with key function — sort messages by content length
msgs = [
    {"role": "user", "content": "Hi"},
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "user", "content": "Thanks!"}
]
msgs.sort(key=lambda m: len(m["content"]))
for m in msgs:
    print(f"{len(m['content'])}: {m['content']}")

# REVERSE
numbers.reverse()    # in-place
print(numbers[::-1]) # slicing — returns new reversed list
```

---

#### 3.1.3 — List Comprehensions: Python's Power Move

List comprehensions build new lists from existing iterables in one line. They replace most `for` loops that build lists.

```python
# Pattern: [expression for item in iterable if condition]

# Basic: square numbers 0-9
squares = [x ** 2 for x in range(10)]
print(squares)  # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

# With filter: only even squares
even_squares = [x ** 2 for x in range(10) if x % 2 == 0]
print(even_squares)  # [0, 4, 16, 36, 64]

# Extract all user messages from a history
history = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi!"},
    {"role": "user", "content": "What's 2+2?"},
    {"role": "assistant", "content": "4"},
]
user_messages = [m["content"] for m in history if m["role"] == "user"]
print(user_messages)  # ['Hello', "What's 2+2?"]

# Transform: uppercase all tool names
tool_names = ["web_search", "calculator", "code_interpreter"]
display_names = [name.replace("_", " ").title() for name in tool_names]
print(display_names)  # ['Web Search', 'Calculator', 'Code Interpreter']

# Flatten nested list (common when aggregating tool results)
nested = [[1, 2], [3, 4], [5, 6]]
flat = [item for sublist in nested for item in sublist]
print(flat)  # [1, 2, 3, 4, 5, 6]
```

**ASCII Memory Diagram: List vs Comprehension**

```
# Traditional loop approach:
result = []
for x in range(5):
    if x % 2 == 0:
        result.append(x * 2)

# Comprehension (same result, one line):
result = [x * 2 for x in range(5) if x % 2 == 0]

Both produce: [0, 4, 8]
The comprehension is NOT just shorter — it's often faster.
```

---

#### 3.1.4 — Tuples: Immutable Sequences (Structural Integrity)

```python
# Tuples are like lists but IMMUTABLE (cannot be changed after creation)
point = (3, 7)
rgb = (255, 128, 0)
single = (42,)          # trailing comma required for single-element tuple
empty = ()

# Indexing works the same
print(point[0])   # 3
print(rgb[-1])    # 0

# Tuples CANNOT be mutated
# point[0] = 5    # TypeError: 'tuple' object does not support item assignment

# Tuple unpacking — Python's most elegant feature
x, y = point
print(f"x={x}, y={y}")

# Swap without temp variable (using tuple packing/unpacking)
a, b = 10, 20
a, b = b, a
print(f"a={a}, b={b}")  # a=20, b=10

# Unpack from function return
def get_model_config():
    return "gpt-4o", 0.7, 1024  # returns a tuple

model, temp, tokens = get_model_config()
print(f"Model: {model}, Temp: {temp}, Tokens: {tokens}")

# Extended unpacking with *
first, *middle, last = [1, 2, 3, 4, 5]
print(first)   # 1
print(middle)  # [2, 3, 4]
print(last)    # 5

# Use tuples for: fixed configs, coordinates, function returns, dict keys
config = ("gpt-4o", 0.7)
lookup = {config: "cached_response"}  # tuples can be dict keys, lists cannot!
```

**When to use tuple vs list:**

```
tuple: fixed structure, multiple return values, dict keys, config constants
list:  growing/shrinking sequences, message history, tool results
```

---

#### 3.1.5 — Essential List Built-ins

```python
scores = [85, 92, 78, 95, 88, 73, 91]

print(len(scores))      # 7
print(sum(scores))      # 602
print(min(scores))      # 73
print(max(scores))      # 95

# enumerate — get index AND value (use this, not range(len(...)))
history = ["msg1", "msg2", "msg3"]
for i, msg in enumerate(history):
    print(f"Turn {i}: {msg}")

# zip — combine two lists element-by-element
roles = ["user", "assistant", "user"]
contents = ["Hello", "Hi!", "How are you?"]
for role, content in zip(roles, contents):
    print(f"{role}: {content}")

# Build list of dicts from parallel lists
messages = [{"role": r, "content": c} for r, c in zip(roles, contents)]
print(messages)

# any() / all() — aggregate boolean checks
has_system_msg = any(m["role"] == "system" for m in messages)
all_have_content = all(m["content"] for m in messages)
print(f"Has system: {has_system_msg}, All have content: {all_have_content}")
```

---

### Lesson 3.2 — Guided Coding Drill (1 hour)

```python
# drill_03_lists.py

print("=== DRILL 1: Message History Manager ===")

class MessageHistory:
    """Manually simulates LangGraph's messages state field."""

    def __init__(self, max_messages=10):
        self._messages = []
        self._max = max_messages

    def add(self, role, content):
        self._messages.append({"role": role, "content": content})
        # Trim to max (sliding window — real LangChain pattern!)
        if len(self._messages) > self._max:
            # Keep system message if present, remove oldest non-system
            system_msgs = [m for m in self._messages if m["role"] == "system"]
            other_msgs = [m for m in self._messages if m["role"] != "system"]
            other_msgs = other_msgs[-(self._max - len(system_msgs)):]
            self._messages = system_msgs + other_msgs

    def get_all(self):
        return list(self._messages)  # return a COPY

    def get_by_role(self, role):
        return [m for m in self._messages if m["role"] == role]

    def last_n(self, n):
        return self._messages[-n:]

    def clear(self):
        self._messages = []

    def __len__(self):
        return len(self._messages)

    def __repr__(self):
        return f"MessageHistory({len(self._messages)} messages)"

# Test it
history = MessageHistory(max_messages=5)
history.add("system", "You are a helpful assistant.")
history.add("user", "Hello!")
history.add("assistant", "Hi there!")
history.add("user", "What's 2+2?")
history.add("assistant", "4")
history.add("user", "Thanks!")  # this should trigger trimming

print(f"Total messages: {len(history)}")
print(f"User messages: {history.get_by_role('user')}")
print(f"Last 2: {history.last_n(2)}")
print()

print("=== DRILL 2: Tool Result Aggregator ===")
tool_results = [
    {"tool": "web_search", "result": "Python is a high-level language.", "success": True},
    {"tool": "calculator", "result": None, "success": False},
    {"tool": "file_read", "result": "Contents of config.json...", "success": True},
]

successful = [r for r in tool_results if r["success"]]
failed = [r for r in tool_results if not r["success"]]
results_text = [r["result"] for r in successful]

print(f"Successful: {len(successful)}, Failed: {len(failed)}")
print(f"Combined results:\n" + "\n---\n".join(results_text))
```

---

### Lesson 3.3 — Homework Challenge (1 hour)

**Challenge: Sliding Window Memory**

Implement a `SlidingWindowMemory` class that:

1. Stores messages as a list of dicts `{"role": str, "content": str}`.
2. Has a `add(role, content)` method.
3. Has a `get_context(window_size)` method that returns only the last `window_size` messages, but _always_ includes the first message if its role is `"system"`.
4. Has a `to_prompt_string()` method that formats the full history as:
   ```
   system: You are a helpful assistant.
   user: Hello!
   assistant: Hi!
   ```
5. Has a `token_estimate()` method that returns `sum(len(m["content"]) // 4 for m in messages)`.

Write 5 test cases demonstrating edge cases (empty history, only system message, window larger than history, etc.).

---

## MODULE 4: Dictionaries — The Heart of LangGraph State

**Duration: 3 hours** | **Theme: Every LangGraph node reads and writes a dict. This is non-negotiable.**

---

### Lesson 4.1 — Concept Lecture (1 hour)

**Why this matters for LangGraph:**
LangGraph's `State` is a `TypedDict` (a dict with type hints). Every node receives the full state dict and returns a partial dict with only the fields it modified. Reducers merge these partial dicts. If you don't understand Python dicts deeply — creation, access, merging, nesting, copying — you will be confused by LangGraph from line one.

---

#### 4.1.1 — Dictionary Fundamentals

```python
# Creation
empty = {}
agent_state = {"messages": [], "step": 0, "is_done": False}
config = dict(model="gpt-4o", temperature=0.7)   # using dict() constructor

# Access
print(agent_state["messages"])      # []
print(agent_state["step"])          # 0

# KeyError — the #1 dict bug in LangGraph code
# print(agent_state["nonexistent"])  # KeyError: 'nonexistent'

# SAFE ACCESS with .get() — use this for state fields that might not exist
print(agent_state.get("nonexistent"))           # None (no error!)
print(agent_state.get("nonexistent", "default")) # "default"
print(agent_state.get("step", 0))               # 0 (key exists, returns its value)

# LangGraph analogy: node checking if optional state field exists
def process_node(state):
    # Never: state["tool_calls"]  — crashes if tool_calls not in state
    # Always:
    tool_calls = state.get("tool_calls", [])
    if tool_calls:
        print("Processing tool calls...")
    return {}

process_node(agent_state)
```

---

#### 4.1.2 — Dictionary Mutation

```python
state = {"messages": [], "step": 0}

# Add or update a key
state["model"] = "gpt-4o"
state["step"] = 1
print(state)

# Delete a key
del state["model"]
# state.pop("model")  # same but returns the deleted value
removed = state.pop("step", None)   # safe pop — returns None if key missing
print(f"Removed: {removed}")

# Update multiple keys at once (critical for LangGraph node returns!)
updates = {"step": 2, "is_done": False, "last_action": "search"}
state.update(updates)
print(state)

# The ** unpacking operator — merges dicts (Python 3.5+)
base_state = {"messages": [], "step": 0}
node_output = {"step": 1, "last_action": "search"}
new_state = {**base_state, **node_output}  # node_output values WIN on conflict
print(new_state)

# This is how LangGraph merges state internally!
```

---

#### 4.1.3 — Dictionary Methods: The Full Toolkit

```python
config = {
    "model": "gpt-4o",
    "temperature": 0.7,
    "max_tokens": 1024,
    "stream": True
}

# Keys, values, items (they return VIEW objects, not lists)
print(config.keys())    # dict_keys(['model', 'temperature', ...])
print(config.values())  # dict_values(['gpt-4o', 0.7, ...])
print(config.items())   # dict_items([('model', 'gpt-4o'), ...])

# Convert to list when you need to index or slice
keys_list = list(config.keys())
print(keys_list[0])     # model

# Iterate
for key, value in config.items():
    print(f"  {key}: {value}")

# Membership test — checks KEYS only
print("model" in config)          # True
print("gpt-4o" in config)         # False (that's a value, not a key!)
print("gpt-4o" in config.values()) # True

# setdefault — set a key ONLY if it doesn't exist
config.setdefault("timeout", 30)   # sets timeout to 30
config.setdefault("model", "gpt-3.5")  # does NOT change model (already exists)
print(config["model"])   # still gpt-4o
print(config["timeout"]) # 30
```

---

#### 4.1.4 — Nested Dictionaries (LangGraph State is Always Nested)

```python
# A realistic LangGraph-style state
state = {
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Search for Python tutorials."},
    ],
    "tool_calls": [
        {"id": "call_001", "name": "web_search", "args": {"query": "Python tutorials"}}
    ],
    "tool_results": {},
    "metadata": {
        "session_id": "abc-123",
        "turn": 3,
        "model": "gpt-4o",
        "usage": {"prompt_tokens": 100, "completion_tokens": 50}
    },
    "is_done": False
}

# Accessing nested values
print(state["messages"][0]["content"])              # You are a helpful assistant.
print(state["metadata"]["usage"]["prompt_tokens"])  # 100
print(state["tool_calls"][0]["args"]["query"])      # Python tutorials

# Safe nested access — no built-in operator, must chain .get()
print(state.get("metadata", {}).get("usage", {}).get("prompt_tokens", 0))  # 100
print(state.get("nonexistent", {}).get("also_missing", "default"))          # default

# Updating nested values
state["metadata"]["turn"] += 1
state["metadata"]["usage"]["completion_tokens"] += 25
print(state["metadata"])
```

---

#### 4.1.5 — Dict Comprehensions and Copying (THE TRAP)

```python
# Dict comprehension
config = {"model": "gpt-4o", "temperature": 0.7, "max_tokens": 1024}

# Filter: keep only non-default settings
defaults = {"temperature": 0.7, "max_tokens": 1024}
custom = {k: v for k, v in config.items() if k not in defaults}
print(custom)  # {'model': 'gpt-4o'}

# Transform: uppercase all string values
upper_config = {k: v.upper() if isinstance(v, str) else v for k, v in config.items()}
print(upper_config)

# --- THE MOST DANGEROUS BUG IN LANGGRAPH CODE ---
print("\n=== THE SHALLOW COPY TRAP ===")
import copy

state_a = {
    "messages": [{"role": "user", "content": "Hello"}],
    "step": 0
}

# WRONG: assignment — no copy at all, same object!
state_b = state_a
state_b["step"] = 99
print(f"state_a step: {state_a['step']}")  # 99! — same object

# WRONG: shallow copy — top level is new dict, but nested objects are shared
state_c = state_a.copy()   # or dict(state_a) or {**state_a}
state_c["messages"].append({"role": "assistant", "content": "Hi"})
print(f"state_a messages: {state_a['messages']}")  # MODIFIED! shared list!

# CORRECT: deep copy — fully independent copy
state_d = copy.deepcopy(state_a)
state_d["messages"].append({"role": "user", "content": "New message"})
print(f"state_a messages after deepcopy: {state_a['messages']}")  # unchanged!
```

**ASCII Diagram: Shallow vs Deep Copy**

```
state_a = {"messages": [msg1], "step": 0}

Shallow copy (state_a.copy()):
  state_c ──► { "messages": ──┐ "step": 0 }   ← new dict
                               │
  state_a ──► { "messages": ──┘ "step": 0 }   ← old dict
                               │
                               ▼
                          [msg1]   ← SAME LIST OBJECT!

Deep copy (copy.deepcopy(state_a)):
  state_d ──► { "messages": ──► [msg1_copy] }  ← fully independent
  state_a ──► { "messages": ──► [msg1] }       ← untouched
```

> **LangGraph Note:** LangGraph handles state immutability through its reducer system — nodes return NEW values, not mutations of existing state. But if you pass a list from state into a function and that function appends to it, you've mutated the state object directly. This causes hard-to-debug issues with checkpointing.

---

#### 4.1.6 — Merging Nested Dicts (Writing Your Own Reducer)

```python
def merge_states(old_state, new_partial):
    """
    Mimics LangGraph's default state reducer:
    - For list fields: append new items to old list
    - For other fields: new value overwrites old value
    """
    result = dict(old_state)  # shallow copy of top level
    for key, value in new_partial.items():
        if key in result and isinstance(result[key], list) and isinstance(value, list):
            result[key] = result[key] + value  # create new list (don't mutate!)
        else:
            result[key] = value
    return result

# Test
current_state = {
    "messages": [{"role": "user", "content": "Hello"}],
    "step": 1,
    "is_done": False
}

node_output = {
    "messages": [{"role": "assistant", "content": "Hi!"}],
    "step": 2
}

next_state = merge_states(current_state, node_output)
print(next_state)
# messages: both messages merged
# step: 2 (overwritten)
# is_done: False (unchanged — not in node_output)
```

---

### Lesson 4.2 — Guided Coding Drill (1 hour)

```python
# drill_04_dicts.py

print("=== DRILL 1: State Manager ===")

def create_initial_state(session_id, system_prompt):
    """Factory function for initial LangGraph-style state."""
    return {
        "session_id": session_id,
        "messages": [{"role": "system", "content": system_prompt}],
        "tool_calls": [],
        "tool_results": {},
        "metadata": {"turn": 0, "total_tokens": 0},
        "is_done": False,
        "error": None
    }

def apply_node_output(state, node_name, output):
    """Apply a node's partial output to the full state."""
    import copy
    new_state = copy.deepcopy(state)
    for key, value in output.items():
        if key == "messages" and isinstance(value, list):
            new_state["messages"].extend(value)
        elif key == "metadata" and isinstance(value, dict):
            new_state["metadata"].update(value)
        else:
            new_state[key] = value
    new_state["metadata"]["turn"] += 1
    print(f"  [{node_name}] Applied output. Turn: {new_state['metadata']['turn']}")
    return new_state

# Simulate a 2-node graph
state = create_initial_state("sess-001", "You are a research assistant.")
print(f"Initial state keys: {list(state.keys())}")

# Node 1: model response
state = apply_node_output(state, "llm_node", {
    "messages": [{"role": "assistant", "content": "I'll search for that."}],
    "tool_calls": [{"id": "c1", "name": "web_search", "args": {"query": "AI news"}}],
    "metadata": {"total_tokens": 150}
})

# Node 2: tool execution
state = apply_node_output(state, "tool_node", {
    "tool_results": {"c1": "Top AI story: ..."},
    "messages": [{"role": "tool", "content": "Top AI story: ...", "tool_call_id": "c1"}]
})

print(f"\nFinal state:")
for key, val in state.items():
    print(f"  {key}: {val}")
```

---

### Lesson 4.3 — Homework Challenge (1 hour)

**Challenge: Nested Dict Reducer**

Write a function `deep_merge(base, updates)` that:

1. Performs a true recursive merge of two nested dicts.
2. For list values: concatenates them (don't overwrite).
3. For dict values: recursively merges.
4. For all other values: `updates` wins.
5. Never mutates `base` (always returns a new dict).

Write 6 test cases including: empty dicts, single-level override, nested dict merge, list concatenation, mixed types, deeply nested (3+ levels).

This is essentially what LangGraph's Annotated reducers do.

---

## MODULE 5: Sets & Control Flow

**Duration: 3 hours** | **Theme: Logic gates and unique collections for routing decisions**

---

### Lesson 5.1 — Concept Lecture (1 hour)

**Why this matters for LangGraph:**
LangGraph graphs have conditional edges — routing functions that look at state and return the name of the next node. These routing functions are built entirely from Python control flow. Sets are perfect for checking which tools are available, which nodes have been visited, or which capabilities are enabled.

---

#### 5.1.1 — Sets: Unique, Unordered Collections

```python
# Creation
available_tools = {"web_search", "calculator", "code_interpreter"}
requested_tools = set(["web_search", "calculator", "web_search"])  # dedup!
print(requested_tools)    # {'web_search', 'calculator'} — no duplicates

# Membership (O(1) — much faster than list for large sets!)
print("web_search" in available_tools)     # True
print("file_upload" in available_tools)    # False

# Set operations — perfect for tool permission logic
user_permissions = {"web_search", "calculator"}
admin_permissions = {"web_search", "calculator", "file_delete", "shell_exec"}

# Union: all permissions from both
all_perms = user_permissions | admin_permissions
print(all_perms)

# Intersection: permissions in BOTH
common = user_permissions & admin_permissions
print(common)  # {'web_search', 'calculator'}

# Difference: in admin but NOT user (admin-only tools)
admin_only = admin_permissions - user_permissions
print(admin_only)  # {'file_delete', 'shell_exec'}

# Symmetric difference: in either but NOT both
exclusive = user_permissions ^ admin_permissions
print(exclusive)

# Subset/superset checks
print(user_permissions.issubset(admin_permissions))    # True
print(admin_permissions.issuperset(user_permissions))  # True

# Mutation
available_tools.add("file_read")
available_tools.discard("nonexistent")  # safe remove (no error if missing)
available_tools.remove("calculator")    # unsafe remove (KeyError if missing)
```

---

#### 5.1.2 — Control Flow: if/elif/else

```python
# Basic if/elif/else
def route_to_node(state):
    """LangGraph conditional edge — returns next node name."""
    if state.get("error"):
        return "error_handler"
    elif state.get("is_done"):
        return "END"
    elif state.get("tool_calls"):
        return "tool_node"
    else:
        return "llm_node"

# Test all branches
print(route_to_node({"error": "timeout"}))         # error_handler
print(route_to_node({"is_done": True}))            # END
print(route_to_node({"tool_calls": ["search"]}))   # tool_node
print(route_to_node({}))                           # llm_node

# Ternary (conditional expression) — single-line if/else
step = 5
status = "active" if step < 10 else "exceeded"
print(status)

# Chained comparisons (Pythonic!)
score = 0.75
if 0.7 <= score < 0.9:
    print("Good score")

# Short-circuit evaluation
def is_valid_state(state):
    # Python evaluates left to right, stops early (short-circuit)
    return (
        state is not None
        and isinstance(state, dict)
        and "messages" in state
        and len(state["messages"]) > 0
    )

print(is_valid_state(None))         # False (stops at first check)
print(is_valid_state({}))           # False (stops at "messages" check)
print(is_valid_state({"messages": []}))  # False (empty list)
print(is_valid_state({"messages": ["x"]}))  # True
```

---

#### 5.1.3 — for Loops and range()

```python
# Basic for loop
tools = ["web_search", "calculator", "code_interpreter"]
for tool in tools:
    print(f"Registering tool: {tool}")

# range(stop), range(start, stop), range(start, stop, step)
for i in range(5):          # 0, 1, 2, 3, 4
    print(i, end=" ")
print()

for i in range(1, 6):       # 1, 2, 3, 4, 5
    print(i, end=" ")
print()

for i in range(0, 10, 2):   # 0, 2, 4, 6, 8
    print(i, end=" ")
print()

for i in range(5, 0, -1):   # 5, 4, 3, 2, 1 (countdown)
    print(i, end=" ")
print()

# enumerate — ALWAYS use this instead of range(len(...))
messages = ["Hello", "Hi!", "Thanks"]
for i, msg in enumerate(messages):
    print(f"[{i}] {msg}")

for i, msg in enumerate(messages, start=1):  # start index at 1
    print(f"Turn {i}: {msg}")

# zip — iterate two lists together
names = ["Alice", "Bob", "Charlie"]
scores = [92, 87, 95]
for name, score in zip(names, scores):
    print(f"{name}: {score}")

# Iterating over dict
config = {"model": "gpt-4o", "temp": 0.7, "tokens": 1024}
for key in config:              # iterates keys
    print(key)

for key, val in config.items(): # iterates key-value pairs
    print(f"{key} = {val}")
```

---

#### 5.1.4 — while Loops, break, and continue

```python
# while loop — the agent's main execution loop!
def run_agent_loop(initial_state, max_steps=10):
    state = initial_state
    step = 0

    while not state.get("is_done") and step < max_steps:
        print(f"Step {step}: Processing...")

        # Simulate node execution
        if step == 0:
            state = {**state, "messages": state["messages"] + ["assistant: thinking"]}
        elif step == 1:
            state = {**state, "tool_calls": ["search"]}
        elif step == 2:
            state = {**state, "tool_calls": [], "is_done": True}

        step += 1

    if step >= max_steps:
        print(f"WARNING: Hit max steps ({max_steps})")
    else:
        print(f"Agent finished in {step} steps.")

    return state

final = run_agent_loop({"messages": [], "is_done": False})

# break — exit loop immediately
print("\n--- break example ---")
for step in range(100):
    if step == 3:
        print(f"Breaking at step {step}")
        break
    print(f"Running step {step}")

# continue — skip rest of this iteration, go to next
print("\n--- continue example ---")
for item in ["web_search", None, "calculator", None, "code_exec"]:
    if item is None:
        continue   # skip None tools
    print(f"Registering: {item}")

# else clause on loops — runs ONLY if loop was NOT broken
print("\n--- for/else example ---")
target = "nonexistent_tool"
available = ["web_search", "calculator"]
for tool in available:
    if tool == target:
        print(f"Found {target}")
        break
else:
    print(f"Tool '{target}' not found — using fallback")  # this runs!
```

---

### Lesson 5.2 — Guided Coding Drill (1 hour)

```python
# drill_05_control_flow.py

print("=== DRILL: Agent Router ===")

# Simulate a LangGraph-style conditional routing system
TOOL_REGISTRY = {"web_search", "calculator", "code_interpreter", "file_read"}

def determine_routing(state):
    """
    Mimics LangGraph's conditional edge logic.
    Returns: name of next node
    """
    # Priority 1: handle errors
    if state.get("error"):
        error_type = state["error"].get("type", "unknown")
        if error_type == "rate_limit":
            return "retry_node"
        elif error_type == "auth_error":
            return "END"
        else:
            return "error_handler"

    # Priority 2: check completion
    if state.get("is_done"):
        return "END"

    # Priority 3: process pending tool calls
    tool_calls = state.get("tool_calls", [])
    if tool_calls:
        # Validate tools are registered
        tool_names = {tc["name"] for tc in tool_calls}
        unknown = tool_names - TOOL_REGISTRY
        if unknown:
            state["error"] = {"type": "unknown_tool", "tools": list(unknown)}
            return "error_handler"
        return "tool_node"

    # Priority 4: check step limit
    step = state.get("metadata", {}).get("turn", 0)
    if step >= 10:
        state["is_done"] = True
        return "END"

    # Default: call the LLM again
    return "llm_node"

# Run test cases
test_states = [
    {"error": {"type": "rate_limit"}},
    {"is_done": True},
    {"tool_calls": [{"name": "web_search", "args": {}}]},
    {"tool_calls": [{"name": "unknown_tool", "args": {}}]},
    {"metadata": {"turn": 11}},
    {},
]

for i, s in enumerate(test_states):
    print(f"State {i} → {determine_routing(s)}")
```

---

### Lesson 5.3 — Homework Challenge (1 hour)

**Challenge: State Machine Simulator**

Build a `StateMachine` class with:

1. A `states` set storing valid state names.
2. A `transitions` dict mapping `(from_state, event) → to_state`.
3. A `current` attribute storing current state.
4. An `add_state(name)` method.
5. An `add_transition(from_state, event, to_state)` method.
6. A `trigger(event)` method that transitions state or raises `ValueError` if invalid.
7. A `run_sequence(events)` method that runs a list of events and returns the path taken.

Build a state machine modeling a simple agent lifecycle: `idle → running → waiting_for_tools → running → done`.

---

## MODULE 6: Functions — The Atom of Agent Behavior

**Duration: 3 hours** | **Theme: Every node, every tool, every reducer is a function**

---

### Lesson 6.1 — Concept Lecture (1 hour)

**Why this matters for LangGraph:**
In LangGraph, _everything is a function_. A node is a function that takes state and returns a partial state. A tool is a function with a name and description. A reducer is a function that merges values. Edges can be functions. Deeply understanding function definition, arguments, scope, and first-class usage is the entry ticket to LangGraph.

---

#### 6.1.1 — Defining Functions

```python
# Basic definition
def greet(name):
    """Docstring: describes what this function does."""
    return f"Hello, {name}!"

result = greet("Alice")
print(result)

# Functions are objects — they can be assigned to variables!
say_hello = greet              # no parentheses — we're not calling it
print(say_hello("Bob"))        # Hello, Bob!
print(type(greet))             # <class 'function'>

# Functions without return return None
def log_step(message):
    print(f"[LOG] {message}")
    # implicit: return None

val = log_step("Starting agent")
print(val)   # None
```

---

#### 6.1.2 — Arguments: Positional, Keyword, Default, \*args, \*\*kwargs

```python
# Positional arguments
def add(a, b):
    return a + b

print(add(3, 4))         # positional: a=3, b=4
print(add(b=4, a=3))    # keyword: same result

# Default arguments
def create_message(content, role="user", model="gpt-4o"):
    return {"role": role, "content": content, "model": model}

print(create_message("Hello"))                              # role="user", model="gpt-4o"
print(create_message("Search for cats", role="assistant")) # override role

# THE MUTABLE DEFAULT ARGUMENT TRAP — Most Common Python Bug!
def bad_append(item, lst=[]):   # DON'T DO THIS
    lst.append(item)
    return lst

print(bad_append("a"))   # ['a']
print(bad_append("b"))   # ['a', 'b'] — WRONG! list persists between calls!
print(bad_append("c"))   # ['a', 'b', 'c'] — still wrong!

# CORRECT: use None as default, create inside function
def good_append(item, lst=None):
    if lst is None:
        lst = []
    lst.append(item)
    return lst

print(good_append("a"))   # ['a']
print(good_append("b"))   # ['b'] — fresh list each time!

# *args — variable positional arguments (packed into a tuple)
def log(*messages):
    for i, msg in enumerate(messages):
        print(f"  [{i}] {msg}")

log("Step 1 done", "Tool called", "Waiting for response")

# **kwargs — variable keyword arguments (packed into a dict)
def configure_model(**settings):
    defaults = {"temperature": 0.7, "max_tokens": 1024, "stream": False}
    defaults.update(settings)   # override defaults with provided settings
    return defaults

config = configure_model(temperature=0.0, model="gpt-4o")
print(config)

# Combining all argument types (order matters!)
def complex_func(required, *args, keyword="default", **kwargs):
    print(f"required: {required}")
    print(f"args: {args}")
    print(f"keyword: {keyword}")
    print(f"kwargs: {kwargs}")

complex_func("first", "second", "third", keyword="custom", extra1=1, extra2=2)
```

---

#### 6.1.3 — Scope: LEGB Rule

```python
# LEGB: Local → Enclosing → Global → Built-in

GLOBAL_MODEL = "gpt-4o"    # Global scope

def outer():
    enclosing_var = "I'm in outer"   # Enclosing scope

    def inner():
        local_var = "I'm in inner"   # Local scope
        print(local_var)
        print(enclosing_var)         # found in Enclosing
        print(GLOBAL_MODEL)          # found in Global

    inner()

outer()

# global keyword (use sparingly — often a design smell)
counter = 0
def increment():
    global counter
    counter += 1

increment()
increment()
print(counter)   # 2

# nonlocal — modify enclosing (non-global) scope
def make_counter():
    count = 0
    def increment():
        nonlocal count
        count += 1
        return count
    return increment

counter_fn = make_counter()
print(counter_fn())   # 1
print(counter_fn())   # 2
print(counter_fn())   # 3
# Each call to make_counter() creates independent counter
counter2 = make_counter()
print(counter2())     # 1 — independent!
```

---

#### 6.1.4 — First-Class Functions and Lambda

```python
# Functions as arguments
def apply_to_each(items, func):
    """Apply a function to every item in a list."""
    return [func(item) for item in items]

tools = ["web_search", "calculator", "code_exec"]
display = apply_to_each(tools, lambda name: name.replace("_", " ").title())
print(display)

# Lambda functions: anonymous, single-expression functions
double = lambda x: x * 2
add = lambda a, b: a + b

print(double(5))
print(add(3, 4))

# Lambdas are most useful as short callbacks
messages = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"},
    {"role": "user", "content": "What is AI?"},
]

# Sort by content length using lambda
sorted_msgs = sorted(messages, key=lambda m: len(m["content"]))
for m in sorted_msgs:
    print(f"{len(m['content'])}: {m['content'][:30]}")

# Filter using lambda
user_only = list(filter(lambda m: m["role"] == "user", messages))
print(user_only)

# map() using lambda
contents = list(map(lambda m: m["content"], messages))
print(contents)

# Functions stored in dicts — the tool registry pattern!
def search_web(query):
    return f"[Web results for: {query}]"

def run_calculator(expression):
    return str(eval(expression))   # Note: never use eval in production!

TOOL_REGISTRY = {
    "web_search": search_web,
    "calculator": run_calculator,
}

def execute_tool(name, args):
    if name not in TOOL_REGISTRY:
        raise ValueError(f"Unknown tool: {name}")
    tool_fn = TOOL_REGISTRY[name]
    return tool_fn(**args)

print(execute_tool("web_search", {"query": "Python tutorials"}))
print(execute_tool("calculator", {"expression": "2 ** 10"}))
```

**LangGraph Analogy:**
`TOOL_REGISTRY` above is _literally_ how LangChain's `ToolNode` works — it maintains a dict of `tool_name → tool_function` and dispatches based on the `tool_calls` in state.

---

#### 6.1.5 — map(), filter(), zip(), any(), all(), enumerate()

```python
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# map(func, iterable) — apply func to each element, returns iterator
doubled = list(map(lambda x: x * 2, numbers))
print(doubled)

# Equivalent comprehension (prefer this for readability)
doubled_comp = [x * 2 for x in numbers]

# filter(func, iterable) — keep elements where func returns True
evens = list(filter(lambda x: x % 2 == 0, numbers))
print(evens)

# zip — combine iterables
names = ["Alice", "Bob", "Charlie"]
ages = [30, 25, 35]
cities = ["NYC", "LA", "Chicago"]
combined = list(zip(names, ages, cities))
print(combined)

# Unzip with *
unzipped_names, unzipped_ages, unzipped_cities = zip(*combined)
print(unzipped_names)

# any() / all() — critical for multi-condition state checks
tool_calls = [
    {"name": "web_search", "complete": True},
    {"name": "calculator", "complete": True},
    {"name": "code_exec", "complete": False},
]

all_done = all(tc["complete"] for tc in tool_calls)
any_done = any(tc["complete"] for tc in tool_calls)
print(f"All tools done: {all_done}")   # False
print(f"Any tool done: {any_done}")    # True

# enumerate
for i, (name, age) in enumerate(zip(names, ages)):
    print(f"{i}. {name} is {age} years old")
```

---

### Lesson 6.2 — Guided Coding Drill (1 hour)

```python
# drill_06_functions.py

print("=== DRILL: Function-Based Tool System ===")

# --- Part 1: Tool Registration System ---
tool_registry = {}

def register_tool(name, description):
    """Decorator-free tool registration using a plain function."""
    def registrar(func):
        tool_registry[name] = {
            "function": func,
            "description": description,
            "name": name
        }
        return func
    return registrar

# Register tools
def web_search(query, max_results=5):
    """Simulate web search."""
    return f"Results for '{query}': [result1, result2, ...] (showing {max_results})"

def calculate(expression):
    """Safely evaluate a math expression."""
    allowed_chars = set("0123456789+-*/()., ")
    if not all(c in allowed_chars for c in expression):
        return "Error: Invalid characters in expression"
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"

def get_current_time():
    """Return current time."""
    import datetime
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Manually register (not using decorator to avoid Bridge Topics)
tool_registry["web_search"] = {"function": web_search, "description": "Search the web", "name": "web_search"}
tool_registry["calculate"] = {"function": calculate, "description": "Evaluate a math expression", "name": "calculate"}
tool_registry["time"] = {"function": get_current_time, "description": "Get current date/time", "name": "time"}

# --- Part 2: Tool Executor ---
def execute_tool_calls(tool_calls):
    """
    Execute a list of tool calls and return results.
    Mimics LangGraph's ToolNode behavior.
    """
    results = {}
    for call in tool_calls:
        tool_name = call.get("name")
        args = call.get("args", {})
        call_id = call.get("id", tool_name)

        if tool_name not in tool_registry:
            results[call_id] = {"error": f"Unknown tool: {tool_name}"}
            continue

        try:
            tool_fn = tool_registry[tool_name]["function"]
            result = tool_fn(**args)
            results[call_id] = {"result": result, "success": True}
        except TypeError as e:
            results[call_id] = {"error": f"Invalid args: {e}", "success": False}
        except Exception as e:
            results[call_id] = {"error": str(e), "success": False}

    return results

# Test
tool_calls = [
    {"id": "c1", "name": "web_search", "args": {"query": "Python tutorials", "max_results": 3}},
    {"id": "c2", "name": "calculate", "args": {"expression": "100 * 3.14159"}},
    {"id": "c3", "name": "time", "args": {}},
    {"id": "c4", "name": "unknown_tool", "args": {}},
]

results = execute_tool_calls(tool_calls)
for call_id, result in results.items():
    print(f"{call_id}: {result}")

# --- Part 3: List all tools ---
print("\n=== Available Tools ===")
for name, info in tool_registry.items():
    print(f"  {name}: {info['description']}")
```

---

### Lesson 6.3 — Homework Challenge (1 hour)

**Challenge: Function-Based Pipeline**

Build a `Pipeline` class using only functions and lists:

1. `__init__(self)` initializes an empty list of steps.
2. `add_step(name, func)` appends `{"name": name, "fn": func}`.
3. `run(initial_input)` executes each step in sequence, passing the output of one step as the input to the next.
4. `run` returns a dict `{"output": final_value, "steps": [{"name": str, "input": any, "output": any}]}`.
5. If any step raises an exception, stop and return `{"error": str, "failed_at_step": name, "steps": [...completed steps...]}`.

Build a pipeline that: normalizes text → extracts keywords → counts them → formats output.

---

## MODULE 7: OOP Fundamentals — Building Your Own Tools

**Duration: 3 hours** | **Theme: Classes are the blueprint for BaseTool, BaseChatModel, and Runnable**

---

### Lesson 7.1 — Concept Lecture (1 hour)

**Why this matters for LangGraph:**
LangChain's entire public API is built on classes. `BaseTool`, `BaseChatModel`, `BaseMemory`, `BaseRetriever` — all classes. When you call `tool.invoke(input)`, you're calling a method on an object. When you define your own tool, you subclass `BaseTool` and override `_run()`. You cannot productively use LangChain without understanding Python OOP.

---

#### 7.1.1 — Classes and **init**

```python
class AgentTool:
    """Base class representing a tool available to an agent."""

    # Class attribute — shared by ALL instances
    tool_count = 0

    def __init__(self, name, description, max_retries=3):
        """
        Instance initializer.
        self = the specific instance being created.
        All instance attributes must go on self.
        """
        # Instance attributes — unique to each instance
        self.name = name
        self.description = description
        self.max_retries = max_retries
        self.call_count = 0
        self._is_enabled = True   # "private" by convention

        # Increment class-level counter
        AgentTool.tool_count += 1

    def __repr__(self):
        """String representation for debugging."""
        return f"AgentTool(name='{self.name}', enabled={self._is_enabled})"

    def __str__(self):
        """Human-readable string representation."""
        return f"{self.name}: {self.description}"


# Instantiation
search_tool = AgentTool("web_search", "Search the internet for information")
calc_tool = AgentTool("calculator", "Evaluate math expressions", max_retries=1)

print(search_tool)              # web_search: Search the internet for information
print(repr(search_tool))        # AgentTool(name='web_search', enabled=True)
print(AgentTool.tool_count)     # 2
print(search_tool.tool_count)   # 2 — accessible on instance too
print(search_tool.name)         # web_search
print(calc_tool.name)           # calculator

# Each instance has its OWN call_count
search_tool.call_count += 1
print(search_tool.call_count)   # 1
print(calc_tool.call_count)     # 0 — unaffected!
```

**ASCII Diagram: Class vs Instance Attributes**

```
AgentTool (class)
├── tool_count = 2          ← shared by all instances

search_tool (instance)
├── name = "web_search"     ← unique to this instance
├── description = "..."
├── max_retries = 3
└── call_count = 1

calc_tool (instance)
├── name = "calculator"     ← unique to this instance
├── description = "..."
├── max_retries = 1
└── call_count = 0
```

---

#### 7.1.2 — Methods

```python
class AgentTool:

    def __init__(self, name, description, max_retries=3):
        self.name = name
        self.description = description
        self.max_retries = max_retries
        self.call_count = 0
        self._enabled = True

    # Instance method — has access to self (the instance)
    def invoke(self, input_data):
        """Execute the tool. Subclasses override _run()."""
        if not self._enabled:
            raise RuntimeError(f"Tool '{self.name}' is disabled")
        self.call_count += 1
        result = self._run(input_data)
        return result

    def _run(self, input_data):
        """Override this in subclasses. Default implementation."""
        return f"[{self.name}] received: {input_data}"

    def enable(self):
        self._enabled = True

    def disable(self):
        self._enabled = False

    def get_schema(self):
        """Returns dict describing this tool (like LangChain's tool schema)."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {}  # subclasses fill this in
        }

    # Class method — has access to the class, not an instance
    @classmethod
    def from_dict(cls, data):
        """Alternative constructor from a dict."""
        return cls(
            name=data["name"],
            description=data["description"],
            max_retries=data.get("max_retries", 3)
        )

    # Static method — neither class nor instance
    @staticmethod
    def validate_input(input_data):
        """Check if input is valid. No self or cls needed."""
        return isinstance(input_data, str) and len(input_data) > 0

    def __repr__(self):
        return f"AgentTool('{self.name}', calls={self.call_count})"


# Test
tool = AgentTool("search", "Search the web")
print(tool.invoke("Python tutorials"))
print(tool.invoke("LangChain docs"))
print(tool.call_count)   # 2

# Class method
tool2 = AgentTool.from_dict({"name": "calc", "description": "Math"})
print(tool2)

# Static method
print(AgentTool.validate_input("hello"))   # True
print(AgentTool.validate_input(""))        # False
```

---

#### 7.1.3 — Inheritance

```python
class AgentTool:
    """Base class."""

    def __init__(self, name, description):
        self.name = name
        self.description = description
        self.call_count = 0

    def invoke(self, input_data):
        self.call_count += 1
        return self._run(input_data)

    def _run(self, input_data):
        raise NotImplementedError("Subclasses must implement _run()")

    def get_schema(self):
        return {"name": self.name, "description": self.description}

    def __repr__(self):
        return f"{self.__class__.__name__}('{self.name}')"


class WebSearchTool(AgentTool):
    """Concrete tool: web search."""

    def __init__(self, name="web_search", max_results=5):
        # Call parent __init__ first!
        super().__init__(name, "Search the web for information")
        # Then add subclass-specific attributes
        self.max_results = max_results

    def _run(self, query):
        # Override the abstract method
        return f"[Search: {query}] Found {self.max_results} results."

    def get_schema(self):
        base = super().get_schema()  # get parent's schema
        base["parameters"] = {
            "query": {"type": "string", "description": "Search query"},
            "max_results": {"type": "integer", "default": self.max_results}
        }
        return base


class CalculatorTool(AgentTool):

    def __init__(self):
        super().__init__("calculator", "Evaluate mathematical expressions")

    def _run(self, expression):
        allowed = set("0123456789+-*/()., ")
        if not all(c in allowed for c in expression):
            return f"Error: Invalid characters"
        try:
            return str(eval(expression))
        except Exception as e:
            return f"Error: {e}"


# Polymorphism — same interface, different behavior
tools = [WebSearchTool(max_results=3), CalculatorTool()]

for tool in tools:
    print(f"\n{tool}")
    print(f"  Schema: {tool.get_schema()}")
    result = tool.invoke("2 + 2" if "calc" in tool.name else "Python best practices")
    print(f"  Result: {result}")

# isinstance checks
print(isinstance(tools[0], WebSearchTool))  # True
print(isinstance(tools[0], AgentTool))       # True (inheritance!)
print(isinstance(tools[0], CalculatorTool))  # False
```

---

#### 7.1.4 — The @property Decorator (Simple Version)

```python
# Note: We're teaching the concept without the full decorator theory
# @property is a built-in decorator — it's acceptable to use before learning decorators

class AgentConfig:
    """Configuration object with validated properties."""

    def __init__(self, model, temperature, max_tokens):
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    @property
    def temperature(self):
        """Getter: called when you access obj.temperature"""
        return self._temperature

    @temperature.setter
    def temperature(self, value):
        """Setter: called when you do obj.temperature = x"""
        if not 0.0 <= value <= 2.0:
            raise ValueError(f"Temperature must be 0.0-2.0, got {value}")
        self._temperature = value

    @property
    def model(self):
        return self._model

    @property
    def max_tokens(self):
        return self._max_tokens

    @max_tokens.setter
    def max_tokens(self, value):
        if value < 1 or value > 128000:
            raise ValueError(f"max_tokens out of range: {value}")
        self._max_tokens = value

    def to_dict(self):
        return {
            "model": self._model,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens
        }

    def __repr__(self):
        return f"AgentConfig(model={self._model}, temp={self._temperature})"


config = AgentConfig("gpt-4o", 0.7, 1024)
print(config.temperature)    # 0.7 — calls getter
config.temperature = 0.0     # calls setter
print(config.temperature)    # 0.0

try:
    config.temperature = 5.0   # out of range
except ValueError as e:
    print(f"Caught: {e}")
```

---

### Lesson 7.2 — Guided Coding Drill (1 hour)

```python
# drill_07_oop.py

print("=== Building a Mini LangChain Tool System ===")

class BaseTool:
    """
    Mimics LangChain's BaseTool.
    This is the actual pattern LangChain uses!
    """

    def __init__(self):
        # These should be defined by subclasses
        if not hasattr(self, 'name'):
            raise NotImplementedError("Subclasses must define 'name'")
        if not hasattr(self, 'description'):
            raise NotImplementedError("Subclasses must define 'description'")

    def invoke(self, input_data):
        """Public method — validates input then calls _run."""
        if not isinstance(input_data, (str, dict)):
            raise TypeError(f"Input must be str or dict, got {type(input_data)}")
        return self._run(input_data)

    def _run(self, input_data):
        raise NotImplementedError("Subclasses must implement _run()")

    def get_schema(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self._get_parameters()
            }
        }

    def _get_parameters(self):
        return {"type": "object", "properties": {}, "required": []}

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"


class WebSearchTool(BaseTool):
    name = "web_search"
    description = "Search the internet. Input: search query string."

    def __init__(self, num_results=5):
        super().__init__()
        self.num_results = num_results

    def _run(self, query):
        if isinstance(query, dict):
            query = query.get("query", "")
        return f"Web results for '{query}': [article1, article2, ...] ({self.num_results} total)"

    def _get_parameters(self):
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"}
            },
            "required": ["query"]
        }


class PythonREPLTool(BaseTool):
    name = "python_repl"
    description = "Execute Python code. Input: valid Python code string."

    def _run(self, code):
        import io, sys
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            exec(code)
            output = sys.stdout.getvalue()
        except Exception as e:
            output = f"Error: {e}"
        finally:
            sys.stdout = old_stdout
        return output or "(no output)"


# Tool registry pattern
tools = [WebSearchTool(num_results=3), PythonREPLTool()]
tool_map = {t.name: t for t in tools}

print("Registered tools:")
for name, tool in tool_map.items():
    print(f"  {tool}")

print("\nExecuting tools:")
print(tool_map["web_search"].invoke("LangGraph tutorial"))
print(tool_map["python_repl"].invoke("print(sum(range(10)))"))
print(tool_map["python_repl"].invoke({"query": "wrong input type for test"}))  # will use dict path
```

---

### Lesson 7.3 — Homework Challenge (1 hour)

**Challenge: Build a Mini BaseModel**

Create a `BaseModel` class that mimics a simplified version of what Pydantic does (without actually using Pydantic — that's a Bridge Topic). Your class should:

1. Define class-level `fields` dict: `{"field_name": {"type": type, "default": value_or_REQUIRED}}`.
2. `__init__(**kwargs)` validates types and raises `TypeError` for wrong types, `ValueError` for missing required fields.
3. `to_dict()` method that returns all field values as a dict.
4. `from_dict(cls, data)` class method.
5. `__repr__` showing class name and all field values.

Build a `ChatMessage(BaseModel)` with fields: `role` (str, required), `content` (str, required), `name` (str, default=None).

---

## MODULE 8: Modules, Imports & Project Structure

**Duration: 3 hours** | **Theme: How to organize a multi-file agent project**

---

### Lesson 8.1 — Concept Lecture (1 hour)

**Why this matters for LangGraph:**
Real LangGraph projects are multi-file. You'll have `state.py`, `nodes.py`, `tools.py`, `graph.py`, and `main.py`. Understanding Python imports is essential for structuring your project properly and debugging `ImportError` and circular imports.

---

#### 8.1.1 — Module Basics

Every `.py` file is a module. Every directory with an `__init__.py` is a package.

```
my_agent/
├── __init__.py          ← makes this directory a package
├── state.py             ← State TypedDict definition
├── nodes.py             ← Node functions
├── tools.py             ← Tool definitions
├── graph.py             ← Graph assembly
└── main.py              ← Entry point
```

---

#### 8.1.2 — Import Styles

```python
# Style 1: import module
import os
import json

result = os.path.join("agents", "planner")   # use module.attribute
data = json.dumps({"key": "value"})

# Style 2: import specific names from a module
from os.path import join, exists
from json import dumps, loads

result = join("agents", "planner")   # no prefix needed

# Style 3: import with alias
import json as j
from os.path import join as path_join

data = j.dumps({"key": "value"})
result = path_join("agents", "planner")

# Style 4: import all (AVOID — pollutes namespace)
# from os.path import *   # bad practice

# Standard library modules you'll use constantly
import os           # file system, environment variables
import sys          # Python path, exit
import json         # JSON serialization
import copy         # deep copy
import datetime     # dates and times
import pathlib      # modern path manipulation
import re           # regular expressions
import time         # sleep, timestamps
import logging      # proper logging (not print!)
```

---

#### 8.1.3 — Structuring a Project

```python
# state.py
from typing import List, Dict, Optional, Any

# We'll use a plain dict with type annotations instead of TypedDict
# (TypedDict requires typing module — fine to use)
from typing import TypedDict

class AgentState(TypedDict, total=False):
    """
    The shared state dictionary passed between all nodes.
    total=False means all keys are optional (can be partial).
    """
    messages: List[Dict[str, str]]
    tool_calls: List[Dict[str, Any]]
    tool_results: Dict[str, Any]
    is_done: bool
    error: Optional[str]
    metadata: Dict[str, Any]
```

```python
# tools.py
from state import AgentState   # relative import alternative: from .state import ...

def web_search(query: str) -> str:
    """Search the web."""
    return f"[Results for: {query}]"

def calculator(expression: str) -> str:
    """Evaluate math."""
    allowed = set("0123456789+-*/()., ")
    if all(c in allowed for c in expression):
        return str(eval(expression))
    return "Error: invalid expression"

TOOLS = {
    "web_search": web_search,
    "calculator": calculator,
}
```

```python
# nodes.py
from state import AgentState
from tools import TOOLS

def llm_node(state: AgentState) -> AgentState:
    """Simulates calling an LLM."""
    messages = state.get("messages", [])
    # In real code: call OpenAI API here
    last_user_msg = next(
        (m["content"] for m in reversed(messages) if m["role"] == "user"),
        ""
    )
    return {
        "messages": [{"role": "assistant", "content": f"I'll help with: {last_user_msg}"}],
        "tool_calls": [{"name": "web_search", "args": {"query": last_user_msg}}]
    }

def tool_node(state: AgentState) -> AgentState:
    """Execute all pending tool calls."""
    tool_calls = state.get("tool_calls", [])
    results = {}
    new_messages = []
    for call in tool_calls:
        tool_fn = TOOLS.get(call["name"])
        if tool_fn:
            result = tool_fn(**call["args"])
            results[call["name"]] = result
            new_messages.append({"role": "tool", "content": result})
    return {
        "tool_results": results,
        "messages": new_messages,
        "tool_calls": []
    }

def should_continue(state: AgentState) -> str:
    """Conditional edge: determine next node."""
    if state.get("error"):
        return "END"
    if state.get("tool_calls"):
        return "tool_node"
    return "END"
```

```python
# main.py
import sys
import json
from state import AgentState
from nodes import llm_node, tool_node, should_continue

def run_graph(initial_state: AgentState, max_steps: int = 10) -> AgentState:
    """
    Manually run our node graph.
    In real LangGraph: graph.invoke(state) does this automatically.
    """
    state = dict(initial_state)
    step = 0

    while step < max_steps:
        # Determine next node
        next_node = should_continue(state)

        if next_node == "END":
            print(f"Graph reached END at step {step}")
            break

        # Execute the node
        print(f"Step {step}: Running {next_node}")

        if next_node == "llm_node":
            output = llm_node(state)
        elif next_node == "tool_node":
            output = tool_node(state)
        else:
            print(f"Unknown node: {next_node}")
            break

        # Merge output into state
        for key, value in output.items():
            if key == "messages" and isinstance(value, list):
                state.setdefault("messages", [])
                state["messages"] = state["messages"] + value
            else:
                state[key] = value

        step += 1

    return state

if __name__ == "__main__":
    # Entry point — only runs when this file is executed directly, not imported
    initial_state: AgentState = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Search for Python tutorials"}
        ],
        "tool_calls": [],
        "is_done": False,
        "metadata": {"session": "test-001"}
    }

    final_state = run_graph(initial_state)
    print("\n=== Final State ===")
    print(json.dumps(final_state, indent=2, default=str))
```

---

#### 8.1.4 — if **name** == "**main**"

```python
# understanding_name.py

print(f"__name__ is: {__name__}")
# When run directly: __name__ is: __main__
# When imported: __name__ is: understanding_name

def main():
    """Convention: put your main logic here."""
    print("Running main logic")

if __name__ == "__main__":
    # This block ONLY runs when the file is executed directly
    # NOT when imported as a module
    main()
```

---

### Lesson 8.2 — Guided Coding Drill (1 hour)

Build the multi-file project structure above. Create each file, then run `python main.py` and verify output.

```bash
# Create directory structure
mkdir my_agent
cd my_agent
touch __init__.py state.py tools.py nodes.py main.py
```

Write each file from the examples in 8.1.3. Then extend `tools.py` with a `datetime_tool` that returns the current date, add it to `TOOLS`, and test that `tool_node` executes it correctly.

---

### Lesson 8.3 — Homework Challenge (1 hour)

**Challenge: Plugin Loader**

Build a `PluginLoader` class that:

1. Accepts a directory path.
2. Uses `os.listdir()` to find all `.py` files in that directory.
3. Uses `importlib.import_module()` to dynamically import each file as a module.
4. Looks for a `register()` function in each module; if found, calls it and collects the returned dict of tools.
5. Merges all tools into one unified registry.
6. Has a `list_tools()` method and an `execute(tool_name, **kwargs)` method.

Create 2 sample plugin files (`search_plugin.py` and `math_plugin.py`) to test.

---

## MODULE 9: Error Handling & Context Managers

**Duration: 3 hours** | **Theme: Robust agents fail gracefully and clean up after themselves**

---

### Lesson 9.1 — Concept Lecture (1 hour)

**Why this matters for LangGraph:**
LangGraph nodes fail. Tools throw exceptions. API calls time out. A node that crashes without proper error handling takes down the entire graph. LangGraph's checkpointing and retry logic is built on top of Python's error handling primitives. You must write nodes that catch, classify, and surface errors properly.

---

#### 9.1.1 — try/except/else/finally

```python
# Basic try/except
try:
    result = 10 / 0
except ZeroDivisionError:
    print("Cannot divide by zero")

# Multiple exception types
def safe_parse(text):
    try:
        return int(text)
    except ValueError:
        print(f"Not a valid integer: '{text}'")
        return None
    except TypeError:
        print(f"Expected string, got {type(text)}")
        return None

print(safe_parse("42"))     # 42
print(safe_parse("hello"))  # Not a valid integer
print(safe_parse(None))     # TypeError

# Catching multiple in one line
def parse_number(text):
    try:
        return float(text)
    except (ValueError, TypeError):
        return 0.0

# else — runs ONLY if no exception was raised
def load_config(path):
    try:
        with open(path) as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Config not found: {path}")
        return {}
    except PermissionError:
        print(f"Cannot read: {path}")
        return {}
    else:
        # Only runs if open() succeeded
        import json
        return json.loads(content)
    # else is useful for code that should only run on success
    # but isn't "cleanup" (that's what finally is for)

# finally — ALWAYS runs, even if exception occurred
def call_api(url):
    connection = None
    try:
        print(f"Connecting to {url}")
        # Simulate connection
        connection = {"url": url, "open": True}
        raise TimeoutError("Request timed out")   # simulated failure
        return {"data": "response"}
    except TimeoutError as e:
        print(f"Timeout: {e}")
        return None
    finally:
        # This runs NO MATTER WHAT
        if connection:
            connection["open"] = False
            print(f"Connection closed (open={connection['open']})")

result = call_api("https://api.example.com")
print(f"Result: {result}")
```

---

#### 9.1.2 — Exception Hierarchy and Custom Exceptions

```python
# Python's Exception Hierarchy (partial):
# BaseException
# └── Exception
#     ├── ValueError
#     ├── TypeError
#     ├── RuntimeError
#     ├── OSError
#     │   ├── FileNotFoundError
#     │   └── PermissionError
#     ├── KeyError
#     ├── IndexError
#     ├── AttributeError
#     └── ...

# Catching base class catches all subclasses
try:
    items = [1, 2, 3]
    print(items[99])
except IndexError:
    print("Index out of range")

# except Exception catches everything except system exits
try:
    raise ValueError("bad value")
except Exception as e:
    print(f"Caught: {type(e).__name__}: {e}")

# Accessing exception info
try:
    x = int("not a number")
except ValueError as e:
    print(f"Exception type: {type(e).__name__}")
    print(f"Exception message: {e}")
    print(f"Exception args: {e.args}")

# Custom exceptions — critical for agent error taxonomy
class AgentError(Exception):
    """Base class for all agent errors."""
    def __init__(self, message, step=None, node=None):
        super().__init__(message)
        self.step = step
        self.node = node

class ToolExecutionError(AgentError):
    """Raised when a tool fails to execute."""
    def __init__(self, message, tool_name, input_data=None, **kwargs):
        super().__init__(message, **kwargs)
        self.tool_name = tool_name
        self.input_data = input_data

class StateValidationError(AgentError):
    """Raised when state has invalid structure."""
    def __init__(self, message, field=None, **kwargs):
        super().__init__(message, **kwargs)
        self.field = field

class MaxStepsExceededError(AgentError):
    """Raised when the agent hits its step limit."""
    pass

# Using custom exceptions in a node
def execute_tool_safely(tool_name, args, tools_dict, step=0, node="tool_node"):
    if tool_name not in tools_dict:
        raise ToolExecutionError(
            f"Tool '{tool_name}' not found in registry",
            tool_name=tool_name,
            input_data=args,
            step=step,
            node=node
        )
    try:
        result = tools_dict[tool_name](**args)
        return result
    except TypeError as e:
        raise ToolExecutionError(
            f"Invalid arguments for tool '{tool_name}': {e}",
            tool_name=tool_name,
            input_data=args,
            step=step,
            node=node
        ) from e   # 'from e' chains exceptions — preserves original traceback

# Handle in the graph runner
def safe_tool_node(state, tools_dict):
    tool_calls = state.get("tool_calls", [])
    results = {}
    errors = []

    for call in tool_calls:
        try:
            result = execute_tool_safely(
                call["name"], call.get("args", {}), tools_dict,
                step=state.get("metadata", {}).get("turn", 0)
            )
            results[call["name"]] = {"success": True, "result": result}
        except ToolExecutionError as e:
            errors.append({"tool": e.tool_name, "error": str(e)})
            results[call["name"]] = {"success": False, "error": str(e)}
        except Exception as e:
            # Catch-all for unexpected errors
            errors.append({"tool": call["name"], "error": f"Unexpected: {e}"})

    return {
        "tool_results": results,
        "error": errors[0]["error"] if errors else None
    }

# Test
tools = {"calculator": lambda expression: str(eval(expression))}
state = {
    "tool_calls": [
        {"name": "calculator", "args": {"expression": "2+2"}},
        {"name": "unknown_tool", "args": {}},
    ],
    "metadata": {"turn": 3}
}
result = safe_tool_node(state, tools)
print(result)
```

---

#### 9.1.3 — Context Managers: the `with` Statement

```python
# Problem: file must be closed even if an error occurs
# WITHOUT context manager (error-prone):
f = open("data.txt", "w")
f.write("hello")
# If an exception happens here, f.close() is never called!
f.close()

# WITH context manager — guaranteed cleanup:
with open("data.txt", "w") as f:
    f.write("hello\nworld\n")
# f is automatically closed here, even if an exception occurs

# Reading files
with open("data.txt", "r") as f:
    content = f.read()
    print(content)

with open("data.txt", "r") as f:
    for line in f:           # iterate line by line (memory-efficient)
        print(line.strip())

# Multiple context managers
with open("input.txt", "r") as src, open("output.txt", "w") as dst:
    for line in src:
        dst.write(line.upper())

# Writing a JSON state checkpoint
import json

def save_state(state, filepath):
    """Save agent state to disk (like LangGraph's checkpointer)."""
    with open(filepath, "w") as f:
        json.dump(state, f, indent=2, default=str)
    print(f"State saved to {filepath}")

def load_state(filepath):
    """Load agent state from disk."""
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {filepath}: {e}")

# Building your own context manager with __enter__/__exit__
class StepTimer:
    """Context manager that times an operation."""

    def __init__(self, step_name):
        self.step_name = step_name
        self.elapsed = None

    def __enter__(self):
        import time
        self._start = time.time()
        print(f"[{self.step_name}] Starting...")
        return self   # the 'as' variable receives this

    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        self.elapsed = time.time() - self._start
        if exc_type:
            print(f"[{self.step_name}] FAILED after {self.elapsed:.3f}s: {exc_val}")
        else:
            print(f"[{self.step_name}] Done in {self.elapsed:.3f}s")
        return False  # False = don't suppress exceptions

# Usage
with StepTimer("llm_call") as timer:
    import time
    time.sleep(0.1)   # simulate work
print(f"Elapsed: {timer.elapsed:.3f}s")

with StepTimer("failing_step") as timer:
    raise ValueError("something went wrong")
# exception still propagates — StepTimer just logs it
```

---

### Lesson 9.2 — Guided Coding Drill (1 hour)

```python
# drill_09_errors.py

import json
import time
import copy

class AgentError(Exception):
    pass

class NodeExecutionError(AgentError):
    def __init__(self, msg, node_name, state_snapshot=None):
        super().__init__(msg)
        self.node_name = node_name
        self.state_snapshot = state_snapshot

class CheckpointManager:
    """Mimics LangGraph's MemorySaver checkpointer."""

    def __init__(self, filepath):
        self.filepath = filepath
        self._checkpoints = []

    def __enter__(self):
        self._load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            print(f"[Checkpoint] Error during session: {exc_val}")
        self._save()
        return False

    def save(self, state, step):
        checkpoint = {
            "step": step,
            "timestamp": time.time(),
            "state": copy.deepcopy(state)
        }
        self._checkpoints.append(checkpoint)
        print(f"[Checkpoint] Saved step {step}")

    def restore_latest(self):
        if not self._checkpoints:
            return None
        return copy.deepcopy(self._checkpoints[-1]["state"])

    def _save(self):
        with open(self.filepath, "w") as f:
            json.dump(self._checkpoints, f, default=str)

    def _load(self):
        try:
            with open(self.filepath, "r") as f:
                self._checkpoints = json.load(f)
            print(f"[Checkpoint] Loaded {len(self._checkpoints)} checkpoints")
        except FileNotFoundError:
            self._checkpoints = []
            print("[Checkpoint] No existing checkpoints found")

# Test the checkpoint manager
state = {
    "messages": [{"role": "user", "content": "Hello"}],
    "step": 0
}

with CheckpointManager("agent_checkpoint.json") as ckpt:
    # Simulate 3 steps
    for i in range(3):
        state["step"] = i
        state["messages"].append({"role": "assistant", "content": f"Step {i} response"})
        ckpt.save(state, i)

    # Restore latest
    restored = ckpt.restore_latest()
    print(f"\nRestored state step: {restored['step']}")
    print(f"Message count: {len(restored['messages'])}")

# Clean up
import os
if os.path.exists("agent_checkpoint.json"):
    os.remove("agent_checkpoint.json")
```

---

### Lesson 9.3 — Homework Challenge (1 hour)

**Challenge: Retry Mechanism**

Build a `retry(func, max_attempts, delay_seconds, catch_exceptions)` function that:

1. Calls `func()` up to `max_attempts` times.
2. If it succeeds (no exception), returns the result immediately.
3. If it raises one of `catch_exceptions`, waits `delay_seconds` and retries.
4. If it raises something NOT in `catch_exceptions`, re-raises immediately.
5. If all attempts fail, raises a custom `MaxRetriesExceeded` exception with the last error as the cause.
6. Prints `[Retry] Attempt {n}/{max} failed: {error}` for each failure.

Test with a function that fails 2 times then succeeds.

LangGraph has a built-in retry policy for nodes — this is its conceptual foundation.

---

## MODULE 10: Capstone — Manual LangChain in Pure Python

**Duration: 3 hours** | **Theme: Build the real thing from scratch to deeply understand it**

---

### Lesson 10.1 — Project Overview (30 min)

**Goal:** Build a fully functional "Manual LangChain" — a sequential agent in pure Python using only the concepts from Modules 1-9. No external libraries (not even `requests`). The output should be a working system that mirrors what LangChain/LangGraph actually does.

**Deliverable files:**

```
manual_langchain/
├── __init__.py
├── state.py          # State TypedDict
├── prompt.py         # PromptTemplate class
├── tools.py          # Tool base class + 3 concrete tools
├── nodes.py          # LLM node, Tool node, Router
├── graph.py          # Graph class (runs nodes in sequence)
└── main.py           # Demo script
```

---

### Lesson 10.2 — Build It (2 hours)

```python
# state.py
from typing import TypedDict, List, Dict, Any, Optional

class AgentState(TypedDict, total=False):
    messages: List[Dict[str, str]]
    tool_calls: List[Dict[str, Any]]
    tool_results: Dict[str, Any]
    is_done: bool
    error: Optional[str]
    metadata: Dict[str, Any]
    final_answer: Optional[str]
```

```python
# prompt.py

class PromptTemplate:
    """
    Mimics LangChain's PromptTemplate.
    Stores a template string and fills it with variables.
    """

    def __init__(self, template, input_variables):
        self.template = template
        self.input_variables = input_variables
        # Validate that all variables in the template are declared
        for var in input_variables:
            placeholder = "{" + var + "}"
            if placeholder not in template:
                raise ValueError(f"Variable '{var}' not found in template")

    def format(self, **kwargs):
        """Fill the template with provided variables."""
        missing = set(self.input_variables) - set(kwargs.keys())
        if missing:
            raise ValueError(f"Missing variables: {missing}")

        result = self.template
        for key, value in kwargs.items():
            result = result.replace("{" + key + "}", str(value))
        return result

    def __or__(self, other):
        """
        The | operator chains components, like LangChain's LCEL.
        (simplified version)
        """
        return Chain([self, other])

    def __repr__(self):
        return f"PromptTemplate(vars={self.input_variables})"


class ChatPromptTemplate:
    """Mimics LangChain's ChatPromptTemplate — returns a list of messages."""

    def __init__(self, messages_templates):
        """
        messages_templates: list of (role, template_string) tuples
        """
        self.message_templates = messages_templates
        self.input_variables = self._extract_variables()

    def _extract_variables(self):
        import re
        variables = set()
        for _, template in self.message_templates:
            found = re.findall(r"\{(\w+)\}", template)
            variables.update(found)
        return list(variables)

    def format_messages(self, **kwargs):
        """Returns a list of message dicts."""
        messages = []
        for role, template in self.message_templates:
            content = template
            for key, value in kwargs.items():
                content = content.replace("{" + key + "}", str(value))
            messages.append({"role": role, "content": content})
        return messages

    def __repr__(self):
        return f"ChatPromptTemplate({len(self.message_templates)} messages)"


class Chain:
    """Simple sequential chain — mimics LCEL pipe operator."""

    def __init__(self, steps):
        self.steps = steps

    def invoke(self, inputs):
        result = inputs
        for step in self.steps:
            if hasattr(step, 'format'):
                result = step.format(**result) if isinstance(result, dict) else step.format(result)
            elif callable(step):
                result = step(result)
            elif hasattr(step, 'invoke'):
                result = step.invoke(result)
        return result
```

```python
# tools.py
import datetime

class ToolError(Exception):
    def __init__(self, msg, tool_name):
        super().__init__(msg)
        self.tool_name = tool_name

class BaseTool:
    name: str = ""
    description: str = ""

    def invoke(self, input_data):
        try:
            if isinstance(input_data, dict):
                return self._run(**input_data)
            return self._run(input_data)
        except ToolError:
            raise
        except Exception as e:
            raise ToolError(str(e), self.name) from e

    def _run(self, *args, **kwargs):
        raise NotImplementedError

    def get_schema(self):
        return {"name": self.name, "description": self.description}

    def __repr__(self):
        return f"{self.__class__.__name__}('{self.name}')"


class EchoTool(BaseTool):
    name = "echo"
    description = "Repeats the input text back to you. Use for testing."

    def _run(self, text):
        return f"Echo: {text}"


class WordCountTool(BaseTool):
    name = "word_count"
    description = "Counts words in a given text."

    def _run(self, text):
        words = text.split()
        unique = set(words)
        return {
            "total": len(words),
            "unique": len(unique),
            "most_common": max(set(words), key=words.count) if words else None
        }


class TimestampTool(BaseTool):
    name = "timestamp"
    description = "Returns the current date and time."

    def _run(self):
        now = datetime.datetime.now()
        return {
            "iso": now.isoformat(),
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
            "unix": int(now.timestamp())
        }


class CalculatorTool(BaseTool):
    name = "calculator"
    description = "Evaluates a safe mathematical expression."

    ALLOWED_CHARS = set("0123456789+-*/()., ")

    def _run(self, expression):
        if not all(c in self.ALLOWED_CHARS for c in str(expression)):
            raise ToolError(f"Invalid characters in expression: {expression}", self.name)
        return str(eval(str(expression)))


def build_tool_registry(*tools):
    """Build a name → tool mapping from tool instances."""
    return {tool.name: tool for tool in tools}
```

```python
# nodes.py
import copy
from state import AgentState
from tools import BaseTool, ToolError

# ---- Simulated LLM ----
class MockLLM:
    """
    Simulates an LLM. In production, this calls OpenAI/Anthropic API.
    Here it uses simple keyword matching to decide tool calls.
    """

    def __init__(self, tools_registry):
        self.tools = tools_registry

    def generate(self, messages):
        """Returns (text_response, tool_calls) tuple."""
        last_user = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"),
            ""
        ).lower()

        # Keyword-based "reasoning" (stands in for actual LLM)
        if "count" in last_user or "words" in last_user:
            text = last_user.replace("count words in ", "").replace("word count", "").strip()
            return ("I'll count the words for you.", [
                {"id": "c1", "name": "word_count", "args": {"text": text or "sample text"}}
            ])
        elif "time" in last_user or "date" in last_user:
            return ("Let me get the current timestamp.", [
                {"id": "c2", "name": "timestamp", "args": {}}
            ])
        elif "calculate" in last_user or any(op in last_user for op in ["+", "-", "*", "/"]):
            import re
            expr = re.search(r"[\d\s\+\-\*\/\(\)\.]+", last_user)
            expression = expr.group(0).strip() if expr else "1+1"
            return ("I'll calculate that.", [
                {"id": "c3", "name": "calculator", "args": {"expression": expression}}
            ])
        elif "echo" in last_user:
            return ("I'll echo that back.", [
                {"id": "c4", "name": "echo", "args": {"text": last_user.replace("echo ", "")}}
            ])
        else:
            return (f"I understand you want: '{last_user}'. I've analyzed your request.", [])


# ---- Nodes ----

def llm_node(state: AgentState, llm: MockLLM) -> dict:
    """
    Calls the LLM with current messages.
    Returns partial state update.
    """
    messages = state.get("messages", [])
    text, tool_calls = llm.generate(messages)

    new_messages = [{"role": "assistant", "content": text}]
    is_done = len(tool_calls) == 0   # done if no tool calls

    return {
        "messages": new_messages,
        "tool_calls": tool_calls,
        "is_done": is_done,
        "final_answer": text if is_done else None
    }


def tool_node(state: AgentState, tools_registry: dict) -> dict:
    """
    Executes all pending tool calls.
    Returns partial state update.
    """
    tool_calls = state.get("tool_calls", [])
    results = {}
    tool_messages = []
    errors = []

    for call in tool_calls:
        name = call["name"]
        args = call.get("args", {})
        call_id = call.get("id", name)

        if name not in tools_registry:
            errors.append(f"Unknown tool: {name}")
            results[call_id] = {"error": f"Tool '{name}' not found"}
            continue

        try:
            tool = tools_registry[name]
            result = tool.invoke(args)
            results[call_id] = {"success": True, "result": result}
            tool_messages.append({
                "role": "tool",
                "content": str(result),
                "tool_call_id": call_id
            })
        except ToolError as e:
            errors.append(str(e))
            results[call_id] = {"success": False, "error": str(e)}

    return {
        "messages": tool_messages,
        "tool_results": results,
        "tool_calls": [],   # clear pending calls
        "error": errors[0] if errors else None
    }


def router(state: AgentState) -> str:
    """
    Conditional edge function.
    Returns the name of the next node to execute.
    """
    if state.get("error"):
        return "END"
    if state.get("is_done"):
        return "END"
    if state.get("tool_calls"):
        return "tool_node"
    return "llm_node"
```

```python
# graph.py
import copy
from state import AgentState

class GraphExecutionError(Exception):
    pass

class Graph:
    """
    Mimics LangGraph's StateGraph.
    Manages nodes, edges, and state flow.
    """

    END = "END"

    def __init__(self):
        self._nodes = {}        # name → (function, kwargs)
        self._entry_point = None
        self._max_steps = 20
        self._history = []      # checkpoints

    def add_node(self, name, func, **kwargs):
        """Register a node function. kwargs are bound to the function."""
        self._nodes[name] = (func, kwargs)
        return self

    def set_entry_point(self, name):
        if name not in self._nodes:
            raise ValueError(f"Node '{name}' not registered")
        self._entry_point = name
        return self

    def set_max_steps(self, n):
        self._max_steps = n
        return self

    def _apply_update(self, state, update):
        """Merge a partial update into the state."""
        new_state = copy.deepcopy(state)
        for key, value in update.items():
            if key == "messages" and isinstance(value, list):
                new_state.setdefault("messages", [])
                new_state["messages"] = new_state["messages"] + value
            elif key == "tool_results" and isinstance(value, dict):
                new_state.setdefault("tool_results", {})
                new_state["tool_results"].update(value)
            else:
                new_state[key] = value
        return new_state

    def invoke(self, initial_state: AgentState, router_func) -> AgentState:
        """
        Run the graph until END or max_steps.
        router_func determines the next node given current state.
        """
        if not self._entry_point:
            raise GraphExecutionError("No entry point set. Call set_entry_point() first.")

        state = copy.deepcopy(initial_state)
        current_node = self._entry_point
        step = 0

        print(f"\n{'='*50}")
        print(f"GRAPH STARTING")
        print(f"{'='*50}")

        while step < self._max_steps:
            # Save checkpoint
            self._history.append({
                "step": step,
                "node": current_node,
                "state": copy.deepcopy(state)
            })

            print(f"\n[Step {step}] Node: {current_node}")

            # Execute node
            if current_node not in self._nodes:
                raise GraphExecutionError(f"Unknown node: {current_node}")

            func, bound_kwargs = self._nodes[current_node]
            try:
                update = func(state, **bound_kwargs)
                state = self._apply_update(state, update)
            except Exception as e:
                print(f"  ERROR in {current_node}: {e}")
                state["error"] = str(e)
                break

            # Route to next node
            next_node = router_func(state)
            print(f"  → Routing to: {next_node}")

            if next_node == self.END:
                print(f"\n{'='*50}")
                print(f"GRAPH COMPLETE (steps={step+1})")
                print(f"{'='*50}")
                break

            current_node = next_node
            step += 1
        else:
            print(f"WARNING: Hit max steps ({self._max_steps})")

        return state

    @property
    def history(self):
        return list(self._history)
```

```python
# main.py
import json
from state import AgentState
from prompt import ChatPromptTemplate
from tools import EchoTool, WordCountTool, TimestampTool, CalculatorTool, build_tool_registry
from nodes import MockLLM, llm_node, tool_node, router
from graph import Graph

def main():
    print("=== Manual LangChain Demo ===\n")

    # 1. Set up tools
    tools_registry = build_tool_registry(
        EchoTool(),
        WordCountTool(),
        TimestampTool(),
        CalculatorTool()
    )
    print("Tools registered:", list(tools_registry.keys()))

    # 2. Set up prompt template
    system_template = ChatPromptTemplate([
        ("system", "You are a helpful assistant with access to these tools: {tool_names}."),
        ("user", "{user_message}")
    ])
    system_messages = system_template.format_messages(
        tool_names=", ".join(tools_registry.keys()),
        user_message="count words in hello world this is a test"
    )
    print("\nFormatted messages:")
    for m in system_messages:
        print(f"  [{m['role']}]: {m['content']}")

    # 3. Set up mock LLM
    llm = MockLLM(tools_registry)

    # 4. Build the graph
    graph = Graph()
    graph.add_node("llm_node", llm_node, llm=llm)
    graph.add_node("tool_node", tool_node, tools_registry=tools_registry)
    graph.set_entry_point("llm_node")
    graph.set_max_steps(10)

    # 5. Create initial state from our prompt template
    initial_state: AgentState = {
        "messages": system_messages,
        "tool_calls": [],
        "is_done": False,
        "metadata": {"session": "demo-001"}
    }

    # 6. Run the graph!
    final_state = graph.invoke(initial_state, router)

    # 7. Display results
    print("\n=== FINAL STATE ===")
    print(f"Messages ({len(final_state.get('messages', []))} total):")
    for m in final_state.get("messages", []):
        role = m["role"].upper()
        content = str(m["content"])[:80]
        print(f"  [{role}]: {content}")

    print(f"\nTool Results:")
    for call_id, result in final_state.get("tool_results", {}).items():
        print(f"  {call_id}: {result}")

    print(f"\nFinal Answer: {final_state.get('final_answer', 'N/A')}")
    print(f"Error: {final_state.get('error', 'None')}")
    print(f"Is Done: {final_state.get('is_done', False)}")

    # 8. Run a few more queries
    queries = [
        "what is the current time and date",
        "calculate 15 * 7 + 33",
        "echo this is a test message"
    ]
    for query in queries:
        print(f"\n{'─'*40}")
        print(f"Query: {query}")
        state: AgentState = {
            "messages": [{"role": "user", "content": query}],
            "tool_calls": [],
            "is_done": False
        }
        result = graph.invoke(state, router)
        print(f"Answer: {result.get('final_answer') or result.get('tool_results')}")


if __name__ == "__main__":
    main()
```

---

### Lesson 10.3 — Capstone Extension (30 min)

**Extend your Manual LangChain with:**

1. Add a `ConversationMemory` class that persists messages to a JSON file and loads them on startup.
2. Add a `StringOutputParser` class with a `parse(text)` method that strips whitespace and removes common LLM artifacts (like "Answer: " prefixes).
3. Wire the `ChatPromptTemplate | MockLLM | StringOutputParser` into a chain using your `Chain` class from `prompt.py`.
4. Add proper logging using Python's `logging` module instead of `print()`.

This is your last step before moving to the real LangChain library.

---

---

# PART 2: COMPANION STUDY GUIDE

---

## Environment Setup

### Step 1: Install pyenv (Python Version Manager)

```bash
# macOS (via Homebrew)
brew install pyenv
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init -)"' >> ~/.zshrc
source ~/.zshrc

# Linux
curl https://pyenv.run | bash
# Add to .bashrc or .zshrc (same lines as above)

# Windows: use pyenv-win
pip install pyenv-win --target $HOME\\.pyenv
```

```bash
# Install Python 3.11 (recommended — stable, fast, wide LangChain support)
pyenv install 3.11.9
pyenv global 3.11.9
python --version   # Python 3.11.9
```

### Step 2: Install uv (Fast Package Manager)

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Verify
uv --version
```

### Step 3: Create Your First Project

```bash
# Create project directory
mkdir python-for-langchain
cd python-for-langchain

# Initialize with uv
uv init
# Creates: pyproject.toml, .python-version, main.py

# Set Python version
uv python pin 3.11

# Create and activate virtual environment
uv venv
source .venv/bin/activate    # macOS/Linux
.venv\Scripts\activate       # Windows

# Add packages (when ready for LangChain)
uv add langchain langchain-openai langgraph
```

### Step 4: VSCode Setup

Install these extensions:

1. **Python** (Microsoft) — language support, debugger, IntelliSense
2. **Pylance** — fast type checking
3. **Ruff** — lightning-fast linting/formatting
4. **Python Indent** — auto-correct indentation

**settings.json** (`.vscode/settings.json`):

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
  "editor.formatOnSave": true,
  "[python]": {
    "editor.defaultFormatter": "charliermarsh.ruff"
  },
  "python.linting.enabled": true,
  "python.analysis.typeCheckingMode": "basic"
}
```

---

## Quick Reference Cards

### Card 1: List Operations

```
┌─────────────────────────────────────────────────────────┐
│                    LIST QUICK REFERENCE                   │
├─────────────────────────────────────────────────────────┤
│ CREATE     │ lst = []  /  lst = [1,2,3]                 │
│ ACCESS     │ lst[0]  lst[-1]  lst[1:3]  lst[::2]        │
│ MUTATE     │ .append(x)   .extend(iter)   .insert(i, x) │
│            │ .pop()  .pop(i)  del lst[i]                 │
│ SEARCH     │ x in lst    lst.index(x)    lst.count(x)   │
│ SORT       │ lst.sort()  sorted(lst)                     │
│            │ sorted(lst, key=lambda x: x[1])             │
│ SIZE       │ len(lst)  sum(lst)  min(lst)  max(lst)      │
│ COMPREHEND │ [x*2 for x in lst if x > 0]                │
│ ITERATE    │ for i, v in enumerate(lst):                 │
│ ZIP        │ for a, b in zip(lst1, lst2):               │
│ COPY       │ lst.copy()  (shallow)                       │
│            │ copy.deepcopy(lst)  (deep — use this!)      │
└─────────────────────────────────────────────────────────┘
```

### Card 2: Dict Operations

```
┌─────────────────────────────────────────────────────────┐
│                    DICT QUICK REFERENCE                   │
├─────────────────────────────────────────────────────────┤
│ CREATE     │ d = {}   d = {"k": "v"}   dict(k="v")      │
│ ACCESS     │ d["key"]  (KeyError if missing)             │
│            │ d.get("key", default)  (SAFE — use this!)  │
│ MUTATE     │ d["key"] = val   del d["key"]              │
│            │ d.pop("key", None)                          │
│            │ d.update(other_dict)                        │
│ MERGE      │ merged = {**d1, **d2}  (d2 wins conflicts) │
│ ITERATE    │ for k in d:                                 │
│            │ for k, v in d.items():                      │
│            │ for v in d.values():                        │
│ CHECK      │ "key" in d   (True/False — checks keys)     │
│ DEFAULTS   │ d.setdefault("key", default_val)            │
│ COMPREHEND │ {k: v*2 for k, v in d.items() if v > 0}    │
│ NEST       │ d["meta"]["turn"] += 1                      │
│ SAFE NEST  │ d.get("meta", {}).get("turn", 0)           │
│ COPY       │ d.copy()  (shallow — careful with nesting!) │
│            │ copy.deepcopy(d)  (always safe)             │
└─────────────────────────────────────────────────────────┘
```

### Card 3: String Operations

```
┌─────────────────────────────────────────────────────────┐
│                   STRING QUICK REFERENCE                  │
├─────────────────────────────────────────────────────────┤
│ FORMAT     │ f"Hello, {name}!"                           │
│            │ f"Pi: {3.14159:.2f}"  f"Pct: {0.8:.1%}"   │
│ CLEAN      │ s.strip()  .lstrip()  .rstrip()             │
│ CASE       │ s.lower()  .upper()  .title()               │
│ SEARCH     │ "sub" in s      (bool — use this!)          │
│            │ s.find("sub")   (index or -1)               │
│            │ s.startswith("x")   .endswith("y")          │
│ SPLIT/JOIN │ s.split(" ")    ", ".join(lst)              │
│ REPLACE    │ s.replace("old", "new")                     │
│ SLICE      │ s[0:5]   s[-3:]   s[::-1]                  │
│ SIZE       │ len(s)   s.count("x")                       │
│ MULTILINE  │ """triple                                   │
│            │    quoted"""                                 │
│ IMMUTABLE  │ s[0] = "X"  → TypeError! Create new string │
└─────────────────────────────────────────────────────────┘
```

### Card 4: Control Flow

```
┌─────────────────────────────────────────────────────────┐
│                 CONTROL FLOW QUICK REFERENCE             │
├─────────────────────────────────────────────────────────┤
│ IF         │ if cond:                                    │
│            │ elif cond:                                  │
│            │ else:                                       │
│ TERNARY    │ x = "yes" if cond else "no"                │
│ FALSY      │ False, None, 0, 0.0, "", [], {}, set()     │
│ FOR        │ for item in iterable:                       │
│ RANGE      │ range(5)    range(1,6)   range(0,10,2)     │
│ ENUMERATE  │ for i, v in enumerate(lst, start=0):        │
│ ZIP        │ for a, b in zip(lst1, lst2):               │
│ WHILE      │ while condition:                            │
│ BREAK      │ break   (exit loop immediately)             │
│ CONTINUE   │ continue   (skip to next iteration)         │
│ FOR/ELSE   │ for x in lst:                               │
│            │   if found: break                           │
│            │ else:                                       │
│            │   # runs if NO break happened               │
│ SHORT-CIR  │ a and b   (b skipped if a is False)        │
│            │ a or b    (b skipped if a is True)         │
└─────────────────────────────────────────────────────────┘
```

### Card 5: Functions

```
┌─────────────────────────────────────────────────────────┐
│                  FUNCTIONS QUICK REFERENCE               │
├─────────────────────────────────────────────────────────┤
│ DEFINE     │ def func(arg1, arg2="default"):             │
│ RETURN     │     return value   (None if omitted)        │
│ CALL       │ func(val1, arg2="override")                 │
│ *ARGS      │ def f(*args):  → args is a tuple            │
│ **KWARGS   │ def f(**kw):   → kw is a dict              │
│ UNPACK     │ f(*my_list)    f(**my_dict)                 │
│ LAMBDA     │ fn = lambda x, y: x + y                    │
│ FIRST-CLASS│ fn = other_fn   (no parentheses!)           │
│            │ [fn for fn in list_of_fns]                  │
│ SCOPE      │ L → E → G → B (Local/Enclosing/Global/Builtin) │
│ GLOBAL     │ global var_name  (modify global)            │
│ NONLOCAL   │ nonlocal var_name  (modify enclosing)       │
│ DEFAULT BUG│ def f(lst=[]):  ← NEVER! Use lst=None      │
│ DOCSTRING  │ def f(): """Does X."""                      │
└─────────────────────────────────────────────────────────┘
```

### Card 6: OOP

```
┌─────────────────────────────────────────────────────────┐
│                     OOP QUICK REFERENCE                   │
├─────────────────────────────────────────────────────────┤
│ CLASS      │ class Foo:                                  │
│ INIT       │     def __init__(self, x):                  │
│            │         self.x = x                          │
│ INSTANCE   │ obj = Foo(42)   obj.x                      │
│ METHOD     │     def method(self): return self.x         │
│ CLASSMETHOD│     @classmethod                            │
│            │     def from_dict(cls, d):                  │
│            │         return cls(d["x"])                   │
│ STATICMETHOD│    @staticmethod                           │
│            │     def validate(x): return x > 0           │
│ INHERIT    │ class Bar(Foo):                             │
│            │     def __init__(self, x, y):               │
│            │         super().__init__(x)                  │
│            │         self.y = y                           │
│ OVERRIDE   │     def method(self):                       │
│            │         base = super().method()              │
│            │         return base + self.y                 │
│ PROPERTY   │     @property                               │
│            │     def val(self): return self._val         │
│            │     @val.setter                              │
│            │     def val(self, v): self._val = v         │
│ ISINSTANCE │ isinstance(obj, Foo)   (True if Foo or subclass)│
│ REPR       │ def __repr__(self): return f"Foo({self.x})" │
└─────────────────────────────────────────────────────────┘
```

### Card 7: Error Handling

```
┌─────────────────────────────────────────────────────────┐
│                ERROR HANDLING QUICK REFERENCE            │
├─────────────────────────────────────────────────────────┤
│ BASIC      │ try:                                        │
│            │     risky_code()                            │
│            │ except ValueError:                          │
│            │     handle_it()                             │
│ MULTI EXCEPT│ except (TypeError, ValueError) as e:      │
│ ELSE       │ else:    # runs if NO exception raised      │
│ FINALLY    │ finally:  # ALWAYS runs                     │
│ RERAISE    │ raise   (reraise caught exception)          │
│ CHAIN      │ raise NewError("msg") from original_err    │
│ CUSTOM     │ class MyError(Exception):                   │
│            │     def __init__(self, msg, code):          │
│            │         super().__init__(msg)               │
│            │         self.code = code                    │
│ CATCH-ALL  │ except Exception as e:   (not bare except!) │
│ INFO       │ type(e).__name__   str(e)   e.args          │
└─────────────────────────────────────────────────────────┘
```

### Card 8: Context Managers & Files

```
┌─────────────────────────────────────────────────────────┐
│           CONTEXT MANAGERS & FILES QUICK REFERENCE       │
├─────────────────────────────────────────────────────────┤
│ READ FILE  │ with open("f.txt") as f:                   │
│            │     content = f.read()                      │
│ WRITE FILE │ with open("f.txt", "w") as f:              │
│            │     f.write("text")                         │
│ APPEND     │ with open("f.txt", "a") as f:              │
│            │     f.write("more")                         │
│ LINES      │ for line in f:   (memory efficient)         │
│ ENCODING   │ open("f.txt", encoding="utf-8")            │
│ JSON READ  │ json.load(f)   (from file handle)           │
│ JSON WRITE │ json.dump(data, f, indent=2)               │
│ JSON STR   │ json.loads(str)   json.dumps(data)          │
│ MULTI CTX  │ with open("a") as f1, open("b") as f2:    │
│ CUSTOM CTX │ class Ctx:                                  │
│            │     def __enter__(self): return self        │
│            │     def __exit__(self, et, ev, tb):         │
│            │         cleanup()                           │
│            │         return False  # don't suppress err  │
└─────────────────────────────────────────────────────────┘
```

---

## LangChain Translation Dictionary

| Python Concept              | LangChain / LangGraph Equivalent | Notes                                            |
| --------------------------- | -------------------------------- | ------------------------------------------------ |
| `dict`                      | `State` (TypedDict)              | LangGraph state is always a dict                 |
| `dict.get(key, default)`    | Safe state field access          | Always use `.get()` in nodes                     |
| `{**d1, **d2}`              | State reducer (default)          | LangGraph merges partial updates this way        |
| `list.append()`             | `add_messages` reducer           | LangGraph's built-in message reducer             |
| `copy.deepcopy()`           | State immutability               | LangGraph copies state between steps             |
| `def node(state) → dict`    | `@node` / graph node             | Nodes take state, return partial update          |
| `def router(state) → str`   | Conditional edge                 | Returns node name or `"END"`                     |
| `class BaseTool`            | `langchain.tools.BaseTool`       | Exact same pattern, real class                   |
| `tool._run(input)`          | `BaseTool._run()`                | The method you override in subclasses            |
| `try/except` in node        | Node error handling              | Caught errors update state["error"]              |
| `lambda x: x["role"]`       | Key functions in sorting         | Used everywhere in LangChain chains              |
| `for item in list`          | Chain iteration                  | LCEL pipes iterate over messages                 |
| `isinstance(x, SomeClass)`  | Type checking in tools           | LangChain does this for input validation         |
| `@classmethod from_dict`    | `model_validate()` (Pydantic)    | Pydantic is the actual tool; this is the concept |
| `@property`                 | Pydantic `@validator`            | Property setters validate data                   |
| `if __name__ == "__main__"` | Entry point scripts              | Same pattern in LangGraph CLI tools              |
| `json.dumps / loads`        | Serializing checkpoints          | MemorySaver serializes state to JSON             |
| `with open() as f`          | Checkpointer context managers    | SqliteSaver uses context managers                |
| `raise CustomError from e`  | Node exception chaining          | Error propagation in graphs                      |
| `*args, **kwargs`           | Tool `.invoke(**kwargs)`         | Tools accept flexible arguments                  |
| `module.function`           | `langchain.tool import Tool`     | Same import pattern                              |
| `set()`                     | Tool name deduplication          | Used internally to check registered tools        |
| `any(cond for x in lst)`    | Agent should-stop checks         | e.g., `any(m.type == "tool" for m in msgs)`      |
| `enumerate(messages)`       | Message indexing                 | Common when building context windows             |
| `zip(roles, contents)`      | Building message lists           | Creating chat history from parallel lists        |
| `dict comprehension`        | State transformation             | Filtering/transforming state fields              |
| `sorted(items, key=fn)`     | Tool priority ordering           | Sort tools by relevance score                    |
| `TOOL_REGISTRY = {}`        | `ToolNode(tools=[...])`          | The dict-of-tools pattern is identical           |
| `Pipeline` class            | `SequentialChain` / LCEL pipe    | Chain of functions = Chain of Runnables          |
| `PromptTemplate.format()`   | `PromptTemplate.invoke()`        | Same concept, Pydantic-validated in LangChain    |

---

## Common Pitfalls

### Pitfall 1: Mutable Default Arguments in Tool Functions

```python
# WRONG — the list persists between calls!
def collect_results(item, results=[]):
    results.append(item)
    return results

print(collect_results("a"))   # ['a']
print(collect_results("b"))   # ['a', 'b'] — BUG!

# CORRECT
def collect_results(item, results=None):
    if results is None:
        results = []
    results.append(item)
    return results
```

**LangGraph impact:** If you pass a mutable default to a node function that's called many times, state bleeds between invocations.

---

### Pitfall 2: Shallow Copying State Dicts

```python
# WRONG
import copy
state = {"messages": [{"role": "user", "content": "Hello"}]}
new_state = state.copy()           # shallow!
new_state["messages"].append({"role": "assistant", "content": "Hi"})
print(state["messages"])           # MODIFIED! Both point to same list

# CORRECT
new_state = copy.deepcopy(state)   # fully independent copy
```

---

### Pitfall 3: Forgetting that `in` Checks Dict KEYS, not Values

```python
config = {"model": "gpt-4o", "temperature": 0.7}
print("gpt-4o" in config)          # False! checks keys
print("gpt-4o" in config.values()) # True
print("model" in config)           # True — this is what you want
```

**LangGraph impact:** `"messages" in state` correctly checks if the key exists. Common source of confusion.

---

### Pitfall 4: Modifying a Dict While Iterating It

```python
# WRONG
d = {"a": 1, "b": 2, "c": 3}
for key in d:
    if d[key] > 1:
        del d[key]    # RuntimeError: dictionary changed size during iteration

# CORRECT: iterate over a copy of keys
for key in list(d.keys()):
    if d[key] > 1:
        del d[key]
```

---

### Pitfall 5: Using `is` Instead of `==` for Value Comparison

```python
x = 1000
y = 1000
print(x == y)   # True  — value equality (use this!)
print(x is y)   # False — identity check (same object)

# The exception: None, True, False
if x is None:   # correct!
    pass
if x == None:   # works but bad style — use `is`
    pass
```

---

### Pitfall 6: Forgetting `self` in Class Methods

```python
class Tool:
    def __init__(self):
        self.name = "my_tool"

    def get_name():     # WRONG — no self!
        return self.name

t = Tool()
t.get_name()   # TypeError: Tool.get_name() takes 0 positional arguments but 1 was given
```

---

### Pitfall 7: Catching `Exception` Too Broadly (or Not at All)

```python
# Too broad — hides bugs
try:
    result = tool.invoke(input)
except Exception:
    pass    # silently ignores everything including programming errors!

# Better: catch specific exceptions and re-raise unexpected ones
try:
    result = tool.invoke(input)
except ToolExecutionError as e:
    # handle known tool errors
    state["error"] = str(e)
except Exception as e:
    # log unexpected errors and re-raise
    logger.error(f"Unexpected error in tool: {e}")
    raise
```

---

### Pitfall 8: Mutating State Inside a Node Instead of Returning Updates

```python
# WRONG — mutating state directly
def bad_node(state):
    state["messages"].append({"role": "assistant", "content": "Hi"})  # mutation!
    state["step"] += 1
    # returning nothing means the graph gets None as the update

# CORRECT — return a new partial state
def good_node(state):
    new_messages = [{"role": "assistant", "content": "Hi"}]
    return {
        "messages": new_messages,  # LangGraph's reducer will merge these
        "step": state.get("step", 0) + 1
    }
```

---

### Pitfall 9: Using `print()` Instead of Logging

```python
# WRONG — hard to control verbosity, can't redirect
def process_node(state):
    print("DEBUG: entering node")
    print(f"DEBUG: state keys: {list(state.keys())}")

# CORRECT — use logging
import logging
logger = logging.getLogger(__name__)

def process_node(state):
    logger.debug("Entering node")
    logger.debug(f"State keys: {list(state.keys())}")
```

---

### Pitfall 10: Integer Truthiness vs None Checks

```python
count = 0

# WRONG — 0 is falsy!
if not count:
    print("No items")   # prints even when count=0 (valid!)

# CORRECT
if count is None:
    print("Count not set")
elif count == 0:
    print("Zero items")

# In LangGraph: always be explicit about None vs 0 vs empty
step = state.get("step")   # might be 0 (falsy!) or None
if step is None:
    step = 0   # unset
# Now step = 0 is intentional, not "falsy = unset"
```

---

## Practice Problem Bank

### Category A: Strings (Problems 1–10)

**A1.** Write a function `extract_json_from_text(text)` that finds and returns the first valid JSON object (surrounded by `{}`) in a string. Return `None` if not found. Test with: `"Here is the result: {\"key\": \"value\"} and some trailing text."`.

**A2.** Write `count_token_estimate(messages)` that takes a list of `{"role": str, "content": str}` dicts and returns the total estimated token count (chars // 4).

**A3.** Write `truncate_to_token_limit(text, max_tokens)` that truncates a string to approximately `max_tokens` tokens by cutting at the last word boundary before the limit.

**A4.** Write `format_tool_result(tool_name, result, success)` that returns a formatted string like: `"[TOOL: web_search] ✓ Result: ..."` or `"[TOOL: web_search] ✗ Error: ..."`.

**A5.** Write `extract_tool_calls(text)` that parses text like `"Action: web_search\nAction Input: Python tutorials"` and returns `{"action": "web_search", "action_input": "Python tutorials"}`.

**A6.** Write `clean_llm_output(text)` that removes common LLM artifacts: strips whitespace, removes `"Answer: "`, `"Response: "`, and `"Final: "` prefixes.

**A7.** Write `validate_template(template, required_vars)` that checks if all required vars appear as `{var_name}` in the template. Returns `(True, [])` or `(False, [missing_vars])`.

**A8.** Write `truncate_message(msg, max_chars=200)` that truncates message content to `max_chars` and adds `"..."` if truncated.

**A9.** Write `camel_to_snake(name)` that converts `"WebSearchTool"` to `"web_search_tool"`.

**A10.** Write `build_system_prompt(agent_role, tools, constraints)` that formats a multi-line system prompt string using f-strings and `"\n".join()`.

---

### Category B: Lists & Tuples (Problems 11–20)

**B11.** Write `sliding_window(messages, window_size)` that returns the last `window_size` messages, always keeping any system messages.

**B12.** Write `deduplicate_messages(messages)` that removes duplicate messages (by content) while preserving order.

**B13.** Write `group_by_role(messages)` that returns a dict `{role: [messages]}`.

**B14.** Write `flatten_tool_results(results_list)` where `results_list` is a list of lists of strings. Return a flat list of all strings.

**B15.** Write `zip_to_messages(roles, contents)` that creates message dicts from two parallel lists, raising `ValueError` if lengths don't match.

**B16.** Write `get_last_n_by_role(messages, role, n)` that returns the last `n` messages matching the given role.

**B17.** Write `messages_to_text(messages)` that formats a message list as `"role: content\n"` for each message.

**B18.** Write `rotate_system_message(messages, new_system)` that replaces the first system message with `new_system`, or prepends one if none exists.

**B19.** Write `interleave(list1, list2)` that produces `[l1[0], l2[0], l1[1], l2[1], ...]`. Handle unequal lengths gracefully.

**B20.** Write `partition(lst, predicate)` that returns `(matching, non_matching)` tuple of two lists.

---

### Category C: Dictionaries (Problems 21–30)

**C21.** Write `deep_get(d, *keys, default=None)` that safely retrieves nested keys: `deep_get(state, "metadata", "usage", "tokens", default=0)`.

**C22.** Write `deep_set(d, value, *keys)` that sets a nested key, creating intermediate dicts as needed.

**C23.** Write `flatten_dict(d, separator=".")` that converts `{"a": {"b": {"c": 1}}}` to `{"a.b.c": 1}`.

**C24.** Write `unflatten_dict(d, separator=".")` — the inverse of `flatten_dict`.

**C25.** Write `diff_states(old_state, new_state)` that returns a dict of only the keys that changed between two states.

**C26.** Write `merge_configs(*configs)` that merges multiple dicts, with later configs winning on conflict.

**C27.** Write `pick(d, *keys)` that returns a new dict with only the specified keys.

**C28.** Write `omit(d, *keys)` that returns a new dict without the specified keys.

**C29.** Write `count_by(items, key_func)` that takes a list and groups by key, returning `{key: count}`.

**C30.** Write `invert_dict(d)` that returns `{v: k for k, v in d.items()}`, raising `ValueError` on duplicate values.

---

### Category D: Functions & OOP (Problems 31–40)

**D31.** Write `memoize(func)` using a closure and dict — cache function results by arguments (without decorators, using a manual wrapper call).

**D32.** Write `compose(*funcs)` that returns a new function applying funcs right-to-left: `compose(f, g)(x)` = `f(g(x))`.

**D33.** Write `partial_apply(func, **kwargs)` that returns a new function with some kwargs pre-filled.

**D34.** Write a `Registry` class with `register(name, obj)`, `get(name)`, `list_names()`, and `__contains__(name)` methods.

**D35.** Write a `Validator` class with `add_rule(name, check_func, error_msg)` and `validate(data)` that runs all rules and returns a list of errors.

**D36.** Write a `Node` base class and 3 concrete subclasses (`InputNode`, `ProcessNode`, `OutputNode`) each with `execute(state) → dict` method.

**D37.** Write `safe_call(func, *args, default=None, **kwargs)` that calls `func(*args, **kwargs)` and returns `default` on any exception.

**D38.** Write a `TypedRegistry` class that only allows registering objects of a specific type (pass the type to `__init__`).

**D39.** Write `apply_pipeline(value, *funcs)` that applies a sequence of functions to a value, returning the final result and a list of intermediate values.

**D40.** Write a `Singleton` base class so that `class Config(Singleton): pass` always returns the same instance.

---

### Category E: Error Handling & Context Managers (Problems 41–50)

**E41.** Write a `RetryContext` context manager that retries the wrapped block up to `n` times on `Exception`.

**E42.** Write `safe_json_parse(text, default=None)` that returns parsed JSON or `default` on any parse error.

**E43.** Write a `Timer` context manager that stores elapsed time in `timer.elapsed` after the block exits.

**E44.** Write `require(condition, message, exc_type=ValueError)` — a one-liner assertion helper that raises the given exception type.

**E45.** Write a custom exception hierarchy: `AgentError → NodeError, ToolError, StateError` with a `to_dict()` method on each.

**E46.** Write `try_or_default(func, default, *catch)` — calls `func()` and returns `default` if any of `catch` exceptions are raised.

**E47.** Write a `Checkpoint` context manager that saves state to JSON on enter and on exit (even if an exception occurs).

**E48.** Write `validate_state(state, required_keys, type_checks)` that raises `StateError` for missing keys or wrong types.

**E49.** Write `with_timeout(func, seconds, *args, **kwargs)` that runs `func` and raises `TimeoutError` if it exceeds `seconds`. (Use threading for this.)

**E50.** Write a `Tracer` class that can be used as a context manager around each node call to log `{node_name, start_time, end_time, success, error}` to a list of traces.

---

## Bridge Topics Appendix

The following topics are **explicitly excluded** from this course because they require the foundation built here first. Each is important for LangChain/LangGraph — but will make more sense after you've used the library.

---

### Bridge Topic 1: Decorators

**What they are:** Functions that wrap other functions to modify behavior.
**Why excluded:** Requires deep understanding of closures, first-class functions, and function metadata (`__name__`, `functools.wraps`). These are intermediate concepts.
**Where you'll see them in LangChain:** `@tool`, `@chain`, `@retry`, `@property` (which we DID cover as a built-in).
**When to learn:** After completing this course and your first LangChain project.

---

### Bridge Topic 2: Generators and Iterators

**What they are:** Functions with `yield` that produce values lazily.
**Why excluded:** Requires understanding Python's iterator protocol (`__iter__`, `__next__`) which is firmly intermediate.
**Where you'll see them in LangChain:** `chain.stream()` returns a generator. LangChain's streaming output uses generators heavily.
**When to learn:** Immediately before using `chain.stream()`.

---

### Bridge Topic 3: Async/Await (asyncio)

**What it is:** Python's cooperative multitasking system for I/O-bound operations.
**Why excluded:** Deserves its own 5-hour module. Async code has different debugging patterns, different error handling, and a separate runtime model. Mixing sync and async incorrectly crashes programs.
**Where you'll see it in LangChain:** `await chain.ainvoke()`, `await tool.arun()`, `async for chunk in chain.astream()`.
**When to learn:** After this course. Study event loops, `asyncio.run()`, `async def`, and `await` before touching LangChain's async API.

---

### Bridge Topic 4: Pydantic

**What it is:** A data validation library that uses Python type hints to validate and serialize data.
**Why excluded:** Pydantic introduces its own model system (`BaseModel`, `Field`, `validator`) that sits on top of Python OOP. It's easier to understand once you've built your own validation from scratch (Module 7 homework).
**Where you'll see it in LangChain:** EVERYWHERE. `BaseModel` is the foundation of `BaseTool`, `BaseChatModel`, message types, and tool schemas.
**When to learn:** Immediately after this course, before touching any LangChain code. The Pydantic docs are excellent.

---

### Bridge Topic 5: Type Hints (full system)

**What they are:** Python's optional static typing annotation system (`List[str]`, `Optional[Dict[str, Any]]`, `Union`, `TypeVar`, `Generic`).
**Why partially excluded:** We used `TypedDict` in Module 8, which is the most relevant piece. The full type system (generics, protocols, TypeVar) is intermediate.
**Where you'll see them in LangChain:** LangChain's source is heavily typed. `Runnable[Input, Output]` uses generics.
**When to learn:** Learn `List`, `Dict`, `Optional`, `Union`, `Any` now. Learn `TypeVar`, `Generic`, `Protocol` when reading LangChain source.

---

---

# PART 3: PREREQUISITE ROADMAP

```
╔══════════════════════════════════════════════════════════════════╗
║                    YOUR LEARNING ROADMAP                         ║
║              From Zero Python to LangGraph Source Code           ║
╚══════════════════════════════════════════════════════════════════╝

PHASE 1: THIS COURSE (30 hours)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓  Module 1:  Python Object Model & Syntax
✓  Module 2:  Strings, Numbers, I/O
✓  Module 3:  Lists & Tuples
✓  Module 4:  Dictionaries (LangGraph State)
✓  Module 5:  Sets & Control Flow
✓  Module 6:  Functions & Lambdas
✓  Module 7:  OOP — Classes & Inheritance
✓  Module 8:  Modules & Project Structure
✓  Module 9:  Error Handling & Context Managers
✓  Module 10: Capstone — Manual LangChain

🎯 CHECKPOINT: Can you build the Capstone project from scratch?
   If yes: proceed. If no: redo the modules you're weak on.

PHASE 2: INTERMEDIATE PYTHON (15 hours — separate course)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
□  Pydantic v2 (5 hrs)
   → BaseModel, Field, validator, model_validator
   → TypedDict vs BaseModel
   → JSON serialization with Pydantic

□  Decorators (3 hrs)
   → Function decorators, functools.wraps
   → Class decorators
   → Stacking decorators

□  Generators & Iterators (3 hrs)
   → yield, next(), iter()
   → Generator expressions
   → Streaming with generators

□  Async/Await Fundamentals (4 hrs)
   → asyncio event loop
   → async def, await
   → asyncio.gather, asyncio.create_task
   → Mixing sync and async code

🎯 CHECKPOINT: Build a CLI tool using Pydantic for config,
   a generator for streaming output, and async for the API call.

PHASE 3: LANGCHAIN FUNDAMENTALS (20 hours)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
□  LangChain Core Concepts (5 hrs)
   → LCEL: chain = prompt | llm | parser
   → Runnable interface: invoke, batch, stream, ainvoke
   → ChatPromptTemplate, SystemMessage, HumanMessage
   → Output parsers: StrOutputParser, JsonOutputParser

□  Tools (5 hrs)
   → @tool decorator
   → BaseTool subclassing (you already know this from Module 7!)
   → StructuredTool
   → Tool calling with bind_tools()

□  Memory & State (5 hrs)
   → ConversationBufferMemory
   → Message history
   → RunnableWithMessageHistory

□  Retrievers & RAG (5 hrs)
   → Document loaders
   → Text splitters
   → Vector stores (FAISS, Chroma)
   → Retrieval QA chain

🎯 CHECKPOINT: Build a RAG chatbot with tool calling and
   conversation memory, from scratch.

PHASE 4: LANGGRAPH (15 hours)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
□  LangGraph Core (5 hrs)
   → StateGraph, TypedDict State
   → add_node, add_edge, add_conditional_edges
   → compile() and invoke()
   → MessagesState (you built this in Module 3!)

□  Persistence & Checkpointing (5 hrs)
   → MemorySaver, SqliteSaver
   → Thread IDs and config
   → Resuming from checkpoints

□  Multi-Agent Patterns (5 hrs)
   → Supervisor pattern
   → Handoffs between agents
   → Parallel node execution
   → Subgraphs

🎯 CHECKPOINT: Build a multi-agent research system with:
   - A supervisor routing between 3 specialized agents
   - SQLite checkpointing for persistence
   - Tool-using agents with error recovery

TOTAL INVESTMENT: ~80 hours from zero Python to LangGraph mastery.
```

---

## When to Stop This Course and Switch to LangChain

**You are ready to start LangChain when you can:**

1. ✅ Write a `StateTool` base class with `invoke()` and `_run()` methods from memory in under 5 minutes.
2. ✅ Build a nested dict state object, merge two partial updates, and explain exactly why `copy.deepcopy()` matters.
3. ✅ Write a routing function that reads state and returns string node names, with 4+ branches.
4. ✅ Parse a structured string (like ReAct format) without looking up any methods.
5. ✅ Build a 3-file Python project (`state.py`, `tools.py`, `main.py`) and run it without import errors.
6. ✅ Explain the difference between a class attribute and an instance attribute without hesitation.
7. ✅ Handle exceptions at 3 different granularities: specific errors, `AgentError` base class, and catch-all.

**If you can do all 7: Close this guide and open the LangChain docs.**

`pip install langchain langchain-openai langgraph`

Your first file to read: `langchain/tools/base.py` — you will recognize it immediately.

---

_Course compiled for Python 3.10+ | Last verified against LangChain 0.3.x and LangGraph 0.2.x_
