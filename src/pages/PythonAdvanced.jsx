import { useState } from 'react'

function PythonAdvanced() {
  const [activeSection, setActiveSection] = useState('decorators')
  const [expandedCode, setExpandedCode] = useState(null)

  const sections = [
    { id: 'decorators', title: 'Decorators & Closures', icon: '🎪' },
    { id: 'generators', title: 'Generators & Iterators', icon: '🌀' },
    { id: 'context-managers', title: 'Context Managers', icon: '🚪' },
    { id: 'metaclasses', title: 'Metaclasses', icon: '🎭' },
    { id: 'async-programming', title: 'Async Programming', icon: '⚡' },
    { id: 'memory-management', title: 'Memory Management', icon: '💾' },
    { id: 'performance', title: 'Performance Optimization', icon: '🚀' },
    { id: 'testing', title: 'Testing & Debugging', icon: '🧪' }
  ]

  const codeExamples = {
    basicDecorator: `# Understanding Decorators: Functions that modify other functions

def timing_decorator(func):
    """Decorator that measures execution time"""
    import time
    
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    
    return wrapper

def logging_decorator(func):
    """Decorator that logs function calls"""
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__} with args: {args}, kwargs: {kwargs}")
        result = func(*args, **kwargs)
        print(f"{func.__name__} returned: {result}")
        return result
    
    return wrapper

# Using decorators
@timing_decorator
@logging_decorator
def fibonacci(n):
    """Calculate nth Fibonacci number"""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# Advanced decorator with parameters
def retry_decorator(max_attempts=3, delay=1):
    """Decorator that retries function on failure"""
    import time
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise e
                    print(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay} seconds...")
                    time.sleep(delay)
        
        return wrapper
    return decorator

@retry_decorator(max_attempts=3, delay=0.5)
def unreliable_network_call():
    import random
    if random.random() < 0.7:  # 70% chance of failure
        raise ConnectionError("Network timeout")
    return "Success! Data retrieved."

# Class-based decorators
class CountCalls:
    """Decorator class that counts function calls"""
    def __init__(self, func):
        self.func = func
        self.count = 0
    
    def __call__(self, *args, **kwargs):
        self.count += 1
        print(f"{self.func.__name__} has been called {self.count} times")
        return self.func(*args, **kwargs)

@CountCalls
def say_hello(name):
    return f"Hello, {name}!"`,

    generators: `# Generators: Memory-efficient iterators that yield values on-demand

def simple_generator():
    """Basic generator function"""
    print("Starting generator")
    yield 1
    print("First yield done")
    yield 2
    print("Second yield done")
    yield 3
    print("Generator finished")

# Generator expressions
squares = (x**2 for x in range(10))  # Generator expression
squares_list = [x**2 for x in range(10)]  # List comprehension

def fibonacci_generator(limit):
    """Generate Fibonacci sequence up to limit"""
    a, b = 0, 1
    while a < limit:
        yield a
        a, b = b, a + b

def read_large_file(filename):
    """Memory-efficient file reading"""
    with open(filename, 'r') as file:
        for line in file:
            # Process each line without loading entire file
            yield line.strip()

# Advanced generator: Processing pipeline
def number_generator(start, end):
    """Generate numbers in range"""
    for num in range(start, end):
        yield num

def square_generator(numbers):
    """Square each number"""
    for num in numbers:
        yield num ** 2

def filter_even(numbers):
    """Filter even numbers"""
    for num in numbers:
        if num % 2 == 0:
            yield num

# Chaining generators (pipeline pattern)
pipeline = filter_even(square_generator(number_generator(1, 11)))
result = list(pipeline)  # [4, 16, 36, 64, 100]

# Generator with send() method
def echo_generator():
    """Generator that can receive values"""
    value = None
    while True:
        received = yield value
        if received is not None:
            value = f"Echo: {received}"
        else:
            value = "Waiting for input..."

# Using generator protocols
class CountDown:
    """Iterator protocol implementation"""
    def __init__(self, start):
        self.start = start
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.start <= 0:
            raise StopIteration
        self.start -= 1
        return self.start + 1

# Memory comparison demonstration
def memory_efficient_processing():
    """Compare memory usage of generator vs list"""
    
    # Memory-hungry approach
    def process_with_list(n):
        return [x**2 for x in range(n)]  # Loads all into memory
    
    # Memory-efficient approach  
    def process_with_generator(n):
        return (x**2 for x in range(n))  # Generates on-demand
    
    # For large datasets, generator uses constant memory
    large_numbers = process_with_generator(1000000)
    # Process items one by one without memory issues
    for i, num in enumerate(large_numbers):
        if i >= 5:  # Process only first 5 for demo
            break
        print(f"Processed: {num}")`,

    contextManagers: `# Context Managers: Proper resource management with 'with' statements

# Basic file handling context manager (built-in)
with open('example.txt', 'w') as file:
    file.write('Hello, World!')
# File automatically closed, even if exception occurs

# Custom context manager using __enter__ and __exit__
class DatabaseConnection:
    """Custom context manager for database connections"""
    
    def __init__(self, database_name):
        self.database_name = database_name
        self.connection = None
    
    def __enter__(self):
        print(f"Connecting to {self.database_name}")
        self.connection = f"Connection to {self.database_name}"
        return self.connection
    
    def __exit__(self, exc_type, exc_value, traceback):
        print(f"Closing connection to {self.database_name}")
        if exc_type:
            print(f"Exception occurred: {exc_type.__name__}: {exc_value}")
            # Return False to propagate exception, True to suppress
            return False
        
        # Cleanup code always runs
        self.connection = None

# Using custom context manager
with DatabaseConnection("users_db") as conn:
    print(f"Working with {conn}")
    # Database operations here
    # Connection automatically closed after block

# Context manager using contextlib
from contextlib import contextmanager

@contextmanager
def temporary_file(filename, content):
    """Context manager for temporary file creation"""
    import os
    
    # Setup code (like __enter__)
    print(f"Creating temporary file: {filename}")
    with open(filename, 'w') as f:
        f.write(content)
    
    try:
        yield filename  # This value is returned to 'as' clause
    finally:
        # Cleanup code (like __exit__)
        print(f"Removing temporary file: {filename}")
        if os.path.exists(filename):
            os.remove(filename)

# Using contextlib context manager
with temporary_file("temp.txt", "Temporary content") as temp_file:
    print(f"Working with {temp_file}")
    with open(temp_file, 'r') as f:
        print(f"Content: {f.read()}")

# Advanced: Nested context managers
class ResourceManager:
    """Manages multiple resources"""
    
    def __init__(self, *resources):
        self.resources = resources
        self.acquired = []
    
    def __enter__(self):
        for resource in self.resources:
            try:
                acquired = resource.__enter__()
                self.acquired.append((resource, acquired))
            except Exception:
                # Cleanup already acquired resources
                self._cleanup()
                raise
        return [acquired for _, acquired in self.acquired]
    
    def __exit__(self, exc_type, exc_value, traceback):
        self._cleanup()
        return False
    
    def _cleanup(self):
        # Cleanup in reverse order
        for resource, _ in reversed(self.acquired):
            try:
                resource.__exit__(None, None, None)
            except Exception:
                pass  # Don't let cleanup exceptions mask original exception

# Context manager for timing code execution
@contextmanager
def timer(description):
    """Time code execution"""
    import time
    start = time.time()
    print(f"Starting: {description}")
    
    try:
        yield
    finally:
        end = time.time()
        print(f"Completed {description} in {end - start:.4f} seconds")

# Using timer context manager
with timer("Complex calculation"):
    # Simulate complex work
    import time
    time.sleep(0.1)
    result = sum(x**2 for x in range(1000))
    print(f"Result: {result}")`,

    asyncProgramming: `# Async Programming: Non-blocking code execution

import asyncio
import aiohttp
import time

# Basic async function
async def simple_async_function():
    """Basic async function demonstration"""
    print("Starting async operation")
    await asyncio.sleep(1)  # Non-blocking sleep
    print("Async operation completed")
    return "Result from async function"

# Async function with multiple awaits
async def fetch_data(url, session):
    """Simulate fetching data from a URL"""
    print(f"Fetching data from {url}")
    
    # Simulate network delay
    await asyncio.sleep(1)
    
    # In real scenario, you'd use aiohttp
    return f"Data from {url}"

async def process_urls():
    """Process multiple URLs concurrently"""
    urls = [
        "https://api.example1.com/data",
        "https://api.example2.com/data", 
        "https://api.example3.com/data"
    ]
    
    # Sequential approach (slow)
    print("=== Sequential Processing ===")
    start_time = time.time()
    results_sequential = []
    for url in urls:
        result = await fetch_data(url, None)
        results_sequential.append(result)
    sequential_time = time.time() - start_time
    
    # Concurrent approach (fast)
    print("\\n=== Concurrent Processing ===")
    start_time = time.time()
    
    # Create tasks for concurrent execution
    tasks = [fetch_data(url, None) for url in urls]
    results_concurrent = await asyncio.gather(*tasks)
    concurrent_time = time.time() - start_time
    
    print(f"Sequential time: {sequential_time:.2f}s")
    print(f"Concurrent time: {concurrent_time:.2f}s")
    print(f"Speedup: {sequential_time/concurrent_time:.2f}x")
    
    return results_concurrent

# Async context managers
class AsyncDatabaseConnection:
    """Async context manager for database connections"""
    
    async def __aenter__(self):
        print("Establishing async database connection")
        await asyncio.sleep(0.1)  # Simulate connection time
        return "Async DB Connection"
    
    async def __aexit__(self, exc_type, exc_value, traceback):
        print("Closing async database connection")
        await asyncio.sleep(0.05)  # Simulate cleanup time

async def database_operations():
    """Using async context manager"""
    async with AsyncDatabaseConnection() as conn:
        print(f"Working with {conn}")
        await asyncio.sleep(0.5)  # Simulate database work

# Async generators
async def async_number_generator(start, end, delay=0.1):
    """Async generator that yields numbers with delay"""
    for i in range(start, end):
        await asyncio.sleep(delay)  # Simulate async operation
        yield i

async def process_async_generator():
    """Process async generator"""
    async for number in async_number_generator(1, 6):
        print(f"Received: {number}")

# Producer-Consumer pattern with asyncio
async def producer(queue, name, count):
    """Produce items and put them in queue"""
    for i in range(count):
        item = f"{name}-item-{i}"
        await asyncio.sleep(0.1)  # Simulate work
        await queue.put(item)
        print(f"Produced: {item}")
    
    # Signal completion
    await queue.put(None)

async def consumer(queue, name):
    """Consume items from queue"""
    while True:
        item = await queue.get()
        if item is None:
            break
        
        print(f"{name} consuming: {item}")
        await asyncio.sleep(0.2)  # Simulate processing time
        queue.task_done()

async def producer_consumer_demo():
    """Demonstrate producer-consumer pattern"""
    queue = asyncio.Queue(maxsize=5)
    
    # Create producer and consumer tasks
    prod_task = asyncio.create_task(producer(queue, "Producer", 10))
    cons_task = asyncio.create_task(consumer(queue, "Consumer"))
    
    # Wait for producer to finish
    await prod_task
    
    # Wait for all items to be processed
    await queue.join()
    
    # Cancel consumer task
    cons_task.cancel()

# Error handling in async code
async def async_operation_with_error():
    """Async function that might fail"""
    await asyncio.sleep(0.1)
    if True:  # Simulate random error
        raise ValueError("Something went wrong!")
    return "Success"

async def handle_async_errors():
    """Proper error handling in async code"""
    tasks = [
        async_operation_with_error(),
        asyncio.sleep(0.2),
        async_operation_with_error()
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"Task {i} failed: {result}")
        else:
            print(f"Task {i} succeeded: {result}")

# Main async function to run examples
async def main():
    """Run all async examples"""
    print("=== Simple Async Function ===")
    result = await simple_async_function()
    print(f"Result: {result}")
    
    print("\\n=== URL Processing ===")
    await process_urls()
    
    print("\\n=== Async Context Manager ===")  
    await database_operations()
    
    print("\\n=== Async Generator ===")
    await process_async_generator()
    
    print("\\n=== Producer-Consumer ===")
    await producer_consumer_demo()
    
    print("\\n=== Error Handling ===")
    await handle_async_errors()

# Run the async program
if __name__ == "__main__":
    asyncio.run(main())`,

    performance: `# Performance Optimization in Python

import time
import sys
from functools import lru_cache
from collections import defaultdict, deque
import profile
import cProfile

# 1. Algorithm Optimization
def slow_fibonacci(n):
    """Inefficient recursive Fibonacci - O(2^n)"""
    if n <= 1:
        return n
    return slow_fibonacci(n-1) + slow_fibonacci(n-2)

@lru_cache(maxsize=None)
def fast_fibonacci(n):
    """Optimized Fibonacci with memoization - O(n)"""
    if n <= 1:
        return n
    return fast_fibonacci(n-1) + fast_fibonacci(n-2)

def iterative_fibonacci(n):
    """Most efficient Fibonacci - O(n) time, O(1) space"""
    if n <= 1:
        return n
    
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

# 2. Data Structure Selection
def compare_data_structures():
    """Compare performance of different data structures"""
    
    # List vs Set for membership testing
    large_list = list(range(100000))
    large_set = set(range(100000))
    
    # List membership test (slow - O(n))
    start = time.time()
    result = 99999 in large_list
    list_time = time.time() - start
    
    # Set membership test (fast - O(1))
    start = time.time()
    result = 99999 in large_set
    set_time = time.time() - start
    
    print(f"List membership test: {list_time:.6f}s")
    print(f"Set membership test: {set_time:.6f}s")
    print(f"Set is {list_time/set_time:.0f}x faster")

# 3. String Optimization
def slow_string_concatenation(strings):
    """Inefficient string concatenation"""
    result = ""
    for s in strings:
        result += s  # Creates new string each time
    return result

def fast_string_concatenation(strings):
    """Efficient string concatenation"""
    return "".join(strings)  # Single operation

# 4. List Comprehensions vs Loops
def traditional_loop(n):
    """Traditional loop approach"""
    result = []
    for i in range(n):
        if i % 2 == 0:
            result.append(i**2)
    return result

def list_comprehension(n):
    """List comprehension approach"""
    return [i**2 for i in range(n) if i % 2 == 0]

def generator_expression(n):
    """Generator expression for memory efficiency"""
    return (i**2 for i in range(n) if i % 2 == 0)

# 5. Memory Optimization
class SlotClass:
    """Memory-efficient class using __slots__"""
    __slots__ = ['name', 'age', 'email']
    
    def __init__(self, name, age, email):
        self.name = name
        self.age = age
        self.email = email

class RegularClass:
    """Regular class with __dict__"""
    
    def __init__(self, name, age, email):
        self.name = name
        self.age = age
        self.email = email

def memory_comparison():
    """Compare memory usage of __slots__ vs regular class"""
    
    # Create many instances
    slot_objects = [SlotClass(f"User{i}", i, f"user{i}@email.com") for i in range(1000)]
    regular_objects = [RegularClass(f"User{i}", i, f"user{i}@email.com") for i in range(1000)]
    
    # Memory usage would be significantly lower for slot_objects
    print(f"Created {len(slot_objects)} slot objects and {len(regular_objects)} regular objects")

# 6. Profiling Code
def profile_example():
    """Example function to profile"""
    
    def cpu_intensive_task():
        total = 0
        for i in range(1000000):
            total += i ** 2
        return total
    
    def memory_intensive_task():
        large_list = []
        for i in range(100000):
            large_list.append([i] * 10)
        return len(large_list)
    
    # Time both functions
    start = time.time()
    result1 = cpu_intensive_task()
    cpu_time = time.time() - start
    
    start = time.time()
    result2 = memory_intensive_task()
    memory_time = time.time() - start
    
    print(f"CPU intensive task: {cpu_time:.4f}s, result: {result1}")
    print(f"Memory intensive task: {memory_time:.4f}s, result: {result2}")

# 7. Caching Strategies
class DataProcessor:
    """Example class with different caching strategies"""
    
    def __init__(self):
        self._cache = {}
    
    @lru_cache(maxsize=128)
    def expensive_calculation(self, x, y):
        """Cached expensive calculation"""
        time.sleep(0.01)  # Simulate expensive operation
        return x ** y + y ** x
    
    def manual_cache(self, key):
        """Manual caching implementation"""
        if key in self._cache:
            return self._cache[key]
        
        # Expensive operation
        result = sum(i**2 for i in range(key))
        self._cache[key] = result
        return result

# 8. Performance Best Practices
def performance_tips():
    """Demonstrate various performance tips"""
    
    # Use local variables (faster than global)
    def optimized_loop():
        local_range = range  # Cache global function
        local_append = [].append  # Cache method
        
        result = []
        for i in local_range(1000):
            local_append(i**2)
    
    # Use appropriate data structures
    def find_common_elements(list1, list2):
        # Convert to sets for fast intersection
        return list(set(list1) & set(list2))
    
    # Avoid repeated attribute access
    def process_objects(objects):
        # Instead of: [obj.method() for obj in objects]
        # Use: [method() for method in [obj.method for obj in objects]]
        pass

# Performance measurement decorator
def measure_time(func):
    """Decorator to measure function execution time"""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.6f} seconds")
        return result
    return wrapper

# Example usage of performance optimization
@measure_time
def run_performance_tests():
    """Run all performance comparison tests"""
    
    print("=== Fibonacci Comparison ===")
    n = 30
    
    # Test slow version (comment out for large n)
    # slow_result = slow_fibonacci(n)
    
    fast_result = fast_fibonacci(n)
    iter_result = iterative_fibonacci(n)
    
    print(f"Fibonacci({n}) = {fast_result}")
    
    print("\\n=== Data Structure Comparison ===")
    compare_data_structures()
    
    print("\\n=== String Concatenation ===")
    test_strings = ["Hello", " ", "World", " ", "Performance", " ", "Test"] * 1000
    
    start = time.time()
    slow_result = slow_string_concatenation(test_strings[:100])  # Smaller for demo
    slow_time = time.time() - start
    
    start = time.time()
    fast_result = fast_string_concatenation(test_strings)
    fast_time = time.time() - start
    
    print(f"Slow concatenation: {slow_time:.6f}s")
    print(f"Fast concatenation: {fast_time:.6f}s")
    
    print("\\n=== Memory Optimization ===")
    memory_comparison()
    
    print("\\n=== Profiling Example ===")
    profile_example()

if __name__ == "__main__":
    run_performance_tests()`
  }

  const renderContent = () => {
    switch(activeSection) {
      case 'decorators':
        return (
          <div className="section-content">
            <h2>🎪 Python Decorators & Closures</h2>
            
            <div className="intro-section">
              <h3>What are Decorators?</h3>
              <p>
                Decorators are a powerful Python feature that allows you to modify or extend the behavior of functions or classes 
                without permanently modifying their code. They're essentially functions that take another function as an argument 
                and return a modified version of that function.
              </p>
            </div>

            <div className="analogy-box">
              <h4>🎭 Gift Wrapping Analogy</h4>
              <p>
                Think of decorators like gift wrapping. You have a present (your original function), and you wrap it with 
                decorative paper (the decorator). The gift itself doesn't change, but now it has additional features - 
                it looks prettier, might have a bow, or special handling instructions. When someone receives it, 
                they still get the original gift, but with extra functionality.
              </p>
            </div>

            <div className="code-example">
              <h4>Complete Decorator Examples</h4>
              <pre>{codeExamples.basicDecorator}</pre>
              <button onClick={() => setExpandedCode(expandedCode === 'decorators' ? null : 'decorators')}>
                {expandedCode === 'decorators' ? 'Hide Decorator Types' : 'Show Different Decorator Types'}
              </button>
              {expandedCode === 'decorators' && (
                <div className="code-explanation">
                  <h5>🎯 Types of Decorators:</h5>
                  <div className="data-types-grid">
                    <div className="data-type-card">
                      <h6>Function Decorators</h6>
                      <p>Most common - decorate functions</p>
                      <code>@timing_decorator</code>
                    </div>
                    <div className="data-type-card">
                      <h6>Class Decorators</h6>
                      <p>Decorate entire classes</p>
                      <code>@dataclass</code>
                    </div>
                    <div className="data-type-card">
                      <h6>Parameterized Decorators</h6>
                      <p>Decorators that accept arguments</p>
                      <code>@retry(max_attempts=3)</code>
                    </div>
                    <div className="data-type-card">
                      <h6>Class-based Decorators</h6>
                      <p>Using classes as decorators</p>
                      <code>@CountCalls</code>
                    </div>
                  </div>
                </div>
              )}
            </div>

            <div className="concept-deep-dive">
              <h3>🔑 Key Decorator Concepts</h3>
              <div className="math-concept">
                <h4>Understanding Closures</h4>
                <p>Decorators rely on closures - inner functions that remember variables from their outer scope even after the outer function has finished executing.</p>
                <div className="code-example">
                  <pre>{`def outer_function(x):
    def inner_function(y):
        return x + y  # 'x' is remembered from outer scope
    return inner_function

# 'x' is "closed over" by inner_function
add_five = outer_function(5)
result = add_five(3)  # Returns 8`}</pre>
                </div>
              </div>
            </div>
          </div>
        )

      case 'generators':
        return (
          <div className="section-content">
            <h2>🌀 Generators & Iterators</h2>
            
            <div className="intro-section">
              <h3>What are Generators?</h3>
              <p>
                Generators are memory-efficient tools that produce items on-demand rather than creating entire sequences in memory. 
                They use the <code>yield</code> keyword to return values one at a time, pausing and resuming execution as needed.
              </p>
            </div>

            <div className="analogy-box">
              <h4>🏭 Factory Assembly Line Analogy</h4>
              <p>
                Think of generators like an assembly line that produces products one at a time, only when requested. 
                Instead of manufacturing all products at once and storing them in a warehouse (like lists do), 
                the assembly line creates each product on-demand, saving storage space and resources.
              </p>
            </div>

            <div className="code-example">
              <h4>Generator Examples & Memory Efficiency</h4>
              <pre>{codeExamples.generators}</pre>
              <button onClick={() => setExpandedCode(expandedCode === 'generators' ? null : 'generators')}>
                {expandedCode === 'generators' ? 'Hide Generator Benefits' : 'Show Memory & Performance Benefits'}
              </button>
              {expandedCode === 'generators' && (
                <div className="code-explanation">
                  <h5>🚀 Generator Advantages:</h5>
                  <ul>
                    <li><strong>Memory Efficiency:</strong> Only one item in memory at a time</li>
                    <li><strong>Lazy Evaluation:</strong> Values computed only when needed</li>
                    <li><strong>Infinite Sequences:</strong> Can represent infinite data streams</li>
                    <li><strong>Pipeline Processing:</strong> Chain generators for complex data processing</li>
                    <li><strong>Better Performance:</strong> No need to store entire sequences</li>
                  </ul>
                </div>
              )}
            </div>

            <div className="concept-deep-dive">
              <h3>⚡ Generator vs List: Performance Comparison</h3>
              <div className="math-concept">
                <h4>Memory Usage Analysis</h4>
                <div className="data-types-grid">
                  <div className="data-type-card">
                    <h5>List Comprehension</h5>
                    <p>Creates entire list in memory</p>
                    <code>[x**2 for x in range(1000000)]</code>
                    <div className="real-example">Memory: ~37MB for 1M integers</div>
                  </div>
                  <div className="data-type-card">
                    <h5>Generator Expression</h5>
                    <p>Creates generator object only</p>
                    <code>(x**2 for x in range(1000000))</code>
                    <div className="real-example">Memory: ~128 bytes (constant)</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )

      case 'context-managers':
        return (
          <div className="section-content">
            <h2>🚪 Context Managers</h2>
            
            <div className="intro-section">
              <h3>What are Context Managers?</h3>
              <p>
                Context managers provide a way to allocate and release resources precisely when you want to. 
                They ensure that cleanup code runs even if an error occurs, making your code more robust and preventing resource leaks.
              </p>
            </div>

            <div className="analogy-box">
              <h4>🏨 Hotel Room Analogy</h4>
              <p>
                Context managers are like staying in a hotel room. When you check in, the room is prepared for you (resources allocated). 
                During your stay, you use the room (your code runs). When you check out, housekeeping cleans the room regardless 
                of whether you left it messy or had to leave suddenly (cleanup always happens).
              </p>
            </div>

            <div className="code-example">
              <h4>Context Manager Implementations</h4>
              <pre>{codeExamples.contextManagers}</pre>
            </div>

            <div className="concept-deep-dive">
              <h3>🛡️ Why Context Managers Matter</h3>
              <div className="data-types-grid">
                <div className="data-type-card">
                  <h4>🔒 Resource Safety</h4>
                  <p>Guarantee resource cleanup</p>
                  <div className="real-example">Files, database connections, locks</div>
                </div>
                <div className="data-type-card">
                  <h4>🎯 Exception Handling</h4>
                  <p>Cleanup runs even if exceptions occur</p>
                  <div className="real-example">Finally block alternative</div>
                </div>
                <div className="data-type-card">
                  <h4>📝 Cleaner Code</h4>
                  <p>No need for manual try/finally blocks</p>
                  <div className="real-example">More readable and maintainable</div>
                </div>
                <div className="data-type-card">
                  <h4>🔄 Reusability</h4>
                  <p>Encapsulate setup/teardown logic</p>
                  <div className="real-example">Use across multiple functions</div>
                </div>
              </div>
            </div>
          </div>
        )

      case 'async-programming':
        return (
          <div className="section-content">
            <h2>⚡ Async Programming in Python</h2>
            
            <div className="intro-section">
              <h3>What is Async Programming?</h3>
              <p>
                Asynchronous programming allows your program to handle multiple tasks concurrently without blocking. 
                Instead of waiting for slow operations (like network requests) to complete, your program can work on other tasks, 
                making it much more efficient for I/O-bound operations.
              </p>
            </div>

            <div className="analogy-box">
              <h4>👨‍🍳 Restaurant Kitchen Analogy</h4>
              <p>
                Imagine a chef cooking multiple dishes. In synchronous cooking, the chef would start one dish, wait for it to finish completely, 
                then start the next. In asynchronous cooking, while one dish is simmering (waiting), the chef starts preparing other dishes. 
                When the timer goes off (task completes), the chef returns to finish that dish. Much more efficient!
              </p>
            </div>

            <div className="code-example">
              <h4>Comprehensive Async Examples</h4>
              <pre>{codeExamples.asyncProgramming}</pre>
            </div>

            <div className="concept-deep-dive">
              <h3>🚀 Async Performance Benefits</h3>
              <div className="math-concept">
                <h4>Concurrent vs Sequential Execution</h4>
                <div className="direction-comparison">
                  <div className="human-directions">
                    <h5>Sequential (Synchronous)</h5>
                    <ul>
                      <li>Task 1: 2 seconds</li>
                      <li>Task 2: 2 seconds</li>
                      <li>Task 3: 2 seconds</li>
                      <li><strong>Total: 6 seconds</strong></li>
                    </ul>
                  </div>
                  <div className="computer-directions">
                    <h5>Concurrent (Asynchronous)</h5>
                    <ul>
                      <li>All tasks start simultaneously</li>
                      <li>CPU switches between tasks</li>
                      <li>While one waits, others work</li>
                      <li><strong>Total: ~2 seconds</strong></li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )

      case 'performance':
        return (
          <div className="section-content">
            <h2>🚀 Performance Optimization</h2>
            
            <div className="intro-section">
              <h3>Python Performance Optimization</h3>
              <p>
                While Python is known for being readable and easy to write, it can also be optimized for performance. 
                Understanding algorithmic complexity, data structure selection, and Python-specific optimizations 
                can make your code significantly faster.
              </p>
            </div>

            <div className="analogy-box">
              <h4>🏎️ Race Car Optimization Analogy</h4>
              <p>
                Optimizing Python code is like tuning a race car. You can upgrade the engine (algorithm choice), 
                reduce weight (memory usage), improve aerodynamics (data structure selection), and fine-tune 
                various components (Python-specific optimizations) to achieve maximum performance.
              </p>
            </div>

            <div className="code-example">
              <h4>Performance Optimization Techniques</h4>
              <pre>{codeExamples.performance}</pre>
            </div>

            <div className="concept-deep-dive">
              <h3>📊 Optimization Strategies</h3>
              <div className="data-types-grid">
                <div className="data-type-card">
                  <h4>🧮 Algorithm Choice</h4>
                  <p>Choose the right algorithm for the problem</p>
                  <div className="real-example">O(2^n) → O(n) with memoization</div>
                </div>
                <div className="data-type-card">
                  <h4>🏗️ Data Structures</h4>
                  <p>Use appropriate data structures</p>
                  <div className="real-example">Set lookup O(1) vs List O(n)</div>
                </div>
                <div className="data-type-card">
                  <h4>💾 Memory Usage</h4>
                  <p>Optimize memory consumption</p>
                  <div className="real-example">__slots__, generators, object pooling</div>
                </div>
                <div className="data-type-card">
                  <h4>🔧 Python Idioms</h4>
                  <p>Use Python-specific optimizations</p>
                  <div className="real-example">List comprehensions, built-in functions</div>
                </div>
              </div>
            </div>
          </div>
        )

      default:
        return <div>Select a topic from the sidebar to continue learning advanced Python!</div>
    }
  }

  return (
    <div className="page">
      <div className="learning-container">
        <div className="sidebar">
          <h3>⚡ Advanced Python Topics</h3>
          <div className="section-nav">
            {sections.map((section) => (
              <button
                key={section.id}
                className={`section-link ${activeSection === section.id ? 'active' : ''}`}
                onClick={() => setActiveSection(section.id)}
              >
                {section.icon} {section.title}
              </button>
            ))}
          </div>
        </div>

        <div className="content-area">
          <div className="section-header">
            <div className="progress-bar">
              <div 
                className="progress-fill" 
                style={{ width: `${((sections.findIndex(s => s.id === activeSection) + 1) / sections.length) * 100}%` }}
              ></div>
            </div>
            <p>Advanced Topic {sections.findIndex(s => s.id === activeSection) + 1} of {sections.length}</p>
          </div>

          {renderContent()}

          <div className="section-navigation">
            <button 
              className="nav-btn"
              onClick={() => {
                const currentIndex = sections.findIndex(s => s.id === activeSection)
                if (currentIndex > 0) {
                  setActiveSection(sections[currentIndex - 1].id)
                }
              }}
              disabled={sections.findIndex(s => s.id === activeSection) === 0}
            >
              ← Previous Topic
            </button>
            <button 
              className="nav-btn"
              onClick={() => {
                const currentIndex = sections.findIndex(s => s.id === activeSection)
                if (currentIndex < sections.length - 1) {
                  setActiveSection(sections[currentIndex + 1].id)
                }
              }}
              disabled={sections.findIndex(s => s.id === activeSection) === sections.length - 1}
            >
              Next Topic →
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

export default PythonAdvanced