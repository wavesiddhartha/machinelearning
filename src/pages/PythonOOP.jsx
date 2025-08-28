import { useState } from 'react'

function PythonOOP() {
  const [activeSection, setActiveSection] = useState('introduction')
  const [expandedCode, setExpandedCode] = useState(null)

  const sections = [
    { id: 'introduction', title: 'Introduction to OOP', icon: '🏛️' },
    { id: 'classes-objects', title: 'Classes & Objects', icon: '🏗️' },
    { id: 'attributes-methods', title: 'Attributes & Methods', icon: '⚙️' },
    { id: 'inheritance', title: 'Inheritance', icon: '👨‍👩‍👧‍👦' },
    { id: 'polymorphism', title: 'Polymorphism', icon: '🔄' },
    { id: 'encapsulation', title: 'Encapsulation', icon: '🔒' },
    { id: 'abstraction', title: 'Abstraction', icon: '🎭' },
    { id: 'magic-methods', title: 'Magic Methods', icon: '✨' },
    { id: 'properties', title: 'Properties & Decorators', icon: '🎯' },
    { id: 'design-patterns', title: 'Design Patterns', icon: '🏛️' }
  ]

  const codeExamples = {
    basicClass: `# Basic Class Definition
class Person:
    # Class variable (shared by all instances)
    species = "Homo sapiens"
    
    def __init__(self, name, age):
        # Instance variables (unique to each instance)
        self.name = name
        self.age = age
        self.energy = 100
    
    def introduce(self):
        return f"Hi, I'm {self.name} and I'm {self.age} years old."
    
    def sleep(self, hours):
        self.energy = min(100, self.energy + hours * 10)
        return f"{self.name} slept for {hours} hours. Energy: {self.energy}"

# Creating instances (objects)
person1 = Person("Alice", 25)
person2 = Person("Bob", 30)

print(person1.introduce())  # Hi, I'm Alice and I'm 25 years old.
print(person2.sleep(8))     # Bob slept for 8 hours. Energy: 100`,

    inheritance: `# Inheritance: Child classes inherit from parent classes
class Animal:
    def __init__(self, name, species):
        self.name = name
        self.species = species
        self.is_alive = True
    
    def eat(self, food):
        return f"{self.name} is eating {food}"
    
    def sleep(self):
        return f"{self.name} is sleeping"

class Dog(Animal):  # Dog inherits from Animal
    def __init__(self, name, breed):
        super().__init__(name, "Canine")  # Call parent constructor
        self.breed = breed
        self.loyalty = 100
    
    def bark(self):
        return f"{self.name} says: Woof! Woof!"
    
    def fetch(self, item):
        return f"{self.name} fetched the {item}!"

class Cat(Animal):  # Cat also inherits from Animal
    def __init__(self, name, color):
        super().__init__(name, "Feline")
        self.color = color
        self.independence = 90
    
    def meow(self):
        return f"{self.name} says: Meow!"
    
    def hunt(self, prey):
        return f"{self.name} is hunting {prey}"

# Using inherited classes
dog = Dog("Buddy", "Golden Retriever")
cat = Cat("Whiskers", "Orange")

print(dog.eat("kibble"))    # Inherited method
print(dog.bark())           # Dog-specific method
print(cat.sleep())          # Inherited method
print(cat.meow())           # Cat-specific method`,

    polymorphism: `# Polymorphism: Same interface, different implementations
class Shape:
    def area(self):
        raise NotImplementedError("Subclass must implement area method")
    
    def perimeter(self):
        raise NotImplementedError("Subclass must implement perimeter method")

class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height
    
    def area(self):
        return self.width * self.height
    
    def perimeter(self):
        return 2 * (self.width + self.height)

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius
    
    def area(self):
        return 3.14159 * self.radius ** 2
    
    def perimeter(self):
        return 2 * 3.14159 * self.radius

# Polymorphism in action
def print_shape_info(shape):
    print(f"Area: {shape.area():.2f}")
    print(f"Perimeter: {shape.perimeter():.2f}")

shapes = [
    Rectangle(5, 3),
    Circle(4),
    Rectangle(2, 8)
]

for shape in shapes:
    print_shape_info(shape)  # Same function works for all shapes!`,

    encapsulation: `# Encapsulation: Hiding internal details and controlling access
class BankAccount:
    def __init__(self, account_number, initial_balance=0):
        self._account_number = account_number    # Protected (convention)
        self.__balance = initial_balance         # Private (name mangling)
        self.__transaction_history = []         # Private list
    
    # Public methods to interact with private data
    def deposit(self, amount):
        if amount > 0:
            self.__balance += amount
            self.__transaction_history.append(f"Deposited: ${amount}")
            return f"Deposited ${amount}. New balance: ${self.__balance}"
        return "Invalid deposit amount"
    
    def withdraw(self, amount):
        if 0 < amount <= self.__balance:
            self.__balance -= amount
            self.__transaction_history.append(f"Withdrew: ${amount}")
            return f"Withdrew ${amount}. New balance: ${self.__balance}"
        return "Insufficient funds or invalid amount"
    
    def get_balance(self):
        return self.__balance
    
    def get_statement(self):
        return self.__transaction_history.copy()  # Return copy, not original
    
    # Property to safely access account number
    @property
    def account_number(self):
        return self._account_number

# Using encapsulation
account = BankAccount("12345", 1000)
print(account.deposit(500))     # Deposited $500. New balance: $1500
print(account.withdraw(200))    # Withdrew $200. New balance: $1300
print(account.get_balance())    # 1300

# These would cause errors or not work as expected:
# print(account.__balance)      # AttributeError (private)
# account.__balance = 999999    # Won't change the actual balance`,

    magicMethods: `# Magic Methods (Dunder methods): Special methods that define object behavior
class Book:
    def __init__(self, title, author, pages):
        self.title = title
        self.author = author
        self.pages = pages
    
    def __str__(self):
        # Called by str() and print()
        return f"'{self.title}' by {self.author}"
    
    def __repr__(self):
        # Called by repr() and in interactive shell
        return f"Book('{self.title}', '{self.author}', {self.pages})"
    
    def __len__(self):
        # Called by len()
        return self.pages
    
    def __eq__(self, other):
        # Called by ==
        if isinstance(other, Book):
            return self.title == other.title and self.author == other.author
        return False
    
    def __lt__(self, other):
        # Called by < (enables sorting)
        if isinstance(other, Book):
            return self.pages < other.pages
        return NotImplemented
    
    def __add__(self, other):
        # Called by +
        if isinstance(other, Book):
            combined_title = f"{self.title} & {other.title}"
            combined_author = f"{self.author} & {other.author}"
            combined_pages = self.pages + other.pages
            return Book(combined_title, combined_author, combined_pages)
        return NotImplemented

# Using magic methods
book1 = Book("1984", "George Orwell", 328)
book2 = Book("Animal Farm", "George Orwell", 112)
book3 = Book("1984", "George Orwell", 328)

print(str(book1))           # '1984' by George Orwell
print(repr(book2))          # Book('Animal Farm', 'George Orwell', 112)
print(len(book1))           # 328
print(book1 == book3)       # True
print(book1 == book2)       # False
print(book1 < book2)        # False (328 pages vs 112 pages)

# Combining books
combined = book1 + book2
print(combined)             # '1984 & Animal Farm' by George Orwell & George Orwell`,

    designPatterns: `# Design Patterns: Proven solutions to common problems

# 1. Singleton Pattern: Ensure only one instance exists
class DatabaseConnection:
    _instance = None
    _connection = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def connect(self, database_url):
        if self._connection is None:
            self._connection = f"Connected to {database_url}"
            print(f"New connection: {self._connection}")
        else:
            print("Using existing connection")
        return self._connection

# 2. Factory Pattern: Create objects without specifying exact class
class VehicleFactory:
    @staticmethod
    def create_vehicle(vehicle_type, **kwargs):
        if vehicle_type.lower() == "car":
            return Car(**kwargs)
        elif vehicle_type.lower() == "motorcycle":
            return Motorcycle(**kwargs)
        elif vehicle_type.lower() == "truck":
            return Truck(**kwargs)
        else:
            raise ValueError(f"Unknown vehicle type: {vehicle_type}")

class Car:
    def __init__(self, make, model, doors=4):
        self.make = make
        self.model = model
        self.doors = doors
    
    def __str__(self):
        return f"{self.make} {self.model} ({self.doors} doors)"

class Motorcycle:
    def __init__(self, make, model, engine_cc):
        self.make = make
        self.model = model
        self.engine_cc = engine_cc
    
    def __str__(self):
        return f"{self.make} {self.model} ({self.engine_cc}cc)"

class Truck:
    def __init__(self, make, model, payload):
        self.make = make
        self.model = model
        self.payload = payload
    
    def __str__(self):
        return f"{self.make} {self.model} (Payload: {self.payload}kg)"

# 3. Observer Pattern: Notify multiple objects about state changes
class NewsAgency:
    def __init__(self):
        self._news = ""
        self._subscribers = []
    
    def subscribe(self, subscriber):
        self._subscribers.append(subscriber)
    
    def unsubscribe(self, subscriber):
        self._subscribers.remove(subscriber)
    
    def set_news(self, news):
        self._news = news
        self._notify_all()
    
    def _notify_all(self):
        for subscriber in self._subscribers:
            subscriber.update(self._news)

class NewsChannel:
    def __init__(self, name):
        self.name = name
    
    def update(self, news):
        print(f"{self.name} broadcasting: {news}")

# Using design patterns
# Singleton
db1 = DatabaseConnection()
db2 = DatabaseConnection()
print(db1 is db2)  # True - same instance

# Factory
car = VehicleFactory.create_vehicle("car", make="Toyota", model="Camry")
bike = VehicleFactory.create_vehicle("motorcycle", make="Honda", model="CBR", engine_cc=600)

# Observer
agency = NewsAgency()
cnn = NewsChannel("CNN")
bbc = NewsChannel("BBC")

agency.subscribe(cnn)
agency.subscribe(bbc)
agency.set_news("Python 3.12 Released!")  # Both channels get notified`
  }

  const renderContent = () => {
    switch(activeSection) {
      case 'introduction':
        return (
          <div className="section-content">
            <h2>🏛️ Introduction to Object-Oriented Programming</h2>
            <div className="intro-section">
              <h3>What is Object-Oriented Programming (OOP)?</h3>
              <p>
                Object-Oriented Programming is a programming paradigm that organizes code into objects and classes. 
                Instead of writing functions that operate on data, OOP combines data and the functions that work on that data into single units called objects.
              </p>
            </div>

            <div className="concept-deep-dive">
              <h3>🤔 Why OOP? Real-World Analogy</h3>
              <div className="analogy-box">
                <h4>Think of OOP like organizing a company:</h4>
                <ul>
                  <li><strong>Classes</strong> are like job descriptions (blueprints for employees)</li>
                  <li><strong>Objects</strong> are like actual employees (instances of job descriptions)</li>
                  <li><strong>Attributes</strong> are like employee properties (name, salary, department)</li>
                  <li><strong>Methods</strong> are like employee skills (what they can do)</li>
                  <li><strong>Inheritance</strong> is like job hierarchies (managers inherit from employees)</li>
                  <li><strong>Encapsulation</strong> is like keeping salary information private</li>
                </ul>
              </div>
            </div>

            <div className="math-concept">
              <h3>🔑 Four Pillars of OOP</h3>
              <div className="data-types-grid">
                <div className="data-type-card">
                  <h4>🏗️ Encapsulation</h4>
                  <p>Bundling data and methods together, hiding internal details</p>
                  <div className="real-example">Like a car - you don't need to know how the engine works to drive it</div>
                </div>
                <div className="data-type-card">
                  <h4>👨‍👩‍👧‍👦 Inheritance</h4>
                  <p>Creating new classes based on existing classes</p>
                  <div className="real-example">Like how a sports car inherits basic car features but adds speed</div>
                </div>
                <div className="data-type-card">
                  <h4>🔄 Polymorphism</h4>
                  <p>Same interface, different implementations</p>
                  <div className="real-example">Like how different animals make different sounds when they "speak"</div>
                </div>
                <div className="data-type-card">
                  <h4>🎭 Abstraction</h4>
                  <p>Hiding complex implementation details, showing only essentials</p>
                  <div className="real-example">Like using a TV remote - simple buttons hide complex electronics</div>
                </div>
              </div>
            </div>
          </div>
        )

      case 'classes-objects':
        return (
          <div className="section-content">
            <h2>🏗️ Classes & Objects in Python</h2>
            
            <div className="intro-section">
              <h3>Understanding Classes and Objects</h3>
              <p>
                A <strong>class</strong> is a blueprint or template for creating objects. An <strong>object</strong> is an instance of a class - 
                a concrete realization with specific values.
              </p>
            </div>

            <div className="analogy-box">
              <h4>🏠 Real-World Analogy: House Blueprint vs Actual House</h4>
              <div className="direction-comparison">
                <div className="human-directions">
                  <h5>Class (Blueprint)</h5>
                  <ul>
                    <li>Defines structure (rooms, doors, windows)</li>
                    <li>Specifies materials needed</li>
                    <li>Shows how everything connects</li>
                    <li>Can be used to build multiple houses</li>
                  </ul>
                </div>
                <div className="computer-directions">
                  <h5>Object (Actual House)</h5>
                  <ul>
                    <li>Has specific address and color</li>
                    <li>Contains actual furniture</li>
                    <li>Has unique owners</li>
                    <li>Each house is different despite same blueprint</li>
                  </ul>
                </div>
              </div>
            </div>

            <div className="code-example">
              <h4>Basic Class Example</h4>
              <pre>{codeExamples.basicClass}</pre>
              <button onClick={() => setExpandedCode(expandedCode === 'basicClass' ? null : 'basicClass')}>
                {expandedCode === 'basicClass' ? 'Hide Explanation' : 'Show Detailed Explanation'}
              </button>
              {expandedCode === 'basicClass' && (
                <div className="code-explanation">
                  <h5>🔍 Code Breakdown:</h5>
                  <ul>
                    <li><code>class Person:</code> - Defines a new class named Person</li>
                    <li><code>species = "Homo sapiens"</code> - Class variable (shared by all instances)</li>
                    <li><code>def __init__(self, name, age):</code> - Constructor method, runs when object is created</li>
                    <li><code>self.name = name</code> - Instance variable (unique to each object)</li>
                    <li><code>def introduce(self):</code> - Instance method (function that belongs to the class)</li>
                    <li><code>person1 = Person("Alice", 25)</code> - Creating an object (instantiation)</li>
                  </ul>
                </div>
              )}
            </div>

            <div className="concept-deep-dive">
              <h3>🔑 Key Concepts Explained</h3>
              <div className="data-types-grid">
                <div className="data-type-card">
                  <h4>🏷️ Class Variables</h4>
                  <p>Shared by all instances of the class</p>
                  <code>species = "Homo sapiens"</code>
                </div>
                <div className="data-type-card">
                  <h4>👤 Instance Variables</h4>
                  <p>Unique to each object instance</p>
                  <code>self.name = "Alice"</code>
                </div>
                <div className="data-type-card">
                  <h4>🔧 Constructor (__init__)</h4>
                  <p>Special method that initializes new objects</p>
                  <code>def __init__(self, name):</code>
                </div>
                <div className="data-type-card">
                  <h4>⚙️ Instance Methods</h4>
                  <p>Functions that operate on instance data</p>
                  <code>def introduce(self):</code>
                </div>
              </div>
            </div>
          </div>
        )

      case 'inheritance':
        return (
          <div className="section-content">
            <h2>👨‍👩‍👧‍👦 Inheritance in Python</h2>
            
            <div className="intro-section">
              <h3>What is Inheritance?</h3>
              <p>
                Inheritance allows you to create new classes that reuse, extend, and modify the behavior of existing classes. 
                The new class (child/derived) inherits attributes and methods from the existing class (parent/base).
              </p>
            </div>

            <div className="analogy-box">
              <h4>🧬 Biological Inheritance Analogy</h4>
              <p>
                Just like how children inherit traits from their parents (eye color, height tendency), 
                child classes inherit properties and behaviors from parent classes, and can also develop their own unique traits.
              </p>
            </div>

            <div className="code-example">
              <h4>Inheritance Example: Animal Kingdom</h4>
              <pre>{codeExamples.inheritance}</pre>
              <button onClick={() => setExpandedCode(expandedCode === 'inheritance' ? null : 'inheritance')}>
                {expandedCode === 'inheritance' ? 'Hide Analysis' : 'Show Code Analysis'}
              </button>
              {expandedCode === 'inheritance' && (
                <div className="code-explanation">
                  <h5>🔍 Inheritance Analysis:</h5>
                  <ul>
                    <li><code>class Dog(Animal):</code> - Dog inherits from Animal (Dog IS-A Animal)</li>
                    <li><code>super().__init__(name, "Canine")</code> - Calls parent constructor</li>
                    <li>Dog inherits: name, species, is_alive, eat(), sleep()</li>
                    <li>Dog adds: breed, loyalty, bark(), fetch()</li>
                    <li>Both Dog and Cat inherit from Animal but have unique behaviors</li>
                  </ul>
                </div>
              )}
            </div>

            <div className="concept-deep-dive">
              <h3>🌳 Types of Inheritance</h3>
              <div className="data-types-grid">
                <div className="data-type-card">
                  <h4>👤 Single Inheritance</h4>
                  <p>Child class inherits from one parent class</p>
                  <code>class Dog(Animal):</code>
                </div>
                <div className="data-type-card">
                  <h4>👥 Multiple Inheritance</h4>
                  <p>Child class inherits from multiple parent classes</p>
                  <code>class Child(Mother, Father):</code>
                </div>
                <div className="data-type-card">
                  <h4>📚 Multilevel Inheritance</h4>
                  <p>Chain of inheritance: A→B→C</p>
                  <code>Animal → Mammal → Dog</code>
                </div>
                <div className="data-type-card">
                  <h4>🌟 Hierarchical Inheritance</h4>
                  <p>Multiple child classes from one parent</p>
                  <code>Animal → [Dog, Cat, Bird]</code>
                </div>
              </div>
            </div>
          </div>
        )

      case 'polymorphism':
        return (
          <div className="section-content">
            <h2>🔄 Polymorphism in Python</h2>
            
            <div className="intro-section">
              <h3>What is Polymorphism?</h3>
              <p>
                Polymorphism means "many forms". It allows objects of different types to be treated as instances of the same type through a common interface. 
                The same method name can have different implementations in different classes.
              </p>
            </div>

            <div className="analogy-box">
              <h4>🎵 Musical Instruments Analogy</h4>
              <p>
                Think of different musical instruments: piano, guitar, drums. They all can "play()" music, but each creates sound differently. 
                You can tell any instrument to "play()" without knowing the specific details of how it produces sound.
              </p>
            </div>

            <div className="code-example">
              <h4>Polymorphism with Shapes</h4>
              <pre>{codeExamples.polymorphism}</pre>
              <button onClick={() => setExpandedCode(expandedCode === 'polymorphism' ? null : 'polymorphism')}>
                {expandedCode === 'polymorphism' ? 'Hide Benefits' : 'Show Polymorphism Benefits'}
              </button>
              {expandedCode === 'polymorphism' && (
                <div className="code-explanation">
                  <h5>✅ Benefits of Polymorphism:</h5>
                  <ul>
                    <li><strong>Code Reusability:</strong> Same function works with different object types</li>
                    <li><strong>Flexibility:</strong> Easy to add new shapes without changing existing code</li>
                    <li><strong>Maintainability:</strong> Changes to specific implementations don't affect client code</li>
                    <li><strong>Abstraction:</strong> Client code doesn't need to know specific object types</li>
                  </ul>
                </div>
              )}
            </div>

            <div className="concept-deep-dive">
              <h3>🎭 Types of Polymorphism in Python</h3>
              <div className="data-types-grid">
                <div className="data-type-card">
                  <h4>🔧 Method Overriding</h4>
                  <p>Child class provides specific implementation of parent method</p>
                  <div className="real-example">Dog.make_sound() returns "Woof!" while Cat.make_sound() returns "Meow!"</div>
                </div>
                <div className="data-type-card">
                  <h4>🎯 Duck Typing</h4>
                  <p>"If it walks like a duck and quacks like a duck, it's a duck"</p>
                  <div className="real-example">Any object with a play() method can be used as a musical instrument</div>
                </div>
                <div className="data-type-card">
                  <h4>➕ Operator Overloading</h4>
                  <p>Same operator works differently with different types</p>
                  <div className="real-example">+ adds numbers, concatenates strings, merges lists</div>
                </div>
                <div className="data-type-card">
                  <h4>🔄 Method Overloading</h4>
                  <p>Same method name, different parameters (simulated in Python)</p>
                  <div className="real-example">Using default parameters or *args for flexibility</div>
                </div>
              </div>
            </div>
          </div>
        )

      case 'encapsulation':
        return (
          <div className="section-content">
            <h2>🔒 Encapsulation in Python</h2>
            
            <div className="intro-section">
              <h3>What is Encapsulation?</h3>
              <p>
                Encapsulation is the practice of restricting access to certain details of an object and exposing only what's necessary. 
                It's about bundling data and methods together and controlling how they're accessed from outside the class.
              </p>
            </div>

            <div className="analogy-box">
              <h4>🏧 ATM Machine Analogy</h4>
              <p>
                When you use an ATM, you can check balance, withdraw money, and deposit funds through simple buttons. 
                The complex internal workings (database connections, security protocols, money counting) are hidden from you. 
                You interact with a safe, controlled interface.
              </p>
            </div>

            <div className="code-example">
              <h4>Encapsulation: Bank Account Example</h4>
              <pre>{codeExamples.encapsulation}</pre>
              <button onClick={() => setExpandedCode(expandedCode === 'encapsulation' ? null : 'encapsulation')}>
                {expandedCode === 'encapsulation' ? 'Hide Security Benefits' : 'Show Security Benefits'}
              </button>
              {expandedCode === 'encapsulation' && (
                <div className="code-explanation">
                  <h5>🛡️ Security and Control Benefits:</h5>
                  <ul>
                    <li><strong>Data Protection:</strong> Balance cannot be directly modified externally</li>
                    <li><strong>Validation:</strong> Deposit/withdrawal methods include validation logic</li>
                    <li><strong>Consistency:</strong> All balance changes go through controlled methods</li>
                    <li><strong>Audit Trail:</strong> All transactions are automatically recorded</li>
                    <li><strong>Interface Stability:</strong> Internal changes don't affect external code</li>
                  </ul>
                </div>
              )}
            </div>

            <div className="concept-deep-dive">
              <h3>🔐 Python Access Modifiers</h3>
              <div className="data-types-grid">
                <div className="data-type-card">
                  <h4>🟢 Public</h4>
                  <p>Accessible from anywhere</p>
                  <code>self.name = "Alice"</code>
                  <div className="real-example">Like your name - publicly visible</div>
                </div>
                <div className="data-type-card">
                  <h4>🟡 Protected (_)</h4>
                  <p>Convention - intended for internal use</p>
                  <code>self._account_number</code>
                  <div className="real-example">Like family information - shared with related classes</div>
                </div>
                <div className="data-type-card">
                  <h4>🔴 Private (__)</h4>
                  <p>Name mangling - harder to access externally</p>
                  <code>self.__balance</code>
                  <div className="real-example">Like your PIN - should stay completely private</div>
                </div>
                <div className="data-type-card">
                  <h4>🎯 Properties</h4>
                  <p>Controlled access through getter/setter methods</p>
                  <code>@property</code>
                  <div className="real-example">Like a gated community - controlled entry</div>
                </div>
              </div>
            </div>
          </div>
        )

      case 'magic-methods':
        return (
          <div className="section-content">
            <h2>✨ Magic Methods (Dunder Methods)</h2>
            
            <div className="intro-section">
              <h3>What are Magic Methods?</h3>
              <p>
                Magic methods (also called dunder methods - double underscore) are special methods in Python that allow you to define 
                how your objects behave with built-in functions and operators. They start and end with double underscores (__).
              </p>
            </div>

            <div className="analogy-box">
              <h4>🎭 Theater Performance Analogy</h4>
              <p>
                Magic methods are like special cues that tell Python how your objects should "perform" in different situations. 
                Just like actors respond to different director's cues (enter, exit, speak louder), objects respond to different operations (print, add, compare).
              </p>
            </div>

            <div className="code-example">
              <h4>Magic Methods in Action: Book Class</h4>
              <pre>{codeExamples.magicMethods}</pre>
              <button onClick={() => setExpandedCode(expandedCode === 'magicMethods' ? null : 'magicMethods')}>
                {expandedCode === 'magicMethods' ? 'Hide Magic Methods List' : 'Show Common Magic Methods'}
              </button>
              {expandedCode === 'magicMethods' && (
                <div className="code-explanation">
                  <h5>🎩 Common Magic Methods:</h5>
                  <div className="data-types-grid">
                    <div className="data-type-card">
                      <h6>Object Creation</h6>
                      <code>__init__, __new__, __del__</code>
                    </div>
                    <div className="data-type-card">
                      <h6>String Representation</h6>
                      <code>__str__, __repr__, __format__</code>
                    </div>
                    <div className="data-type-card">
                      <h6>Arithmetic Operations</h6>
                      <code>__add__, __sub__, __mul__, __div__</code>
                    </div>
                    <div className="data-type-card">
                      <h6>Comparison Operations</h6>
                      <code>__eq__, __lt__, __gt__, __ne__</code>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        )

      case 'design-patterns':
        return (
          <div className="section-content">
            <h2>🏛️ Design Patterns in Python</h2>
            
            <div className="intro-section">
              <h3>What are Design Patterns?</h3>
              <p>
                Design patterns are proven solutions to common programming problems. They're like architectural blueprints 
                that can be customized to solve recurring design problems in object-oriented programming.
              </p>
            </div>

            <div className="analogy-box">
              <h4>🏗️ Architecture Patterns Analogy</h4>
              <p>
                Just like architects have standard patterns for building design (open floor plan, split-level, colonial), 
                software developers have standard patterns for code organization. These patterns have been tested and refined over time.
              </p>
            </div>

            <div className="code-example">
              <h4>Common Design Patterns in Python</h4>
              <pre>{codeExamples.designPatterns}</pre>
            </div>

            <div className="concept-deep-dive">
              <h3>🎯 Pattern Categories</h3>
              <div className="data-types-grid">
                <div className="data-type-card">
                  <h4>🏗️ Creational Patterns</h4>
                  <p>Deal with object creation mechanisms</p>
                  <div className="real-example">Singleton, Factory, Builder, Prototype</div>
                </div>
                <div className="data-type-card">
                  <h4>🔧 Structural Patterns</h4>
                  <p>Deal with object composition and relationships</p>
                  <div className="real-example">Adapter, Decorator, Facade, Composite</div>
                </div>
                <div className="data-type-card">
                  <h4>🎬 Behavioral Patterns</h4>
                  <p>Deal with communication between objects</p>
                  <div className="real-example">Observer, Strategy, Command, State</div>
                </div>
                <div className="data-type-card">
                  <h4>🐍 Pythonic Patterns</h4>
                  <p>Python-specific patterns and idioms</p>
                  <div className="real-example">Context Managers, Decorators, Generators</div>
                </div>
              </div>
            </div>
          </div>
        )

      default:
        return <div>Select a topic from the sidebar to begin learning!</div>
    }
  }

  return (
    <div className="page">
      <div className="learning-container">
        <div className="sidebar">
          <h3>🐍 Python OOP Complete Guide</h3>
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
            <p>Section {sections.findIndex(s => s.id === activeSection) + 1} of {sections.length}</p>
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
              ← Previous
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
              Next →
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

export default PythonOOP