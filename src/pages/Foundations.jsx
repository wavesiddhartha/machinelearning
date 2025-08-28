function Foundations() {
  return (
    <div className="page">
      <div className="content">
        <h1>🌱 Complete Foundations: Understanding Everything from Zero</h1>
        
        <section className="intro-section">
          <h2>🎯 What You'll Understand by the End</h2>
          <p>This isn't just about learning to code. You'll understand <strong>HOW and WHY</strong> everything works, from the ground up. We'll answer questions like:</p>
          <ul>
            <li>What is a computer and how does it actually work?</li>
            <li>What is programming and why do we need it?</li>
            <li>What are numbers, data, and information?</li>
            <li>How do computers store and process information?</li>
            <li>What is mathematics and why is it everywhere in computing?</li>
            <li>What are algorithms and why are they important?</li>
          </ul>
        </section>

        <section className="what-is-computer">
          <h2>💻 What is a Computer? (The Foundation of Everything)</h2>
          
          <div className="concept-deep-dive">
            <h3>🤔 Let's Start with the Absolute Basics</h3>
            <p>A computer is fundamentally a machine that can:</p>
            <ol>
              <li><strong>Store information</strong> (memory)</li>
              <li><strong>Process information</strong> (calculations)</li>
              <li><strong>Follow instructions</strong> (programs)</li>
              <li><strong>Communicate</strong> (input/output)</li>
            </ol>
            
            <h4>🧠 Think of it Like a Human Brain</h4>
            <p>Just like humans:</p>
            <ul>
              <li><strong>Memory:</strong> We remember things (like your phone number)</li>
              <li><strong>Processing:</strong> We think and make decisions (like calculating tips)</li>
              <li><strong>Instructions:</strong> We follow recipes or directions</li>
              <li><strong>Communication:</strong> We see, hear, speak, and write</li>
            </ul>
            
            <div className="analogy-box">
              <h4>🏭 The Factory Analogy</h4>
              <p>Imagine a computer as a super-efficient factory:</p>
              <table className="analogy-table">
                <thead>
                  <tr>
                    <th>Factory Part</th>
                    <th>Computer Part</th>
                    <th>What It Does</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>🏢 Warehouse</td>
                    <td>Memory (RAM/Storage)</td>
                    <td>Stores materials/data</td>
                  </tr>
                  <tr>
                    <td>⚙️ Assembly Line</td>
                    <td>CPU (Processor)</td>
                    <td>Does the actual work</td>
                  </tr>
                  <tr>
                    <td>📋 Instructions Manual</td>
                    <td>Programs/Software</td>
                    <td>Step-by-step directions</td>
                  </tr>
                  <tr>
                    <td>🚪 Loading Dock</td>
                    <td>Input/Output</td>
                    <td>Receives/sends materials</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </section>

        <section className="binary-and-data">
          <h2>🔢 How Computers Actually Store and Process Information</h2>
          
          <h3>💡 The Binary Secret</h3>
          <p>Here's something amazing: Computers only understand TWO things - ON and OFF (like a light switch). That's it!</p>
          
          <div className="binary-explanation">
            <h4>🔍 Why Only ON and OFF?</h4>
            <p>Think about electricity:</p>
            <ul>
              <li>⚡ <strong>ON (1):</strong> Electricity flows</li>
              <li>🔌 <strong>OFF (0):</strong> No electricity</li>
            </ul>
            
            <h4>🌟 The Magic: How 0s and 1s Become Everything</h4>
            <p>Just like how we can create any number using only 10 digits (0-9), computers create everything using only 2 digits (0,1)!</p>
            
            <div className="binary-examples">
              <h5>📊 Binary Number Examples:</h5>
              <table className="binary-table">
                <thead>
                  <tr>
                    <th>Decimal (What We Use)</th>
                    <th>Binary (What Computers Use)</th>
                    <th>How It Works</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>0</td>
                    <td>0</td>
                    <td>Nothing</td>
                  </tr>
                  <tr>
                    <td>1</td>
                    <td>1</td>
                    <td>One thing</td>
                  </tr>
                  <tr>
                    <td>2</td>
                    <td>10</td>
                    <td>One group of 2, zero ones</td>
                  </tr>
                  <tr>
                    <td>3</td>
                    <td>11</td>
                    <td>One group of 2, one extra</td>
                  </tr>
                  <tr>
                    <td>4</td>
                    <td>100</td>
                    <td>One group of 4, zero 2s, zero ones</td>
                  </tr>
                  <tr>
                    <td>8</td>
                    <td>1000</td>
                    <td>One group of 8</td>
                  </tr>
                </tbody>
              </table>
            </div>
            
            <h4>🎯 Real Example: How Your Name is Stored</h4>
            <p>When you type your name, each letter gets converted to binary:</p>
            <div className="name-example">
              <p><strong>Letter 'A':</strong></p>
              <ul>
                <li>Computer assigns 'A' the number 65</li>
                <li>65 in binary is: 01000001</li>
                <li>So when you type 'A', computer stores: 01000001</li>
              </ul>
              
              <p><strong>Your name "HELLO":</strong></p>
              <div className="binary-word">
                <div className="letter-binary">
                  <div className="letter">H</div>
                  <div className="binary-code">01001000</div>
                </div>
                <div className="letter-binary">
                  <div className="letter">E</div>
                  <div className="binary-code">01000101</div>
                </div>
                <div className="letter-binary">
                  <div className="letter">L</div>
                  <div className="binary-code">01001100</div>
                </div>
                <div className="letter-binary">
                  <div className="letter">L</div>
                  <div className="binary-code">01001100</div>
                </div>
                <div className="letter-binary">
                  <div className="letter">O</div>
                  <div className="binary-code">01001111</div>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section className="what-is-programming">
          <h2>💭 What is Programming? (The Art of Giving Instructions)</h2>
          
          <h3>🎯 Programming is Just Detailed Instructions</h3>
          <p>Programming is telling a computer exactly what to do, step by step. Think of it like giving directions to someone who has never been to your house.</p>
          
          <div className="programming-analogy">
            <h4>🏠 The House Directions Analogy</h4>
            <div className="direction-comparison">
              <div className="human-directions">
                <h5>👤 Directions for Humans:</h5>
                <p>"Go to my house"</p>
                <p><em>Humans understand context and can figure things out</em></p>
              </div>
              
              <div className="computer-directions">
                <h5>🤖 Directions for Computers:</h5>
                <ol>
                  <li>Start your car</li>
                  <li>Put car in drive</li>
                  <li>Drive straight for 0.5 miles</li>
                  <li>Turn right at the traffic light</li>
                  <li>Drive straight for 0.3 miles</li>
                  <li>Turn left into the driveway with number 123</li>
                  <li>Stop the car</li>
                  <li>Turn off the engine</li>
                </ol>
                <p><em>Computers need EVERY detail spelled out</em></p>
              </div>
            </div>
          </div>
          
          <h3>🔧 Why Do We Need Programming Languages?</h3>
          <p>Remember, computers only understand 0s and 1s (binary). But writing everything in 0s and 1s would be impossible for humans!</p>
          
          <div className="language-levels">
            <h4>📚 The Language Hierarchy</h4>
            
            <div className="language-level">
              <h5>🤖 Machine Language (What Computer Understands)</h5>
              <code>10110000 01100001 10110011 00000001 ...</code>
              <p><em>Pure binary - impossible for humans to read</em></p>
            </div>
            
            <div className="language-level">
              <h5>⚙️ Assembly Language (Slightly Better)</h5>
              <code>MOV AL, 61h<br/>CMP AL, 01h<br/>JE end</code>
              <p><em>Still very difficult and specific to each processor</em></p>
            </div>
            
            <div className="language-level">
              <h5>🐍 High-Level Language (Python - Human Friendly!)</h5>
              <code>name = "Hello"<br/>print(name)</code>
              <p><em>Easy to read and understand - almost like English!</em></p>
            </div>
          </div>
          
          <h4>🌉 How Python Becomes Machine Code</h4>
          <div className="translation-process">
            <div className="step">
              <div className="step-number">1</div>
              <div className="step-content">
                <h6>You Write Python</h6>
                <code>print("Hello World")</code>
              </div>
            </div>
            <div className="arrow">⬇️</div>
            <div className="step">
              <div className="step-number">2</div>
              <div className="step-content">
                <h6>Python Interpreter Translates</h6>
                <p>Converts your code to instructions</p>
              </div>
            </div>
            <div className="arrow">⬇️</div>
            <div className="step">
              <div className="step-number">3</div>
              <div className="step-content">
                <h6>Computer Executes</h6>
                <code>01001000 01100101 01101100...</code>
              </div>
            </div>
          </div>
        </section>

        <section className="what-is-data">
          <h2>📊 What is Data? Understanding Information</h2>
          
          <h3>🔍 Data vs Information vs Knowledge</h3>
          <div className="data-hierarchy">
            <div className="data-level">
              <h4>📝 Data (Raw Facts)</h4>
              <p>Individual pieces of information without context</p>
              <div className="examples">
                <span className="data-item">25</span>
                <span className="data-item">°C</span>
                <span className="data-item">Tuesday</span>
                <span className="data-item">John</span>
                <span className="data-item">8:30</span>
              </div>
            </div>
            
            <div className="data-level">
              <h4>ℹ️ Information (Processed Data)</h4>
              <p>Data with context and meaning</p>
              <div className="examples">
                <p>"The temperature is 25°C on Tuesday"</p>
                <p>"John arrived at 8:30"</p>
              </div>
            </div>
            
            <div className="data-level">
              <h4>🧠 Knowledge (Understanding)</h4>
              <p>Information combined with experience and insights</p>
              <div className="examples">
                <p>"25°C on Tuesday means it's a warm day, good for outdoor activities"</p>
                <p>"John arriving at 8:30 means he was 30 minutes late for the 8:00 meeting"</p>
              </div>
            </div>
          </div>
          
          <h3>🗂️ Types of Data in Computing</h3>
          <div className="data-types-grid">
            <div className="data-type-card">
              <h4>🔢 Numbers</h4>
              <ul>
                <li><strong>Integers:</strong> 1, 2, 100, -5</li>
                <li><strong>Decimals:</strong> 3.14, -0.5, 2.718</li>
                <li><strong>Why Important:</strong> Math, calculations, measurements</li>
              </ul>
              <div className="real-example">
                <strong>Real Use:</strong> Bank account balance, age, temperature
              </div>
            </div>
            
            <div className="data-type-card">
              <h4>📝 Text</h4>
              <ul>
                <li><strong>Characters:</strong> 'A', '7', '@', ' '</li>
                <li><strong>Strings:</strong> "Hello", "user@email.com"</li>
                <li><strong>Why Important:</strong> Names, addresses, messages</li>
              </ul>
              <div className="real-example">
                <strong>Real Use:</strong> Your name, email, social media posts
              </div>
            </div>
            
            <div className="data-type-card">
              <h4>✅ True/False</h4>
              <ul>
                <li><strong>Boolean:</strong> True or False</li>
                <li><strong>Examples:</strong> Is logged in? Has paid? Is online?</li>
                <li><strong>Why Important:</strong> Decisions and conditions</li>
              </ul>
              <div className="real-example">
                <strong>Real Use:</strong> Login status, payment confirmation, game state
              </div>
            </div>
            
            <div className="data-type-card">
              <h4>📋 Collections</h4>
              <ul>
                <li><strong>Lists:</strong> [1, 2, 3], ["apple", "banana"]</li>
                <li><strong>Groups:</strong> Multiple related items</li>
                <li><strong>Why Important:</strong> Store many things together</li>
              </ul>
              <div className="real-example">
                <strong>Real Use:</strong> Shopping cart items, friend list, search results
              </div>
            </div>
          </div>
        </section>

        <section className="why-mathematics">
          <h2>🧮 Why Mathematics is Everywhere in Computing</h2>
          
          <h3>🤔 "But I'm Bad at Math!" - Let's Fix That Mindset</h3>
          <p>Here's the truth: You use math every day without realizing it!</p>
          
          <div className="daily-math">
            <h4>🏠 Math You Already Do Daily:</h4>
            <div className="math-examples">
              <div className="math-example">
                <h5>🛒 Shopping</h5>
                <p><strong>Math involved:</strong> Addition (total cost), percentages (discounts), comparison (which is cheaper?)</p>
                <p><strong>Computing connection:</strong> E-commerce algorithms, price optimization</p>
              </div>
              
              <div className="math-example">
                <h5>🍳 Cooking</h5>
                <p><strong>Math involved:</strong> Ratios (recipe scaling), time (cooking duration), temperature</p>
                <p><strong>Computing connection:</strong> Algorithm optimization, machine learning ratios</p>
              </div>
              
              <div className="math-example">
                <h5>🚗 Driving</h5>
                <p><strong>Math involved:</strong> Speed, distance, time, fuel efficiency</p>
                <p><strong>Computing connection:</strong> GPS algorithms, route optimization</p>
              </div>
              
              <div className="math-example">
                <h5>🎮 Games</h5>
                <p><strong>Math involved:</strong> Scores, levels, probabilities, strategies</p>
                <p><strong>Computing connection:</strong> Game algorithms, AI decision making</p>
              </div>
            </div>
          </div>
          
          <h3>🔧 How Math Powers Modern Technology</h3>
          <div className="tech-math-examples">
            <div className="tech-example">
              <h4>🔍 Google Search</h4>
              <p><strong>Math Behind It:</strong></p>
              <ul>
                <li>Algorithms rank billions of web pages</li>
                <li>Statistics determine relevance</li>
                <li>Linear algebra processes massive data</li>
              </ul>
              <p><strong>Simple Explanation:</strong> Like finding the best restaurant among thousands by considering reviews, distance, and your preferences</p>
            </div>
            
            <div className="tech-example">
              <h4>📱 Phone Camera</h4>
              <p><strong>Math Behind It:</strong></p>
              <ul>
                <li>Matrices process image data</li>
                <li>Calculus optimizes focus</li>
                <li>Statistics reduce noise</li>
              </ul>
              <p><strong>Simple Explanation:</strong> Like automatically adjusting a painting to make it look perfect</p>
            </div>
            
            <div className="tech-example">
              <h4>🎵 Music Streaming</h4>
              <p><strong>Math Behind It:</strong></p>
              <ul>
                <li>Algorithms recommend songs</li>
                <li>Statistics analyze listening patterns</li>
                <li>Probability predicts preferences</li>
              </ul>
              <p><strong>Simple Explanation:</strong> Like a friend who knows your music taste and suggests new songs</p>
            </div>
          </div>
        </section>

        <section className="what-are-algorithms">
          <h2>🧩 What are Algorithms? (The Heart of Everything)</h2>
          
          <h3>🎯 Algorithm = Recipe for Solving Problems</h3>
          <p>An algorithm is just a step-by-step procedure for solving a problem. You use algorithms every day!</p>
          
          <div className="algorithm-examples">
            <h4>🏠 Daily Life Algorithms:</h4>
            
            <div className="daily-algorithm">
              <h5>☕ Making Coffee Algorithm</h5>
              <ol>
                <li>Fill kettle with water</li>
                <li>Turn on kettle</li>
                <li>While water is heating:
                  <ul>
                    <li>Get coffee mug</li>
                    <li>Add coffee grounds to filter</li>
                  </ul>
                </li>
                <li>When water boils, pour over coffee</li>
                <li>Wait 4 minutes</li>
                <li>Enjoy coffee!</li>
              </ol>
              <p><strong>Key concepts:</strong> Sequence (order matters), conditions (when water boils), loops (while water is heating)</p>
            </div>
            
            <div className="daily-algorithm">
              <h5>🛒 Shopping Algorithm</h5>
              <ol>
                <li>Make shopping list</li>
                <li>For each item on list:
                  <ul>
                    <li>Find item in store</li>
                    <li>Check price</li>
                    <li>If price is acceptable, add to cart</li>
                    <li>Else, look for alternative</li>
                  </ul>
                </li>
                <li>Go to checkout</li>
                <li>Pay for items</li>
              </ol>
              <p><strong>Key concepts:</strong> Loops (for each item), decisions (if price acceptable), data (shopping list)</p>
            </div>
          </div>
          
          <h3>💻 Computer Algorithms</h3>
          <div className="computer-algorithms">
            <h4>🔍 Search Algorithm (Finding Your Photo)</h4>
            <p><strong>Problem:</strong> Find a specific photo among 10,000 photos on your phone</p>
            
            <div className="search-methods">
              <div className="search-method">
                <h5>🐌 Slow Way (Linear Search)</h5>
                <ol>
                  <li>Start with first photo</li>
                  <li>Is this the photo I want?</li>
                  <li>If yes: Found it!</li>
                  <li>If no: Go to next photo</li>
                  <li>Repeat until found</li>
                </ol>
                <p><strong>Time:</strong> Could take looking through all 10,000 photos!</p>
              </div>
              
              <div className="search-method">
                <h5>⚡ Fast Way (Smart Organization)</h5>
                <ol>
                  <li>Organize photos by date</li>
                  <li>Remember approximate date of photo</li>
                  <li>Jump to that date section</li>
                  <li>Look around that area</li>
                </ol>
                <p><strong>Time:</strong> Might only need to check 50 photos!</p>
              </div>
            </div>
          </div>
          
          <h3>🧠 Why Algorithm Thinking Matters</h3>
          <div className="algorithm-benefits">
            <div className="benefit">
              <h4>🎯 Problem Solving</h4>
              <p>Break big problems into small, manageable steps</p>
            </div>
            <div className="benefit">
              <h4>⚡ Efficiency</h4>
              <p>Find the best way to do something (faster, cheaper, better)</p>
            </div>
            <div className="benefit">
              <h4>🔧 Automation</h4>
              <p>Teach computers to do repetitive tasks for us</p>
            </div>
            <div className="benefit">
              <h4>🧩 Logic</h4>
              <p>Think clearly and systematically about any problem</p>
            </div>
          </div>
        </section>

        <section className="putting-it-together">
          <h2>🎯 Putting It All Together: The Big Picture</h2>
          
          <div className="big-picture">
            <h3>🌟 How Everything Connects</h3>
            
            <div className="connection-flow">
              <div className="connection-step">
                <h4>1️⃣ Computer Hardware</h4>
                <p>Physical machine with memory, processor, input/output</p>
                <div className="step-detail">Provides the foundation - like having a brain and hands</div>
              </div>
              
              <div className="flow-arrow">⬇️</div>
              
              <div className="connection-step">
                <h4>2️⃣ Binary/Data</h4>
                <p>How information is stored and represented</p>
                <div className="step-detail">The language computers understand - like having a vocabulary</div>
              </div>
              
              <div className="flow-arrow">⬇️</div>
              
              <div className="connection-step">
                <h4>3️⃣ Programming Languages</h4>
                <p>Human-friendly way to give instructions</p>
                <div className="step-detail">The bridge between human thoughts and machine actions</div>
              </div>
              
              <div className="flow-arrow">⬇️</div>
              
              <div className="connection-step">
                <h4>4️⃣ Algorithms</h4>
                <p>Step-by-step solutions to problems</p>
                <div className="step-detail">The logic and reasoning - like having a methodology</div>
              </div>
              
              <div className="flow-arrow">⬇️</div>
              
              <div className="connection-step">
                <h4>5️⃣ Mathematics</h4>
                <p>The patterns and relationships that make algorithms efficient</p>
                <div className="step-detail">The optimization and intelligence behind the solutions</div>
              </div>
              
              <div className="flow-arrow">⬇️</div>
              
              <div className="connection-step">
                <h4>6️⃣ Real Applications</h4>
                <p>Solving actual human problems</p>
                <div className="step-detail">The end goal - making life better and easier</div>
              </div>
            </div>
          </div>
        </section>

        <section className="next-steps">
          <h2>🚀 Your Learning Journey Starts Here</h2>
          
          <div className="journey-overview">
            <p>Now that you understand the foundations, you're ready to dive deep into each area:</p>
            
            <div className="learning-path-preview">
              <div className="path-step">
                <h4>🌱 Next: Computer Science Fundamentals</h4>
                <p>Dive deeper into how computers work, memory, processors, and systems</p>
              </div>
              
              <div className="path-step">
                <h4>🧮 Then: Mathematical Foundations</h4>
                <p>Build the math skills you need, starting from basic arithmetic to advanced concepts</p>
              </div>
              
              <div className="path-step">
                <h4>💻 Programming Concepts</h4>
                <p>Learn the fundamental programming concepts that work in any language</p>
              </div>
              
              <div className="path-step">
                <h4>🐍 Python Mastery</h4>
                <p>Master Python from basics to advanced topics with real projects</p>
              </div>
              
              <div className="path-step">
                <h4>🤖 AI & Machine Learning</h4>
                <p>Apply everything you've learned to build intelligent systems</p>
              </div>
            </div>
          </div>
          
          <div className="encouragement">
            <h3>💪 Remember</h3>
            <ul>
              <li>Every expert was once a beginner</li>
              <li>Understanding beats memorization</li>
              <li>Practice makes permanent</li>
              <li>Questions are signs of intelligence, not ignorance</li>
            </ul>
          </div>
        </section>
      </div>
    </div>
  )
}

export default Foundations