function ComputerScience() {
  return (
    <div className="page">
      <div className="content">
        <h1>💻 Complete Computer Science: How Everything Actually Works</h1>
        
        <section className="introduction">
          <h2>🎯 What You'll Master</h2>
          <p>By the end of this section, you'll understand computers at a level that 95% of people never reach. You'll know:</p>
          <ul>
            <li>How electricity becomes computation</li>
            <li>Why computers are so fast and reliable</li>
            <li>How your code turns into actual machine operations</li>
            <li>Why some programs are fast and others are slow</li>
            <li>How the internet actually works</li>
            <li>How operating systems manage everything</li>
          </ul>
        </section>

        <section className="from-transistors-to-computation">
          <h2>⚡ From Electricity to Intelligence: The Complete Journey</h2>
          
          <h3>🔬 Level 1: Transistors - The Building Blocks</h3>
          <div className="transistor-explanation">
            <p>Everything starts with a <strong>transistor</strong> - a tiny electronic switch that can be ON or OFF.</p>
            
            <div className="transistor-analogy">
              <h4>💡 The Light Switch Analogy</h4>
              <div className="analogy-comparison">
                <div className="light-switch">
                  <h5>🏠 Home Light Switch</h5>
                  <ul>
                    <li>ON: Light bulb glows</li>
                    <li>OFF: Light bulb is dark</li>
                    <li>You control it manually</li>
                  </ul>
                </div>
                <div className="transistor-switch">
                  <h5>⚡ Electronic Transistor</h5>
                  <ul>
                    <li>ON: Current flows (represents 1)</li>
                    <li>OFF: No current (represents 0)</li>
                    <li>Controlled by electrical signals</li>
                  </ul>
                </div>
              </div>
            </div>
            
            <div className="scale-visualization">
              <h4>📏 The Mind-Blowing Scale</h4>
              <p>Your phone has over <strong>11 BILLION transistors</strong>. To put this in perspective:</p>
              <ul>
                <li>🌟 More transistors than there are stars visible to the naked eye</li>
                <li>🌍 More than the world's population</li>
                <li>🏠 Like having 11 billion tiny light switches in your pocket</li>
                <li>⚡ Each switching billions of times per second</li>
              </ul>
            </div>
          </div>
          
          <h3>🚪 Level 2: Logic Gates - Making Decisions</h3>
          <div className="logic-gates">
            <p>Combine transistors in clever ways, and you get <strong>logic gates</strong> - circuits that can make decisions.</p>
            
            <div className="gate-examples">
              <div className="logic-gate">
                <h4>🚪 AND Gate</h4>
                <p><strong>Rule:</strong> Output is ON only if BOTH inputs are ON</p>
                <div className="gate-table">
                  <table>
                    <thead>
                      <tr><th>Input A</th><th>Input B</th><th>Output</th><th>Real Life Example</th></tr>
                    </thead>
                    <tbody>
                      <tr><td>OFF</td><td>OFF</td><td>OFF</td><td>No key AND no password = Can't login</td></tr>
                      <tr><td>OFF</td><td>ON</td><td>OFF</td><td>No key but have password = Can't login</td></tr>
                      <tr><td>ON</td><td>OFF</td><td>OFF</td><td>Have key but no password = Can't login</td></tr>
                      <tr><td>ON</td><td>ON</td><td>ON</td><td>Have key AND password = Can login! ✅</td></tr>
                    </tbody>
                  </table>
                </div>
              </div>
              
              <div className="logic-gate">
                <h4>🔄 OR Gate</h4>
                <p><strong>Rule:</strong> Output is ON if EITHER input is ON</p>
                <div className="gate-table">
                  <table>
                    <thead>
                      <tr><th>Input A</th><th>Input B</th><th>Output</th><th>Real Life Example</th></tr>
                    </thead>
                    <tbody>
                      <tr><td>OFF</td><td>OFF</td><td>OFF</td><td>No cash AND no card = Can't buy</td></tr>
                      <tr><td>OFF</td><td>ON</td><td>ON</td><td>No cash but have card = Can buy! ✅</td></tr>
                      <tr><td>ON</td><td>OFF</td><td>ON</td><td>Have cash but no card = Can buy! ✅</td></tr>
                      <tr><td>ON</td><td>ON</td><td>ON</td><td>Have both cash AND card = Can buy! ✅</td></tr>
                    </tbody>
                  </table>
                </div>
              </div>
              
              <div className="logic-gate">
                <h4>🚫 NOT Gate</h4>
                <p><strong>Rule:</strong> Output is opposite of input</p>
                <div className="gate-table">
                  <table>
                    <thead>
                      <tr><th>Input</th><th>Output</th><th>Real Life Example</th></tr>
                    </thead>
                    <tbody>
                      <tr><td>OFF</td><td>ON</td><td>Door unlocked → Security system ON</td></tr>
                      <tr><td>ON</td><td>OFF</td><td>Door locked → Security system OFF</td></tr>
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>
          
          <h3>🧮 Level 3: Arithmetic Logic Unit (ALU) - The Calculator Brain</h3>
          <div className="alu-explanation">
            <p>Combine many logic gates, and you can build an <strong>ALU</strong> - the part of the computer that does math!</p>
            
            <div className="alu-example">
              <h4>➕ How Computers Add Numbers</h4>
              <p>Let's see how a computer adds 3 + 2 = 5:</p>
              
              <div className="binary-addition">
                <div className="addition-step">
                  <h5>Step 1: Convert to Binary</h5>
                  <div className="conversion">
                    <span>3 in decimal = </span><span className="binary">011 in binary</span>
                    <br/>
                    <span>2 in decimal = </span><span className="binary">010 in binary</span>
                  </div>
                </div>
                
                <div className="addition-step">
                  <h5>Step 2: Binary Addition</h5>
                  <div className="binary-math">
                    <pre>{`  011  (3 in binary)
+ 010  (2 in binary)
-----
  101  (5 in binary)`}</pre>
                  </div>
                </div>
                
                <div className="addition-step">
                  <h5>Step 3: Logic Gates Do the Work</h5>
                  <p>Each digit position uses XOR gates, AND gates, and carry logic to compute the result!</p>
                </div>
              </div>
            </div>
          </div>
          
          <h3>💾 Level 4: Memory - Storing Information</h3>
          <div className="memory-explanation">
            <h4>🏠 Memory is Like a Giant Apartment Building</h4>
            <div className="memory-analogy">
              <div className="apartment-building">
                <h5>🏢 Apartment Building</h5>
                <ul>
                  <li>📍 Each apartment has an address (like 5B)</li>
                  <li>🚪 You can store things in each apartment</li>
                  <li>📝 Building manager keeps track of what's where</li>
                  <li>🔑 You need the address to find your stuff</li>
                </ul>
              </div>
              <div className="computer-memory">
                <h5>💾 Computer Memory</h5>
                <ul>
                  <li>📍 Each memory location has an address (like 0x4A2B)</li>
                  <li>💾 You can store data in each location</li>
                  <li>🖥️ Operating system manages what's where</li>
                  <li>🔑 Programs use addresses to find data</li>
                </ul>
              </div>
            </div>
            
            <div className="memory-types">
              <h4>🗂️ Types of Memory</h4>
              
              <div className="memory-type">
                <h5>⚡ RAM (Random Access Memory) - The Workshop</h5>
                <ul>
                  <li><strong>Speed:</strong> Super fast (nanoseconds)</li>
                  <li><strong>Size:</strong> Medium (8-32 GB typical)</li>
                  <li><strong>Persistence:</strong> Temporary (loses data when power off)</li>
                  <li><strong>Analogy:</strong> Your workshop desk - everything you're working on right now</li>
                  <li><strong>Cost:</strong> Expensive per GB</li>
                </ul>
              </div>
              
              <div className="memory-type">
                <h5>💿 Storage (SSD/Hard Drive) - The Warehouse</h5>
                <ul>
                  <li><strong>Speed:</strong> Slower (milliseconds)</li>
                  <li><strong>Size:</strong> Large (256 GB - 8 TB)</li>
                  <li><strong>Persistence:</strong> Permanent (keeps data when power off)</li>
                  <li><strong>Analogy:</strong> Your storage warehouse - long-term storage</li>
                  <li><strong>Cost:</strong> Cheap per GB</li>
                </ul>
              </div>
              
              <div className="memory-type">
                <h5>🚀 Cache - The Pocket</h5>
                <ul>
                  <li><strong>Speed:</strong> Extremely fast (picoseconds)</li>
                  <li><strong>Size:</strong> Tiny (1-32 MB)</li>
                  <li><strong>Persistence:</strong> Temporary</li>
                  <li><strong>Analogy:</strong> Your pocket - things you need instantly</li>
                  <li><strong>Cost:</strong> Very expensive per GB</li>
                </ul>
              </div>
            </div>
          </div>
          
          <h3>🧠 Level 5: The CPU - The Brain</h3>
          <div className="cpu-explanation">
            <h4>🏭 CPU is Like a Super-Efficient Factory</h4>
            
            <div className="cpu-components">
              <div className="cpu-component">
                <h5>📋 Control Unit - The Factory Manager</h5>
                <ul>
                  <li>Reads instructions from memory</li>
                  <li>Decides what needs to be done</li>
                  <li>Coordinates all other parts</li>
                  <li>Like a foreman directing workers</li>
                </ul>
              </div>
              
              <div className="cpu-component">
                <h5>🧮 ALU - The Worker</h5>
                <ul>
                  <li>Does all the actual calculations</li>
                  <li>Handles logical operations</li>
                  <li>Processes data according to instructions</li>
                  <li>Like the assembly line worker</li>
                </ul>
              </div>
              
              <div className="cpu-component">
                <h5>📁 Registers - The Desktop</h5>
                <ul>
                  <li>Tiny, super-fast storage</li>
                  <li>Holds data currently being worked on</li>
                  <li>Like having papers spread on your desk</li>
                  <li>Fastest memory in the computer</li>
                </ul>
              </div>
            </div>
            
            <div className="instruction-cycle">
              <h4>🔄 The Instruction Cycle (How CPU Works)</h4>
              <p>The CPU does this billions of times per second:</p>
              
              <div className="cycle-steps">
                <div className="cycle-step">
                  <div className="step-number">1</div>
                  <div className="step-content">
                    <h6>📥 Fetch</h6>
                    <p>Get the next instruction from memory</p>
                    <div className="example">"Get instruction: ADD 5, 3"</div>
                  </div>
                </div>
                
                <div className="cycle-step">
                  <div className="step-number">2</div>
                  <div className="step-content">
                    <h6>🔍 Decode</h6>
                    <p>Figure out what the instruction means</p>
                    <div className="example">"This means add two numbers"</div>
                  </div>
                </div>
                
                <div className="cycle-step">
                  <div className="step-number">3</div>
                  <div className="step-content">
                    <h6>⚡ Execute</h6>
                    <p>Actually do the operation</p>
                    <div className="example">"5 + 3 = 8"</div>
                  </div>
                </div>
                
                <div className="cycle-step">
                  <div className="step-number">4</div>
                  <div className="step-content">
                    <h6>💾 Store</h6>
                    <p>Save the result</p>
                    <div className="example">"Store 8 in memory location X"</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section className="operating-systems">
          <h2>🖥️ Operating Systems: The Master Coordinator</h2>
          
          <h3>🎭 What is an Operating System?</h3>
          <div className="os-explanation">
            <p>An Operating System (OS) is like the <strong>manager of a busy restaurant</strong>:</p>
            
            <div className="restaurant-analogy">
              <div className="restaurant-role">
                <h4>👨‍🍳 In a Restaurant</h4>
                <ul>
                  <li>🍽️ Seats customers at tables</li>
                  <li>📋 Takes orders and routes them to kitchen</li>
                  <li>⏰ Coordinates timing of multiple orders</li>
                  <li>🧑‍🍳 Assigns tasks to different chefs</li>
                  <li>💰 Handles payments and resources</li>
                  <li>🚪 Manages who can enter kitchen</li>
                </ul>
              </div>
              
              <div className="os-role">
                <h4>🖥️ Operating System</h4>
                <ul>
                  <li>💾 Allocates memory to programs</li>
                  <li>📨 Routes messages between programs</li>
                  <li>⏰ Schedules when programs run</li>
                  <li>🔄 Assigns CPU time to different tasks</li>
                  <li>📁 Manages files and storage</li>
                  <li>🔐 Controls security and permissions</li>
                </ul>
              </div>
            </div>
          </div>
          
          <h3>🔄 Process Management</h3>
          <div className="process-management">
            <h4>🎪 The Juggling Act</h4>
            <p>Your computer appears to run many programs at once, but it's actually <strong>juggling</strong> them super fast!</p>
            
            <div className="multitasking-explanation">
              <div className="time-slice">
                <h5>⏱️ Time Slicing</h5>
                <p>The OS gives each program tiny time slices (milliseconds):</p>
                <div className="time-example">
                  <div className="time-slot">Browser: 10ms</div>
                  <div className="time-slot">Music: 10ms</div>
                  <div className="time-slot">Game: 10ms</div>
                  <div className="time-slot">Browser: 10ms</div>
                  <div className="continue">...</div>
                </div>
                <p>Switches so fast you can't notice!</p>
              </div>
              
              <div className="scheduling">
                <h5>📅 Scheduling Priorities</h5>
                <ul>
                  <li><strong>🚨 Real-time:</strong> Music playback (can't have gaps!)</li>
                  <li><strong>⚡ High priority:</strong> User interface (must feel responsive)</li>
                  <li><strong>📊 Normal:</strong> Background calculations</li>
                  <li><strong>🐌 Low priority:</strong> File indexing, backups</li>
                </ul>
              </div>
            </div>
          </div>
          
          <h3>💾 Memory Management</h3>
          <div className="memory-management">
            <h4>🏢 Memory as an Office Building</h4>
            <p>The OS manages memory like a building manager assigns office space:</p>
            
            <div className="memory-allocation">
              <div className="allocation-example">
                <h5>📍 Virtual Memory Magic</h5>
                <p>Each program thinks it has the entire computer to itself!</p>
                
                <div className="virtual-example">
                  <div className="program-view">
                    <h6>🎮 Game's Perspective</h6>
                    <p>"I have addresses 0-1000 to use"</p>
                  </div>
                  <div className="program-view">
                    <h6>🌐 Browser's Perspective</h6>
                    <p>"I have addresses 0-1000 to use"</p>
                  </div>
                  <div className="reality">
                    <h6>🖥️ Reality (OS manages)</h6>
                    <p>Game actually uses 2000-3000<br/>
                       Browser actually uses 4000-5000</p>
                  </div>
                </div>
              </div>
              
              <div className="protection">
                <h5>🛡️ Memory Protection</h5>
                <p>Programs can't access each other's memory:</p>
                <ul>
                  <li>🔒 Each program has its own "sandbox"</li>
                  <li>🚫 Can't read other programs' data</li>
                  <li>💥 If program crashes, others keep running</li>
                  <li>🔐 Security prevents malicious access</li>
                </ul>
              </div>
            </div>
          </div>
        </section>

        <section className="how-internet-works">
          <h2>🌐 How the Internet Actually Works</h2>
          
          <h3>📮 The Internet is Like a Global Postal System</h3>
          <div className="internet-postal-analogy">
            <div className="postal-system">
              <h4>📬 Traditional Mail</h4>
              <ol>
                <li>✍️ Write letter with recipient's address</li>
                <li>📪 Put in mailbox</li>
                <li>🚛 Local post office collects</li>
                <li>✈️ Routes through distribution centers</li>
                <li>🚚 Local delivery to destination</li>
                <li>📭 Arrives at recipient's mailbox</li>
              </ol>
            </div>
            
            <div className="internet-system">
              <h4>🌐 Internet Communication</h4>
              <ol>
                <li>💻 Create data packet with IP address</li>
                <li>📡 Send to your router</li>
                <li>🏢 Internet Service Provider (ISP) receives</li>
                <li>🔄 Routes through internet backbone</li>
                <li>🎯 Destination ISP delivers</li>
                <li>📱 Arrives at recipient's device</li>
              </ol>
            </div>
          </div>
          
          <h3>📦 Data Packets</h3>
          <div className="packet-explanation">
            <h4>🧩 Breaking Messages into Pieces</h4>
            <p>Large messages are split into small packets - like tearing a long letter into postcards:</p>
            
            <div className="packet-example">
              <div className="original-message">
                <h5>📝 Original Message</h5>
                <p>"Hey! How are you doing today? I hope you're having a great time!"</p>
              </div>
              
              <div className="packet-breakdown">
                <h5>📦 Split into Packets</h5>
                <div className="packet">
                  <div className="packet-header">Packet 1/3 | To: 192.168.1.100 | From: 192.168.1.50</div>
                  <div className="packet-data">"Hey! How are you doing today?"</div>
                </div>
                <div className="packet">
                  <div className="packet-header">Packet 2/3 | To: 192.168.1.100 | From: 192.168.1.50</div>
                  <div className="packet-data">"I hope you're having a"</div>
                </div>
                <div className="packet">
                  <div className="packet-header">Packet 3/3 | To: 192.168.1.100 | From: 192.168.1.50</div>
                  <div className="packet-data">"great time!"</div>
                </div>
              </div>
              
              <div className="packet-journey">
                <h5>🛣️ Different Routes</h5>
                <p>Each packet might take a different path to reach the destination!</p>
                <ul>
                  <li>Packet 1: Your house → ISP → West Coast → Destination</li>
                  <li>Packet 2: Your house → ISP → Central → Destination</li>
                  <li>Packet 3: Your house → ISP → East Coast → Destination</li>
                </ul>
                <p>They get reassembled at the destination in the correct order!</p>
              </div>
            </div>
          </div>
          
          <h3>🏠 IP Addresses and DNS</h3>
          <div className="ip-dns-explanation">
            <h4>🗺️ Addresses on the Internet</h4>
            
            <div className="address-comparison">
              <div className="physical-address">
                <h5>🏠 Physical Address</h5>
                <p>123 Main Street<br/>
                   Springfield, IL 62701<br/>
                   United States</p>
                <p><em>Unique location in physical world</em></p>
              </div>
              
              <div className="ip-address">
                <h5>🌐 IP Address</h5>
                <p>192.168.1.100</p>
                <p><em>Unique location on internet</em></p>
              </div>
            </div>
            
            <div className="dns-explanation">
              <h4>📞 DNS: The Internet's Phone Book</h4>
              <p>Just like you remember "Mom" instead of her phone number, DNS lets you use names instead of IP addresses:</p>
              
              <div className="dns-example">
                <div className="dns-lookup">
                  <div className="step">You type: "google.com"</div>
                  <div className="arrow">↓</div>
                  <div className="step">DNS looks up: "What's google.com's IP?"</div>
                  <div className="arrow">↓</div>
                  <div className="step">DNS returns: "172.217.12.142"</div>
                  <div className="arrow">↓</div>
                  <div className="step">Your browser connects to: 172.217.12.142</div>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section className="why-computers-fast">
          <h2>⚡ Why Are Computers So Fast?</h2>
          
          <h3>🏃‍♂️ Speed Comparison</h3>
          <div className="speed-comparison">
            <div className="human-vs-computer">
              <div className="human-speed">
                <h4>👤 Human</h4>
                <ul>
                  <li>🧮 Mental math: ~1 calculation per second</li>
                  <li>📖 Reading: ~200 words per minute</li>
                  <li>✍️ Writing: ~40 words per minute</li>
                  <li>🤔 Decision making: seconds to minutes</li>
                </ul>
              </div>
              
              <div className="computer-speed">
                <h4>💻 Computer</h4>
                <ul>
                  <li>🧮 Calculations: 3+ billion per second</li>
                  <li>📖 Reading: millions of words per second</li>
                  <li>✍️ Writing: millions of words per second</li>
                  <li>🤔 Decisions: billions per second</li>
                </ul>
              </div>
            </div>
          </div>
          
          <h3>🔑 Secrets to Computer Speed</h3>
          <div className="speed-secrets">
            <div className="speed-secret">
              <h4>⚡ 1. Electricity is Fast</h4>
              <p>Electricity travels at nearly the speed of light: <strong>300,000 km per second</strong></p>
              <div className="speed-example">
                <p>In the time it takes you to blink (0.3 seconds), electricity could travel around Earth 2,250 times!</p>
              </div>
            </div>
            
            <div className="speed-secret">
              <h4>🔄 2. Parallel Processing</h4>
              <p>Computers can do many things simultaneously:</p>
              <div className="parallel-example">
                <h5>🏭 Assembly Line Analogy</h5>
                <p><strong>Sequential (one at a time):</strong> Build entire car, then start next car</p>
                <p><strong>Parallel (assembly line):</strong> While installing wheels on car 1, someone else is painting car 2, and another person is installing engine in car 3!</p>
                <p>Result: Cars finish much faster!</p>
              </div>
            </div>
            
            <div className="speed-secret">
              <h4>💾 3. Smart Memory Hierarchy</h4>
              <p>Computers predict what data you'll need next:</p>
              <ul>
                <li>📚 Keep frequently used data in fast cache</li>
                <li>🔮 Predict patterns in your behavior</li>
                <li>🚀 Pre-load data before you ask for it</li>
              </ul>
            </div>
            
            <div className="speed-secret">
              <h4>🎯 4. Specialized Circuits</h4>
              <p>Different parts of the computer are optimized for specific tasks:</p>
              <ul>
                <li>🧮 Math circuits for calculations</li>
                <li>🎮 Graphics circuits for visual processing</li>
                <li>🔊 Audio circuits for sound processing</li>
                <li>🌐 Network circuits for internet communication</li>
              </ul>
            </div>
          </div>
        </section>

        <section className="putting-it-together">
          <h2>🧩 Putting It All Together: The Complete Picture</h2>
          
          <div className="complete-system">
            <h3>🎬 A Day in the Life of Your Computer</h3>
            <p>Let's trace what happens when you search for "cute cats" on Google:</p>
            
            <div className="system-trace">
              <div className="trace-step">
                <div className="step-number">1</div>
                <div className="step-detail">
                  <h4>⌨️ You Type</h4>
                  <p>Keyboard sends electrical signals representing "cute cats"</p>
                  <div className="tech-detail">Each letter becomes a binary code sent to CPU</div>
                </div>
              </div>
              
              <div className="trace-step">
                <div className="step-number">2</div>
                <div className="step-detail">
                  <h4>🖥️ OS Processes Input</h4>
                  <p>Operating system receives keystrokes, updates screen display</p>
                  <div className="tech-detail">Graphics card draws letters on screen using millions of pixels</div>
                </div>
              </div>
              
              <div className="trace-step">
                <div className="step-number">3</div>
                <div className="step-detail">
                  <h4>🌐 Browser Prepares Request</h4>
                  <p>Browser creates HTTP request packet with your search term</p>
                  <div className="tech-detail">CPU runs browser code, ALU processes text, memory stores data</div>
                </div>
              </div>
              
              <div className="trace-step">
                <div className="step-number">4</div>
                <div className="step-detail">
                  <h4>📡 Network Stack</h4>
                  <p>Data splits into packets, each gets Google's IP address</p>
                  <div className="tech-detail">DNS lookup converts "google.com" to IP address like 172.217.12.142</div>
                </div>
              </div>
              
              <div className="trace-step">
                <div className="step-number">5</div>
                <div className="step-detail">
                  <h4>🛣️ Internet Journey</h4>
                  <p>Packets travel through multiple routers to reach Google's servers</p>
                  <div className="tech-detail">Each router uses routing tables to find best path, like GPS for data</div>
                </div>
              </div>
              
              <div className="trace-step">
                <div className="step-number">6</div>
                <div className="step-detail">
                  <h4>🏢 Google's Servers</h4>
                  <p>Massive data centers process your search across millions of web pages</p>
                  <div className="tech-detail">Distributed computing: thousands of computers work together in parallel</div>
                </div>
              </div>
              
              <div className="trace-step">
                <div className="step-number">7</div>
                <div className="step-detail">
                  <h4>📤 Results Return</h4>
                  <p>Search results travel back through internet to your computer</p>
                  <div className="tech-detail">HTML, CSS, JavaScript code sent in packets, reassembled by your browser</div>
                </div>
              </div>
              
              <div className="trace-step">
                <div className="step-number">8</div>
                <div className="step-detail">
                  <h4>🎨 Display Results</h4>
                  <p>Your browser renders the web page with cute cat pictures!</p>
                  <div className="tech-detail">Graphics card processes image data, millions of transistors create the display</div>
                </div>
              </div>
            </div>
            
            <div className="amazing-facts">
              <h4>🤯 Mind-Blowing Facts About This Simple Search:</h4>
              <ul>
                <li>⏰ <strong>Time:</strong> All of this happens in less than 0.5 seconds</li>
                <li>🔢 <strong>Calculations:</strong> Billions of calculations across multiple computers</li>
                <li>🌍 <strong>Distance:</strong> Data might travel thousands of miles</li>
                <li>⚡ <strong>Electricity:</strong> Trillions of electrons flow through circuits</li>
                <li>🧠 <strong>Decisions:</strong> Millions of logical decisions made automatically</li>
              </ul>
            </div>
          </div>
        </section>

        <section className="next-steps">
          <h2>🚀 What's Next?</h2>
          
          <div className="next-learning">
            <p>Now that you understand how computers work at the fundamental level, you're ready to:</p>
            
            <div className="next-topics">
              <div className="next-topic">
                <h4>🧮 Master Mathematics</h4>
                <p>Understand the mathematical concepts that make all this computation possible</p>
              </div>
              
              <div className="next-topic">
                <h4>💻 Learn Programming</h4>
                <p>Write instructions that control this amazing machinery</p>
              </div>
              
              <div className="next-topic">
                <h4>🤖 Build AI Systems</h4>
                <p>Create programs that can learn and make intelligent decisions</p>
              </div>
            </div>
          </div>
          
          <div className="key-insights">
            <h3>💡 Key Insights to Remember</h3>
            <ul>
              <li>🔬 Everything starts with simple ON/OFF switches</li>
              <li>🧩 Complex behaviors emerge from simple rules</li>
              <li>⚡ Speed comes from doing simple things very, very fast</li>
              <li>🤝 Modern computing is all about coordination and cooperation</li>
              <li>🌟 You now understand more about computers than 99% of people!</li>
            </ul>
          </div>
        </section>
      </div>
    </div>
  )
}

export default ComputerScience