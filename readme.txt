Quantum Scientific Calculator ⚛
================================

Project Overview
----------------
This project is a web-based Quantum Scientific Calculator that performs arithmetic, scientific, and complex calculations using simulated quantum algorithms through Qiskit.

It provides a clean and modern GUI built with HTML, CSS, and JavaScript (running in a Flask backend).  
All mathematical operations are evaluated through Python’s AST parsing and a simulated quantum backend using single-qubit rotation circuits.

Features
--------
1. Quantum-based arithmetic operations (+, -, ×, ÷, ^)
2. Scientific functions: sin, cos, tan, sqrt, log, ln, exp, abs
3. Complex number calculations (e.g., complex(3,4))
4. Dual angle modes: Degrees and Radians
5. Quantum-only mode using Qiskit backend
6. Scrollable calculation history
7. Attractive responsive GUI with glowing quantum theme

Project Structure
-----------------
|-- app.py              -> Flask backend and quantum evaluation logic  
|-- templates/
|     └── calculator.html -> Front-end user interface   
|-- requirements.txt      -> Required Python packages  
|-- README.txt            -> Project documentation (this file)

Installation Instructions
-------------------------
1. Clone this repository:
   git clone https://github.com/Abhims898/Quantum-Scientific-Calculator.git
   cd quantum-scientific-calculator

2. Create a Python virtual environment (recommended):
   python3 -m venv venv
   source venv/bin/activate    (Linux/macOS)
   venv\Scripts\activate       (Windows)

3. Install dependencies:
   pip install -r requirements.txt

   If Qiskit is not installed, manually install it:
   pip install qiskit

4. Run the Flask application:
   python app.py

5. Open your browser and navigate to:
   http://127.0.0.1:5000/

Requirements
------------
- Python 3.8 or higher
- Flask
- Qiskit (for quantum simulation)
- NumPy

Example Usage
-------------
Expression:  sin(60) [deg]
Result:      0.866025

Expression:  ln(100)
Result:      4.60517

Expression:  complex(3,4)
Result:      3.0 + 4.0j

Quantum Backend
---------------
The calculator uses Qiskit's QuantumCircuit and Statevector to simulate single-qubit RY rotations.  
Each numeric value is encoded as a qubit rotation angle.  
Mathematical results are decoded using qubit expectation values to mimic quantum computation.

If Qiskit is not installed, the app automatically falls back to a classical simulation mode.

Notes
-----
- ln(x) computes the natural logarithm (base e)
- log(x) computes the logarithm (base e) for consistency with quantum operations
- Angle mode (degrees/radians) affects all trigonometric functions
- The history panel automatically scrolls and can be cleared anytime

License
-------
This project is open-source and available under the MIT License.

Author
------
Developed by: Abhishek  
GitHub: https://github.com/Abhims898/Quantum-Scientific-Calculator 
Year: 2025
