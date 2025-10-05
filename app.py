from flask import Flask, render_template, request, jsonify
from qiskit import QuantumCircuit, Aer, execute
import math
import cmath
import numpy as np

app = Flask(__name__)

# Memory storage
memory = 0

# --- Quantum Arithmetic Functions ---
def quantum_adder(a, b, n_bits=8):
    qc = QuantumCircuit(n_bits*2 + 1, n_bits*2 + 1)
    for i in range(n_bits):
        if (a >> i) & 1: qc.x(i)
        if (b >> i) & 1: qc.x(i+n_bits)
    for i in range(n_bits):
        qc.cx(i, i+n_bits)
        qc.ccx(i, i+n_bits, n_bits*2)
    qc.measure(range(n_bits*2 + 1), range(n_bits*2 + 1))
    simulator = Aer.get_backend('qasm_simulator')
    result = execute(qc, simulator, shots=1).result()
    counts = result.get_counts()
    sum_bin = list(counts.keys())[0][::-1]
    return int(sum_bin, 2)

def quantum_subtractor(a, b):
    return quantum_adder(a, (2**8 - b)) % (2**8)

def quantum_multiplier(a, b):
    result = 0
    for _ in range(b):
        result = quantum_adder(result, a)
    return result

def quantum_divider(a, b):
    if b == 0: return "Error: Division by zero"
    q, r = 0, a
    while r >= b:
        r -= b
        q = quantum_adder(q, 1)
    return q, r

# --- Scientific Functions ---
def quantum_scientific(func, value):
    try:
        if func == 'sin': return round(math.sin(math.radians(value)), 6)
        if func == 'cos': return round(math.cos(math.radians(value)), 6)
        if func == 'tan': return round(math.tan(math.radians(value)), 6)
        if func == 'log': return round(math.log(value), 6)
        if func == 'exp': return round(math.exp(value), 6)
        if func == 'sqrt': return round(math.sqrt(value), 6)
    except Exception as e:
        return str(e)

# --- Complex Number Operations ---
def complex_operation(a, b, op):
    try:
        a_c = complex(a.replace('i','j'))
        b_c = complex(b.replace('i','j'))
        if op == "add": return str(a_c + b_c)
        if op == "subtract": return str(a_c - b_c)
        if op == "multiply": return str(a_c * b_c)
        if op == "divide":
            if b_c == 0: return "Error: Division by zero"
            return str(a_c / b_c)
    except Exception as e:
        return str(e)

# --- Matrix Operations ---
def matrix_operation(matA, matB, op):
    try:
        A = np.array(matA, dtype=float)
        B = np.array(matB, dtype=float)
        if A.shape != B.shape and op=="add":
            return "Error: Matrices must be the same size for addition"
        if op == "add": return (A + B).tolist()
        if op == "multiply":
            if A.shape[1] != B.shape[0]:
                return "Error: Invalid dimensions for multiplication"
            return np.dot(A,B).tolist()
    except Exception as e:
        return str(e)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/calculate", methods=["POST"])
def calculate():
    global memory
    data = request.json
    operation = data.get("operation")
    a = data.get("a")
    b = data.get("b")
    value = data.get("value")
    matA = data.get("matA")
    matB = data.get("matB")
    
    # Memory Functions
    if operation == "M+": memory += float(value); return jsonify(result=memory)
    if operation == "M-": memory -= float(value); return jsonify(result=memory)
    if operation == "MR": return jsonify(result=memory)

    # Arithmetic
    if operation == "add": return jsonify(result=quantum_adder(int(a),int(b)))
    if operation == "subtract": return jsonify(result=quantum_subtractor(int(a),int(b)))
    if operation == "multiply": return jsonify(result=quantum_multiplier(int(a),int(b)))
    if operation == "divide":
        q,r = quantum_divider(int(a),int(b))
        return jsonify(result=f"Quotient: {q}, Remainder: {r}")

    # Scientific
    if operation in ["sin","cos","tan","log","exp","sqrt"]:
        return jsonify(result=quantum_scientific(operation, float(value)))

    # Complex numbers
    if operation in ["cadd","csub","cmul","cdiv"]:
        mapping = {"cadd":"add","csub":"subtract","cmul":"multiply","cdiv":"divide"}
        return jsonify(result=complex_operation(a,b,mapping[operation]))

    # Matrix
    if operation in ["madd","mmul"]:
        mapping = {"madd":"add","mmul":"multiply"}
        return jsonify(result=matrix_operation(matA, matB, mapping[operation]))

    return jsonify(result="Invalid Operation")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
