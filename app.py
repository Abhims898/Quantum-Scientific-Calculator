from flask import Flask, render_template, request, jsonify
import numpy as np
from qiskit import QuantumCircuit, execute
from qiskit_aer import Aer

app = Flask(__name__)

# Memory storage
memory = 0

# Helper for complex numbers
def parse_complex(val):
    try:
        return complex(val)
    except:
        return float(val)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/calculate", methods=["POST"])
def calculate():
    global memory
    data = request.json
    op = data.get("operation")
    a = data.get("a")
    b = data.get("b")
    value = data.get("value")
    matA = data.get("matA")
    matB = data.get("matB")

    try:
        # Memory functions
        if op == "M+":
            memory += parse_complex(value)
            return jsonify(result=memory)
        if op == "M-":
            memory -= parse_complex(value)
            return jsonify(result=memory)
        if op == "MR":
            return jsonify(result=memory)

        # Basic calculations
        if op in ["+", "-", "*", "/", "^"]:
            a = parse_complex(a)
            b = parse_complex(b)
            if op == "+": res = a + b
            if op == "-": res = a - b
            if op == "*": res = a * b
            if op == "/": res = a / b
            if op == "^": res = a ** b
            return jsonify(result=res)

        # Scientific functions
        a = parse_complex(a)
        if op == "sin": res = np.sin(a)
        if op == "cos": res = np.cos(a)
        if op == "tan": res = np.tan(a)
        if op == "log": res = np.log(a)
        if op == "sqrt": res = np.sqrt(a)
        return jsonify(result=res)

        # Matrix operations
        if matA and matB:
            matA = np.array(matA)
            matB = np.array(matB)
            if op == "mat_add": res = matA + matB
            if op == "mat_sub": res = matA - matB
            if op == "mat_mul": res = np.matmul(matA, matB)
            return jsonify(result=res.tolist())

        # Quantum simulation (1 qubit example)
        if op == "quantum":
            qc = QuantumCircuit(1,1)
            qc.h(0)
            qc.measure(0,0)
            backend = Aer.get_backend('qasm_simulator')
            job = execute(qc, backend, shots=1024)
            counts = job.result().get_counts()
            return jsonify(result=counts)

        return jsonify(result="Invalid operation")

    except Exception as e:
        return jsonify(result=str(e))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
