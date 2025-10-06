from flask import Flask, render_template, request, jsonify
import numpy as np
import sympy as sp
from qiskit import QuantumCircuit, Aer, execute
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

memory = 0

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/calculate", methods=["POST"])
def calculate():
    global memory
    data = request.json
    expression = data.get("expression")
    mode = data.get("mode")

    try:
        result = ""
        plot_url = None

        if mode == "scientific":
            result = str(eval(expression, {"__builtins__": None}, np.__dict__))
        elif mode == "complex":
            result = str(complex(expression))
        elif mode == "matrix":
            mat = np.array(eval(expression))
            result = str(mat)
        elif mode == "quantum":
            qc_data = data.get("qc_data", [])
            num_qubits = int(data.get("num_qubits", 1))
            qc = QuantumCircuit(num_qubits, num_qubits)

            # Apply gates
            for gate in qc_data:
                g = gate.get("gate").lower()
                q = gate.get("qubits", [0])
                if g == "h": [qc.h(qubit) for qubit in q]
                elif g == "x": [qc.x(qubit) for qubit in q]
                elif g == "y": [qc.y(qubit) for qubit in q]
                elif g == "z": [qc.z(qubit) for qubit in q]
                elif g == "cx": qc.cx(q[0], q[1])
                elif g == "ccx": qc.ccx(q[0], q[1], q[2])
                elif g == "measure": qc.measure_all()

            simulator = Aer.get_backend("qasm_simulator")
            job = execute(qc, simulator, shots=1024)
            counts = job.result().get_counts()
            result = str(counts)

            # Generate circuit image
            fig = qc.draw(output="mpl")
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            plot_url = "data:image/png;base64," + base64.b64encode(buf.read()).decode()
            buf.close()

        elif mode == "memory":
            action = data.get("action")
            if action == "M+":
                memory += float(expression)
                result = str(memory)
            elif action == "M-":
                memory -= float(expression)
                result = str(memory)
            elif action == "MR":
                result = str(memory)
        else:
            result = "Invalid mode"

        return jsonify({"result": result, "plot": plot_url})

    except Exception as e:
        return jsonify({"result": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
