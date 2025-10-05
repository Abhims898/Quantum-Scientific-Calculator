from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
import sympy as sp
from qiskit import QuantumCircuit, execute, Aer
from qiskit.visualization import plot_histogram, circuit_drawer
import matplotlib.pyplot as plt
import io

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
    mode = data.get("mode")  # scientific, quantum, matrix, complex, memory

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
            qc_data = data.get("qc_data", [])  # list of dicts {gate, qubits}
            num_qubits = data.get("num_qubits", 1)
            qc = QuantumCircuit(num_qubits, num_qubits)

            # Apply gates
            for gate in qc_data:
                g = gate.get("gate")
                q = gate.get("qubits", [0])
                if g.lower() == "h":
                    for qubit in q: qc.h(qubit)
                elif g.lower() == "x":
                    for qubit in q: qc.x(qubit)
                elif g.lower() == "y":
                    for qubit in q: qc.y(qubit)
                elif g.lower() == "z":
                    for qubit in q: qc.z(qubit)
                elif g.lower() == "cx":
                    qc.cx(q[0], q[1])
                elif g.lower() == "ccx":
                    qc.ccx(q[0], q[1], q[2])
                elif g.lower() == "measure":
                    for qubit in q: qc.measure(qubit, qubit)

            simulator = Aer.get_backend("aer_simulator")
            job = execute(qc, simulator, shots=1024)
            counts = job.result().get_counts()
            result = str(counts)

            # Generate circuit image
            fig = qc.draw(output='mpl')
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            plot_url = "data:image/png;base64," + \
                str(io.BytesIO(buf.read()).getvalue().hex())
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
