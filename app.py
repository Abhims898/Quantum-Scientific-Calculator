from flask import Flask, render_template, request, jsonify
import ast, math, cmath, traceback
import numpy as np

# ------------ Quantum setup ------------
USE_QUANTUM = True
try:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    QUANTUM_READY = True
except Exception:
    QUANTUM_READY = False
    USE_QUANTUM = False

SCALE = 0.12  # scaling factor for encoding numeric values into rotation angles

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True

history = []

# ---------- Quantum primitive helpers ----------
def _state_after_ry(theta: float):
    theta = float(theta)
    if USE_QUANTUM and QUANTUM_READY:
        qc = QuantumCircuit(1)
        qc.ry(theta, 0)
        sv = Statevector.from_instruction(qc)
        return np.array(sv.data, dtype=complex)
    else:
        # fallback: simulate statevector manually
        return np.array([math.cos(theta / 2) + 0j, math.sin(theta / 2) + 0j])

def _expectations_from_sv(sv):
    a, b = sv[0], sv[1]
    exp_z = (abs(a) ** 2) - (abs(b) ** 2)
    exp_x = 2 * (a.conjugate() * b).real
    return float(exp_z), float(exp_x)

def _theta_from_expectations(exp_z, exp_x):
    return math.atan2(exp_x, exp_z)

def encode_decode_through_qubit(value: float):
    """Encode numeric value through simulated qubit rotation and decode back."""
    theta = SCALE * float(value)
    sv = _state_after_ry(theta)
    ez, ex = _expectations_from_sv(sv)
    theta_rec = _theta_from_expectations(ez, ex)
    decoded = theta_rec / SCALE
    return round(decoded, 6)

# ---------- Quantum-based operations ----------
def q_add(a: float, b: float):
    theta = SCALE * (float(a) + float(b))
    sv = _state_after_ry(theta)
    ez, ex = _expectations_from_sv(sv)
    return round(_theta_from_expectations(ez, ex) / SCALE, 6)

def q_sub(a: float, b: float):
    theta = SCALE * (float(a) - float(b))
    sv = _state_after_ry(theta)
    ez, ex = _expectations_from_sv(sv)
    return round(_theta_from_expectations(ez, ex) / SCALE, 6)

def q_sin(x: float):
    sv = _state_after_ry(x)
    ez, ex = _expectations_from_sv(sv)
    return round(ex, 6)

def q_cos(x: float):
    sv = _state_after_ry(x)
    ez, ex = _expectations_from_sv(sv)
    return round(ez, 6)

# ---------- Serialization helper ----------
def serialize_result(v):
    if isinstance(v, complex):
        return f"{round(v.real,6)} + {round(v.imag,6)}j"
    if isinstance(v, (np.ndarray,)):
        return [serialize_result(x) for x in v]
    if isinstance(v, (np.generic, float, int)):
        return round(float(v), 6)
    return v

# ---------- AST evaluator ----------
ALLOWED_NAMES = {"pi": math.pi, "e": math.e}

def eval_node(node, vars_env, mode):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Num):
        return node.n
    if isinstance(node, ast.Name):
        if node.id in vars_env:
            return vars_env[node.id]
        if node.id in ALLOWED_NAMES:
            return ALLOWED_NAMES[node.id]
        raise ValueError(f"Unknown name '{node.id}'")
    if isinstance(node, ast.UnaryOp):
        val = eval_node(node.operand, vars_env, mode)
        return +val if isinstance(node.op, ast.UAdd) else -val
    if isinstance(node, ast.BinOp):
        left = eval_node(node.left, vars_env, mode)
        right = eval_node(node.right, vars_env, mode)

        if isinstance(left, complex) or isinstance(right, complex):
            if isinstance(node.op, ast.Add): c = left + right
            elif isinstance(node.op, ast.Sub): c = left - right
            elif isinstance(node.op, ast.Mult): c = left * right
            elif isinstance(node.op, ast.Div): c = left / right
            elif isinstance(node.op, ast.Pow): c = left ** right
            else: raise ValueError("Unsupported complex op")
            return complex(encode_decode_through_qubit(c.real), encode_decode_through_qubit(c.imag))

        if isinstance(node.op, ast.Add): return q_add(float(left), float(right))
        if isinstance(node.op, ast.Sub): return q_sub(float(left), float(right))
        if isinstance(node.op, ast.Mult): return encode_decode_through_qubit(float(left) * float(right))
        if isinstance(node.op, ast.Div): return encode_decode_through_qubit(float(left) / float(right))
        if isinstance(node.op, ast.Pow): return encode_decode_through_qubit(float(left) ** float(right))
        raise ValueError("Unsupported binary op")

    if isinstance(node, ast.Call):
        fname = node.func.id if isinstance(node.func, ast.Name) else None
        args = [eval_node(a, vars_env, mode) for a in node.args]

        def conv(x): return math.radians(float(x)) if mode == "deg" else float(x)

        if fname == "sin": return q_sin(conv(args[0]))
        if fname == "cos": return q_cos(conv(args[0]))
        if fname == "tan":
            sx, cx = q_sin(conv(args[0])), q_cos(conv(args[0]))
            return round(sx / cx, 6)
        if fname == "sqrt": return encode_decode_through_qubit(math.sqrt(float(args[0])))
        if fname == "ln": return encode_decode_through_qubit(math.log(float(args[0])))      # Natural log
        if fname == "log": return encode_decode_through_qubit(math.log10(float(args[0])))   # Base-10 log
        if fname == "exp": return encode_decode_through_qubit(math.exp(float(args[0])))
        if fname == "abs": return encode_decode_through_qubit(abs(args[0]))
        if fname == "complex": return complex(args[0], args[1] if len(args) > 1 else 0)
        raise ValueError(f"Function '{fname}' not allowed")

    raise ValueError(f"Unsupported node: {type(node)}")

# ---------- Evaluation handler ----------
def evaluate_code(code_text: str, mode="deg"):
    lines = [ln for ln in code_text.splitlines() if ln.strip()]
    if not lines:
        return ""
    vars_env = {}
    last_expr = None
    for ln in lines:
        if "=" in ln and not ln.strip().startswith("#"):
            left, right = ln.split("=", 1)
            name = left.strip()
            node = ast.parse(right, mode="eval")
            vars_env[name] = eval_node(node.body, vars_env, mode)
        else:
            last_expr = ln
    if last_expr is None:
        if vars_env:
            return list(vars_env.values())[-1]
        return ""
    node = ast.parse(last_expr, mode="eval")
    return eval_node(node.body, vars_env, mode)

# ---------- Flask endpoints ----------
@app.route("/")
def index():
    msg = ""
    if not (USE_QUANTUM and QUANTUM_READY):
        msg = "⚠️ Quantum backend not available. Install qiskit & qiskit-aer."
    return render_template("calculator.html", quantum=(USE_QUANTUM and QUANTUM_READY), error_msg=msg)

@app.route("/calculate", methods=["POST"])
def calculate():
    data = request.get_json() or {}
    expr = data.get("expression", "")
    mode = data.get("mode", "deg")  # degree or radian mode
    try:
        val = evaluate_code(expr, mode)
        ser = serialize_result(val)
        history.append({"expr": f"{expr} [{mode}]", "result": ser})
        if len(history) > 300:
            history.pop(0)
        return jsonify({"ok": True, "result": ser})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "traceback": traceback.format_exc()})

@app.route("/history")
def get_history():
    return jsonify(list(reversed(history)))

@app.route("/clear_history", methods=["POST"])
def clear_history():
    history.clear()
    return jsonify({"ok": True})

if __name__ == "__main__":
    app.run(debug=True)
