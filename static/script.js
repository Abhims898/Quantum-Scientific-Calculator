const modeSelect = document.getElementById("mode");
const actionSelect = document.getElementById("action");
const quantumData = document.getElementById("quantumData");
const numQubits = document.getElementById("num_qubits");
const qcPlot = document.getElementById("qc_plot");

modeSelect.addEventListener("change", () => {
    if(modeSelect.value === "memory"){
        actionSelect.style.display = "inline";
        quantumData.style.display = "none";
        numQubits.style.display = "none";
    } else if(modeSelect.value === "quantum") {
        quantumData.style.display = "block";
        numQubits.style.display = "inline";
        actionSelect.style.display = "none";
    } else {
        actionSelect.style.display = "none";
        quantumData.style.display = "none";
        numQubits.style.display = "none";
    }
});

function calculate() {
    const expression = document.getElementById("expression").value;
    const mode = modeSelect.value;
    const action = actionSelect.value;

    let payload = { expression, mode, action };

    if(mode === "quantum") {
        try {
            payload.qc_data = JSON.parse(quantumData.value || "[]");
            payload.num_qubits = parseInt(numQubits.value) || 1;
        } catch(e) {
            document.getElementById("result").innerText = "Invalid quantum JSON";
            return;
        }
    }

    fetch("/calculate", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payload)
    })
    .then(res => res.json())
    .then(data => {
        document.getElementById("result").innerText = data.result;
        if(data.plot){
            qcPlot.src = data.plot;
            qcPlot.style.display = "block";
        } else {
            qcPlot.style.display = "none";
        }
    });
}
