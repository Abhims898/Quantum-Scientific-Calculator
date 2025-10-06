let qcData = [];

document.getElementById("mode").addEventListener("change", function() {
    const mode = this.value;
    document.getElementById("quantum-controls").style.display = (mode === "quantum") ? "block" : "none";
});

function addGate(gate) {
    let qubits = prompt("Enter qubit indices (comma separated, starting from 0):");
    if(!qubits) return;
    let qArray = qubits.split(",").map(x => parseInt(x.trim()));
    qcData.push({ gate: gate, qubits: qArray });
    alert(`${gate.toUpperCase()} gate added on qubits: ${qArray.join(", ")}`);
}

async function calculate() {
    const expression = document.getElementById("expression").value;
    const mode = document.getElementById("mode").value;

    let data = { expression, mode };
    if(mode === "quantum") {
        data.qc_data = qcData;
        data.num_qubits = parseInt(document.getElementById("num_qubits").value);
    }

    try {
        const response = await fetch("/calculate", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify(data)
        });

        const result = await response.json();
        document.getElementById("result").innerText = result.result || "No result.";

        const img = document.getElementById("quantum-plot");
        if(result.plot){
            img.src = result.plot;
            img.style.display = "block";
        } else {
            img.style.display = "none";
        }

        if(mode === "quantum") qcData = [];
    } catch(err) {
        document.getElementById("result").innerText = "Error: " + err.message;
    }
}
