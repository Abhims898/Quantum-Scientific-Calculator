async function calculate(operation) {
    const a = document.getElementById("a").value;
    const b = document.getElementById("b").value;
    const value = document.getElementById("value").value;
    let matA = null, matB = null;
    try {
        matA = document.getElementById("matA").value ? JSON.parse(document.getElementById("matA").value) : null;
        matB = document.getElementById("matB").value ? JSON.parse(document.getElementById("matB").value) : null;
    } catch(e) {
        alert("Invalid matrix format! Use JSON array like [[1,2],[3,4]]");
        return;
    }

    const response = await fetch("/calculate", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({operation, a, b, value, matA, matB})
    });

    const data = await response.json();
    const result = data.result;
    document.getElementById("result").innerText = result;

    // Add to history
    const historyList = document.getElementById("historyList");
    const li = document.createElement("li");
    li.textContent = `${operation}: ${result}`;
    historyList.prepend(li);

    // Animate quantum canvas
    animateQuantum();
}

// Quantum Canvas Animation
function animateQuantum() {
    const canvas = document.getElementById('quantumCanvas');
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    for(let i=0;i<8;i++){
        ctx.beginPath();
        ctx.arc(50 + i*60, 50, 20, 0, 2*Math.PI);
        ctx.fillStyle = `rgba(0, 255, 255, ${Math.random()})`;
        ctx.shadowBlur = 20;
        ctx.shadowColor = '#0ff';
        ctx.fill();
    }
}

setInterval(animateQuantum, 500);
