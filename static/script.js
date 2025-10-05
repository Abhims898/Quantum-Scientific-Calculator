// Quantum Animation
function animateQuantum(){
    const canvas = document.getElementById('quantumCanvas');
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0,0,canvas.width,canvas.height);
    for(let i=0;i<8;i++){
        ctx.beginPath();
        ctx.arc(50+i*60,50,20,0,2*Math.PI);
        ctx.fillStyle = `rgba(0,255,255,${Math.random()})`;
        ctx.shadowBlur=20; ctx.shadowColor="#0ff"; ctx.fill();
    }
}
setInterval(animateQuantum,500);

// Matrix Editor Functions
function addRow(matrixId){
    const table = document.getElementById(matrixId);
    const cols = table.rows[0] ? table.rows[0].cells.length : 2;
    const row = table.insertRow();
    for(let i=0;i<cols;i++){
        const cell = row.insertCell();
        const input = document.createElement("input");
        input.type="text"; input.value="0";
        cell.appendChild(input);
    }
}
function addCol(matrixId){
    const table = document.getElementById(matrixId);
    const rows = table.rows.length || 2;
    if(rows==0){ addRow(matrixId); addRow(matrixId); return; }
    for(let i=0;i<rows;i++){
        const cell = table.rows[i].insertCell();
        const input = document.createElement("input");
        input.type="text"; input.value="0";
        cell.appendChild(input);
    }
}
function getMatrixValues(matrixId){
    const table = document.getElementById(matrixId);
    const matrix=[];
    for(let r=0;r<table.rows.length;r++){
        const row=[];
        for(let c=0;c<table.rows[r].cells.length;c++){
            row.push(parseFloat(table.rows[r].cells[c].firstChild.value || 0));
        }
        matrix.push(row);
    }
    return matrix;
}

// Calculator
async function calculate(operation){
    const a = document.getElementById("a").value;
    const b = document.getElementById("b").value;
    const value = document.getElementById("value").value;

    let matA=null, matB=null;
    if(operation.includes("mat")){
        matA = getMatrixValues('matrixA');
        matB = getMatrixValues('matrixB');
    }

    const response = await fetch("/calculate", {
        method:"POST",
        headers:{"Content-Type":"application/json"},
        body: JSON.stringify({operation,a,b,value,matA,matB})
    });

    const data = await response.json();
    document.getElementById("result").innerText = JSON.stringify(data.result);

    // Add to history
    const li = document.createElement("li");
    li.textContent = `${operation}: ${JSON.stringify(data.result)}`;
    document.getElementById("historyList").prepend(li);

    animateQuantum();
}
