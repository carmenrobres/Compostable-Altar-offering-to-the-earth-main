const VARS = ["humidity","ec","n","ph","p","k"];

const $ = (id) => document.getElementById(id);

$("uploadBtn").onclick = async () => {
  const f = $("fileInput").files[0];
  if (!f) {
    alert("Pick a CSV first.");
    return;
  }

  const form = new FormData();
  form.append("file", f);

  $("status").textContent = "Uploading...";
  const res = await fetch("/upload", { method: "POST", body: form });
  const data = await res.json();

  if (!res.ok) {
    $("status").textContent = "";
    alert(data.error || "Upload failed.");
    return;
  }

  $("status").textContent = "Upload ok.";
};

$("predictBtn").onclick = async () => {
  const model = $("model").value;
  const horizon = $("horizon").value;

  // Clear previous plots before running new forecast
  VARS.forEach(v => {
    const div = document.getElementById("plot-" + v);
    if (div) div.innerHTML = "";
  });

  $("status").textContent = "Running forecast...";
  const res = await fetch("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model, horizon })
  });
  const data = await res.json();
  if (!res.ok) {
    $("status").textContent = "";
    alert(data.error || "Predict failed.");
    return;
  }
  $("status").textContent = "Done.";

  renderAll(data);
  // --- Show error metrics if available ---
   // --- Show error metrics if available ---
    if (data.error && Object.keys(data.error).length > 0) {
    const err = data.error;
    let errText = `${data.model.toUpperCase()} errors: `;
    if (err.RMSE) errText += `RMSE: ${err.RMSE.toFixed(3)}  `;
    if (err.MAE) errText += `MAE: ${err.MAE.toFixed(3)}  `;
    if (err.MAPE) errText += `MAPE: ${err.MAPE.toFixed(2)}%`;

    let box = document.getElementById("error-box");
    if (!box) {
        box = document.createElement("div");
        box.id = "error-box";
        box.style.cssText = `
        margin:10px 0;
        padding:6px 12px;
        border:1px solid #ccc;
        border-radius:6px;
        background:#f8f8f8;
        font-size:14px;
        color:#333;
        `;
        document.body.insertBefore(box, document.getElementById("plots"));
    }
    box.textContent = errText;
    box.style.display = "block";
    } else {
    const box = document.getElementById("error-box");
    if (box) box.style.display = "none";
    }


};


function renderAll(payload) {
  console.log("Payload from backend:", payload);

  if (!payload || !payload.series) {
    alert("No data received.");
    return;
  }

  const x = payload.x || [];
  const series = payload.series || {};

  // Loop through each main variable (humidity, ec, n, ph, p, k)
  for (const v of VARS) {
    const divId = "plot-" + v;
    const traces = [];

    for (const [name, values] of Object.entries(series)) {
      const n = name.toLowerCase();

    const isMatch =
    (v === "humidity" && (n.includes("humidity") || n.includes("soil_humidity") || n.includes("humidity_soil"))) ||
    (v === "ec" && (n.includes("ec") || n.includes("soil_ec") || n.includes("ec_soil"))) ||
    (v === "n" && (n.includes("_n") || n.includes("soil_n") || n.includes("n_soil"))) ||
    (v === "ph" && (n.includes("_ph") || n.includes("soil_ph") || n.includes("ph_soil"))) ||
    (v === "p" && (/_p($|[^a-zA-Z])/.test(n) || /^p($|[^a-zA-Z])/.test(n)) && !/ph/i.test(n)) ||
    (v === "k" && (n.includes("_k") || n.includes("soil_k") || n.includes("k_soil")));


      if (isMatch) {
        traces.push({
          x: x,
          y: values,
          mode: "lines",
          name: name
        });
      }
    }

    if (traces.length > 0) {
      Plotly.newPlot(divId, traces, {
        title: v.toUpperCase(),
        xaxis: { title: "Time" },
        yaxis: { title: "Value" },
        margin: { t: 40, r: 10, l: 45, b: 40 }
      });
    } else {
      document.getElementById(divId).innerHTML = `<p>No data for ${v}</p>`;
    }
  }
}




