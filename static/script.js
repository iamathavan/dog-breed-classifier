const uploadBox  = document.getElementById('uploadBox');
const fileInput  = document.getElementById('fileInput');
const preview    = document.getElementById('preview');
const placeholder = document.getElementById('placeholder');
const predictBtn = document.getElementById('predictBtn');
const results    = document.getElementById('results');
const resultCards = document.getElementById('resultCards');
const loader     = document.getElementById('loader');

let selectedFile = null;

uploadBox.addEventListener('click', () => fileInput.click());

uploadBox.addEventListener('dragover', e => {
  e.preventDefault();
  uploadBox.style.borderColor = '#4299e1';
});
uploadBox.addEventListener('dragleave', () => {
  uploadBox.style.borderColor = '#cbd5e0';
});
uploadBox.addEventListener('drop', e => {
  e.preventDefault();
  handleFile(e.dataTransfer.files[0]);
});

fileInput.addEventListener('change', () => handleFile(fileInput.files[0]));

function handleFile(file) {
  if (!file || !file.type.startsWith('image/')) return;
  selectedFile = file;

  const reader = new FileReader();
  reader.onload = e => {
    preview.src = e.target.result;
    preview.hidden = false;
    placeholder.hidden = true;
  };
  reader.readAsDataURL(file);
  predictBtn.disabled = false;
  results.hidden = true;
}

predictBtn.addEventListener('click', async () => {
  if (!selectedFile) return;

  loader.hidden = false;
  predictBtn.disabled = true;
  results.hidden = true;
  resultCards.innerHTML = '';

  const formData = new FormData();
  formData.append('file', selectedFile);

  try {
    const res = await fetch('/predict', { method: 'POST', body: formData });
    const data = await res.json();

    if (data.error) {
      alert('Error: ' + data.error);
      return;
    }

    data.predictions.forEach((pred, i) => {
      const card = document.createElement('div');
      card.className = 'card' + (i === 0 ? ' top' : '');
      card.innerHTML = `
        <div style="flex:1">
          <div style="display:flex; justify-content:space-between">
            <span class="breed-name">${i === 0 ? '🏆 ' : ''}${pred.breed}</span>
            <span class="confidence">${pred.confidence}%</span>
          </div>
          <div class="bar-bg">
            <div class="bar-fill" style="width:${pred.confidence}%"></div>
          </div>
        </div>
      `;
      resultCards.appendChild(card);
    });

    results.hidden = false;
  } catch (err) {
    alert('Something went wrong. Is the server running?');
  } finally {
    loader.hidden = true;
    predictBtn.disabled = false;
  }
});