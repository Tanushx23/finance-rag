const uploadBtn = document.getElementById('uploadBtn');
const fileInput = document.getElementById('fileInput');
const fileLabelText = document.getElementById('fileLabelText');
const uploadStatus = document.getElementById('uploadStatus');
const questionInput = document.getElementById('questionInput');
const askBtn = document.getElementById('askBtn');
const queryStatus = document.getElementById('queryStatus');
const answerBox = document.getElementById('answerBox');
const answerText = document.getElementById('answerText');
const toolTag = document.getElementById('toolTag');

fileInput.addEventListener('change', () => {
  fileLabelText.textContent = fileInput.files[0]
    ? fileInput.files[0].name
    : 'Choose a CSV file';
});

uploadBtn.addEventListener('click', async () => {
  const file = fileInput.files[0];
  if (!file) {
    uploadStatus.textContent = 'Please choose a CSV file first.';
    uploadStatus.className = 'status error';
    return;
  }

  uploadBtn.disabled = true;
  uploadStatus.textContent = 'Uploading and building index... (first request may take 15-30s)';
  uploadStatus.className = 'status';

  try {
    const formData = new FormData();
    formData.append('file', file);
    const res = await fetch('/upload', { method: 'POST', body: formData });
    const data = await res.json();

    if (!res.ok) {
      uploadStatus.textContent = data.error || 'Upload failed.';
      uploadStatus.className = 'status error';
    } else {
      uploadStatus.textContent = `Loaded ${data.transactions_loaded} transactions. Ready to answer questions.`;
      uploadStatus.className = 'status success';
      questionInput.disabled = false;
      askBtn.disabled = false;
    }
  } catch (err) {
    uploadStatus.textContent = 'Network error -- please try again.';
    uploadStatus.className = 'status error';
  } finally {
    uploadBtn.disabled = false;
  }
});

async function askQuestion() {
  const question = questionInput.value.trim();
  if (!question) return;

  askBtn.disabled = true;
  queryStatus.textContent = 'Thinking...';
  queryStatus.className = 'status';
  answerBox.style.display = 'none';

  try {
    const res = await fetch('/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question })
    });
    const data = await res.json();

    if (!res.ok) {
      queryStatus.textContent = data.error || 'Something went wrong.';
      queryStatus.className = 'status error';
    } else {
      queryStatus.textContent = '';
      // The LLM sometimes wraps numbers in markdown bold (**text**) --
      // render that as real bold instead of showing literal asterisks.
      // Escape HTML first so nothing in the answer can inject markup,
      // then convert only the safe **bold** pattern.
      const escaped = data.answer
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
      answerText.innerHTML = escaped.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
      toolTag.textContent = data.tool_used ? `via ${data.tool_used}` : '';
      answerBox.style.display = 'block';
    }
  } catch (err) {
    queryStatus.textContent = 'Network error -- please try again.';
    queryStatus.className = 'status error';
  } finally {
    askBtn.disabled = false;
  }
}

askBtn.addEventListener('click', askQuestion);
questionInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') askQuestion();
});