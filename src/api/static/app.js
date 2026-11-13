const $ = id => document.getElementById(id);
let lastResponse = null;
let busy = false;
let indexReady = false;

function message(text, error = false) {
  $('message').hidden = !text;
  $('message').textContent = text;
  $('message').className = error ? 'error' : '';
}
async function api(path, options = {}) {
  const response = await fetch(path, options);
  const payload = await response.json();
  if (!response.ok) {
    const detail = typeof payload.detail === 'string' ? payload.detail : 'Câu hỏi hoặc tham số không hợp lệ.';
    throw new Error(detail);
  }
  return payload;
}
function selectTab(panel) {
  document.querySelectorAll('[role=tab]').forEach(tab => tab.setAttribute('aria-selected', String(tab.dataset.panel === panel)));
  document.querySelectorAll('[role=tabpanel]').forEach(tab => tab.hidden = tab.id !== panel);
}
function metric(label, value) {
  const el = document.createElement('div'); el.className = 'metric';
  const title = document.createElement('small'); title.textContent = label;
  const number = document.createElement('strong'); number.textContent = value;
  el.append(title, number); return el;
}
function render(data) {
  const result = data.result;
  $('metrics').replaceChildren(
    metric('Tổng thời gian', `${result.total_seconds.toFixed(2)} s`),
    metric('Truy xuất', `${result.retrieval_seconds.toFixed(2)} s`),
    metric('Sinh câu trả lời', result.generation_seconds == null ? '—' : `${result.generation_seconds.toFixed(2)} s`),
    metric('Đoạn tìm được', result.retrieved_chunks.length)
  );
  $('asked').textContent = result.input;
  $('trace').textContent = `${data.run_settings.pipeline}` + (result.search_query ? ` · Query: ${result.search_query} · Xử lý query: ${result.query_processing_seconds.toFixed(2)}s · Reranking: ${result.reranking_seconds.toFixed(2)}s` : '');
  let answer = result.answer || 'Đã hoàn tất truy xuất. Tắt chế độ “Chỉ truy xuất” để sinh câu trả lời.';
  result.retrieved_chunks.forEach((chunk, i) => { answer = answer.split(`[${chunk.chunk_id}]`).join(`[${i + 1}]`); });
  // Render model/doc text as text, never as HTML.
  $('answer').textContent = answer;
  $('token-info').textContent = result.prompt_eval_count == null ? '' : `Token đầu vào: ${result.prompt_eval_count} · Đầu ra: ${result.eval_count ?? '—'} · Trích dẫn [1], [2] tương ứng các nguồn truy xuất.`;
  $('sources').replaceChildren();
  const scoreType = result.retrieved_chunks[0]?.score_type;
  const scoreLabel = scoreType === 'rrf' ? 'điểm RRF' : scoreType === 'cross_encoder' ? 'điểm Cross-Encoder' : 'điểm cosine similarity';
  $('source-summary').textContent = `${result.retrieved_chunks.length} đoạn; ${scoreLabel}`;
  result.retrieved_chunks.forEach((chunk, i) => {
    const details = document.createElement('details'); details.open = i === 0;
    const summary = document.createElement('summary');
    summary.textContent = `[${i + 1}] Điểm ${chunk.score.toFixed(4)} · Ký tự ${chunk.start_char}–${chunk.end_char}` + (chunk.retrieval_rank ? ` · Hạng trước rerank: ${chunk.retrieval_rank}` : '');
    const text = document.createElement('p'); text.className = 'chunk-text'; text.textContent = chunk.text;
    const id = document.createElement('p'); const code = document.createElement('code'); code.textContent = chunk.chunk_id; id.append(code);
    details.append(summary, text);
    try {
      const url = new URL(chunk.source_url);
      if (['http:', 'https:'].includes(url.protocol)) {
        const link = document.createElement('a'); link.href = url.href; link.target = '_blank'; link.rel = 'noopener noreferrer';
        link.className = 'source-link'; link.textContent = 'Mở tài liệu nguồn ↗'; details.append(link);
      }
    } catch (_) { /* Invalid source URL is not made clickable. */ }
    details.append(id); $('sources').append(details);
  });
  $('json-output').textContent = JSON.stringify(data, null, 2);
  $('empty').hidden = true; $('results').hidden = false; selectTab('answer-panel');
  if (result.done_reason === 'length') message('Câu trả lời đã chạm giới hạn sinh và có thể chưa hoàn tất.');
}
$('top-k').addEventListener('input', () => $('top-k-value').textContent = $('top-k').value);
$('retrieval-only').addEventListener('change', () => {
  if (!busy) $('submit').textContent = $('retrieval-only').checked ? 'Truy xuất ↗' : 'Tìm và trả lời ↗';
});
document.querySelectorAll('[data-question]').forEach(button => button.addEventListener('click', () => {
  $('question').value = button.dataset.question; $('question').focus();
}));
document.querySelectorAll('[role=tab]').forEach(button => button.addEventListener('click', () => selectTab(button.dataset.panel)));
$('health').addEventListener('click', async () => {
  $('health').disabled = true; $('health-status').textContent = 'Đang kiểm tra…';
  try { await api(`/api/health?retrieval_only=${$('retrieval-only').checked}&pipeline=${$('pipeline').value}`); $('health-status').textContent = 'Đã kết nối Ollama. Model Ollama cần thiết đã có sẵn.'; }
  catch (error) { $('health-status').textContent = error.message; }
  finally { $('health').disabled = false; }
});
$('ask-form').addEventListener('submit', async event => {
  event.preventDefault(); if (busy) return;
  const question = $('question').value.trim();
  if (!question) { message('Bạn chưa nhập câu hỏi.', true); return; }
  busy = true; $('submit').disabled = true; $('submit').textContent = 'Đang xử lý…';
  $('pipeline').disabled = true;
  $('results').hidden = true; lastResponse = null;
  message($('retrieval-only').checked ? 'Đang tìm bằng chứng…' : 'Đang truy xuất và sinh câu trả lời…');
  try {
    const data = await api('/api/ask', {method:'POST', headers:{'Content-Type':'application/json'},
      body:JSON.stringify({question, top_k:Number($('top-k').value), retrieval_only:$('retrieval-only').checked, pipeline:$('pipeline').value})});
    lastResponse = data; message(''); render(data);
  } catch (error) { message(error.message, true); $('empty').hidden = false; }
  finally { busy = false; $('pipeline').disabled = false; $('submit').disabled = !indexReady; $('submit').textContent = $('retrieval-only').checked ? 'Truy xuất ↗' : 'Tìm và trả lời ↗'; }
});
$('download').addEventListener('click', () => {
  if (!lastResponse) return;
  const url = URL.createObjectURL(new Blob([JSON.stringify(lastResponse, null, 2)], {type:'application/json;charset=utf-8'}));
  const link = document.createElement('a'); link.href = url; link.download = 'rag_result.json'; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
async function loadPipeline() {
  if (busy) return;
  const pipeline = $('pipeline').value;
  indexReady = false; $('submit').disabled = true;
  try {
  const config = await api(`/api/config?pipeline=${pipeline}`);
  if ($('pipeline').value !== pipeline) return;
  $('embedding').textContent = config.embedding_model; $('generation').textContent = config.generation_model;
  $('top-k').value = config.top_k; $('top-k-value').textContent = config.top_k;
  indexReady = config.index_ready && config.reranker_ready; $('submit').disabled = !indexReady;
  message('');
  if (!config.index_ready) message(`Chưa có index. Chạy chunk rồi index với --config configs/${pipeline}.yaml.`, true);
  else if (!config.reranker_ready) message('Chưa có reranker. Chạy python experiments/prepare_reranker.py trước.', true);
  else if (config.query_processing) message('Cấu hình này gọi Qwen viết lại query trước khi truy xuất, kể cả chế độ chỉ truy xuất.');
  } catch (error) { message(`Không đọc được cấu hình: ${error.message}`, true); }
}
$('pipeline').addEventListener('change', loadPipeline);
loadPipeline();
