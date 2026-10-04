// Shared by every benchmark page: the run id travels in the URL, a submit posts the values to the local server, and the page
// shows an inline confirmation that is gone after a reload.
var RUN = new URLSearchParams(location.search).get('run') || '';
function post(task, values) {
  return fetch('/submit', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ run: RUN, task: task, values: values }) });
}
function showNotice(text) {
  var notice = document.getElementById('notice');
  notice.textContent = text;
  notice.style.display = 'block';
}
