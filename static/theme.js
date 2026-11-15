// Applied before first paint so the page never flashes the wrong theme.
(function () {
  try {
    var choice = localStorage.getItem('theme');
    if (choice === 'light' || choice === 'dark') {
      document.documentElement.setAttribute('data-theme', choice);
    }
  } catch (_) { /* storage blocked: follow the system setting */ }
})();
