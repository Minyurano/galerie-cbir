// Galerie CBIR — comportements communs (sélection multiple, confirmations)
document.addEventListener('DOMContentLoaded', () => {
  // Confirmation avant les actions destructrices
  document.querySelectorAll('form[data-confirm]').forEach(f =>
    f.addEventListener('submit', e => { if (!confirm(f.dataset.confirm)) e.preventDefault(); }));
  document.querySelectorAll('button[data-confirm]').forEach(b =>
    b.addEventListener('click', e => { if (!confirm(b.dataset.confirm)) e.preventDefault(); }));

  // Sélection multiple dans la galerie
  const all = document.getElementById('selectAll');
  const checks = [...document.querySelectorAll('.thumb-check')];
  const count = document.getElementById('selCount');
  const needSel = [...document.querySelectorAll('[data-needs-selection]')];
  function update() {
    const n = checks.filter(c => c.checked).length;
    if (count) count.textContent = `${n} sélectionnée(s)`;
    needSel.forEach(b => b.disabled = n === 0);
    checks.forEach(c => c.closest('.thumb-card').classList.toggle('selected', c.checked));
    if (all) all.checked = n > 0 && n === checks.length;
  }
  checks.forEach(c => c.addEventListener('change', update));
  if (all) all.addEventListener('change', () => { checks.forEach(c => c.checked = all.checked); update(); });
  update();
});
