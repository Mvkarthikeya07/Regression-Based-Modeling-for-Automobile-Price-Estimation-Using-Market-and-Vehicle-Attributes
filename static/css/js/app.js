// static/js/app.js
document.addEventListener('DOMContentLoaded', function () {
  const clearBtn = document.getElementById('clearBtn');
  const estimateAjax = document.getElementById('estimateAjax');
  const priceDisplay = document.getElementById('priceDisplay');
  const form = document.getElementById('predictForm');

  clearBtn?.addEventListener('click', () => {
    form.reset();
    priceDisplay.textContent = '—';
  });

  estimateAjax?.addEventListener('click', async () => {
    // collect form data
    const formData = new FormData(form);
    const payload = {};
    for (const [k, v] of formData.entries()) payload[k] = v;

    priceDisplay.textContent = 'Predicting…';

    try {
      const resp = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      const json = await resp.json();
      if (resp.ok && json.predicted_price !== undefined) {
        priceDisplay.textContent = '₹ ' + Number(json.predicted_price).toLocaleString();
      } else {
        priceDisplay.textContent = 'Error';
        console.error('Prediction error', json);
      }
    } catch (err) {
      priceDisplay.textContent = 'Error';
      console.error(err);
    }
  });
});
