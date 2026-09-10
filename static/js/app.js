/**
 * CarPrice AI — Frontend Application
 *
 * SPA navigation, form handling, API communication,
 * prediction history, analytics, and theme management.
 */

// ── State ────────────────────────────────────────────────────
const state = {
    currentPage: 'predict',
    modelInfo: null,
    historyPage: 1,
    historyData: null,
    theme: localStorage.getItem('theme') || 'light',
};

// ── Init ─────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    initTheme();
    initNavigation();
    initForm();
    loadModelInfo();
    loadAnalytics();
    loadHistory();
});

// ── Theme ────────────────────────────────────────────────────
function initTheme() {
    document.documentElement.setAttribute('data-theme', state.theme);
    updateThemeIcon();

    document.getElementById('themeToggle').addEventListener('click', () => {
        state.theme = state.theme === 'light' ? 'dark' : 'light';
        document.documentElement.setAttribute('data-theme', state.theme);
        localStorage.setItem('theme', state.theme);
        updateThemeIcon();
    });
}

function updateThemeIcon() {
    const btn = document.getElementById('themeToggle');
    btn.textContent = state.theme === 'light' ? '🌙' : '☀️';
}

// ── Navigation ───────────────────────────────────────────────
function initNavigation() {
    document.querySelectorAll('[data-page]').forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            navigateTo(link.dataset.page);
            // Close mobile menu
            document.getElementById('navLinks').classList.remove('open');
        });
    });

    document.getElementById('mobileMenuBtn')?.addEventListener('click', () => {
        document.getElementById('navLinks').classList.toggle('open');
    });
}

function navigateTo(page) {
    state.currentPage = page;

    // Update nav active state
    document.querySelectorAll('[data-page]').forEach(l => l.classList.remove('active'));
    document.querySelectorAll(`[data-page="${page}"]`).forEach(l => l.classList.add('active'));

    // Show/hide sections
    document.querySelectorAll('.page-section').forEach(s => s.classList.remove('active'));
    document.getElementById(`page-${page}`)?.classList.add('active');

    // Lazy-load data for specific pages
    if (page === 'history') loadHistory();
    if (page === 'analytics') loadAnalytics();
    if (page === 'model') loadModelPage();
}

// ── Form ─────────────────────────────────────────────────────
function initForm() {
    const form = document.getElementById('predictionForm');
    form.addEventListener('submit', handlePrediction);

    // Dynamic model dropdown based on brand selection
    document.getElementById('brand').addEventListener('change', (e) => {
        updateModelDropdown(e.target.value);
        // Clear specs when brand changes
        autoFillSpecs(null, null);
    });

    document.getElementById('model').addEventListener('change', (e) => {
        const brand = document.getElementById('brand').value;
        autoFillSpecs(brand, e.target.value);
    });
}

function autoFillSpecs(brand, model) {
    if (!state.modelInfo?.car_specs || !brand || !model) return;
    
    const specs = state.modelInfo.car_specs[brand]?.[model];
    if (!specs) return;

    // Helper to set and highlight
    const setVal = (id, val) => {
        const el = document.getElementById(id);
        if (el && val !== undefined) {
            el.value = val;
            el.classList.add('highlight-autofill');
            setTimeout(() => el.classList.remove('highlight-autofill'), 1500);
        }
    };

    setVal('engine', specs.engine);
    setVal('mileage', specs.mileage);
    setVal('power', specs.power);
    setVal('seats', specs.seats);
    setVal('fuel_type', specs.fuel_type);
    setVal('transmission', specs.transmission);
    
    showToast(`Loaded default specs for ${brand} ${model}`, 'success');
}

function updateModelDropdown(brand) {
    const modelSelect = document.getElementById('model');
    modelSelect.innerHTML = '<option value="">Select Model</option>';

    if (!state.modelInfo?.brand_model_map || !brand) return;

    const models = state.modelInfo.brand_model_map[brand] || [];
    models.forEach(m => {
        const opt = document.createElement('option');
        opt.value = m;
        opt.textContent = m;
        modelSelect.appendChild(opt);
    });
}

async function handlePrediction(e) {
    e.preventDefault();

    const form = e.target;
    const btn = document.getElementById('predictBtn');
    const resultContainer = document.getElementById('predictionResult');

    // Collect form data
    const data = {
        brand: form.brand.value,
        model: form.model.value,
        year: parseInt(form.year.value),
        kilometers_driven: parseInt(form.kilometers_driven.value),
        fuel_type: form.fuel_type.value,
        transmission: form.transmission.value,
        seller_type: form.seller_type?.value || 'Individual',
        engine: parseInt(form.engine.value),
        mileage: parseFloat(form.mileage.value),
        power: parseFloat(form.power.value),
        seats: parseInt(form.seats.value),
    };

    // Client-side validation
    const errors = validateInput(data);
    if (errors.length > 0) {
        showToast(errors[0], 'error');
        return;
    }

    // Loading state
    btn.disabled = true;
    btn.innerHTML = '<span class="loading-dot"></span><span class="loading-dot"></span><span class="loading-dot"></span> Analyzing...';
    resultContainer.classList.remove('visible');

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data),
        });

        const result = await response.json();

        if (!response.ok) {
            const errorMsg = result.errors ? result.errors.join(', ') : result.error || 'Prediction failed';
            throw new Error(errorMsg);
        }

        // Display result
        displayPredictionResult(result, data);
        showToast('Prediction completed successfully!', 'success');

    } catch (err) {
        showToast(err.message, 'error');
    } finally {
        btn.disabled = false;
        btn.innerHTML = '⚡ Predict Car Price';
    }
}

function displayPredictionResult(result, inputData) {
    const container = document.getElementById('predictionResult');

    document.getElementById('resultPrice').textContent = formatCurrency(result.predicted_price);
    document.getElementById('resultModel').textContent = `${inputData.brand} ${inputData.model}`;
    document.getElementById('resultVersion').textContent = `v${result.model_version}`;
    document.getElementById('resultTimestamp').textContent = formatDate(result.timestamp);

    // Detail items
    document.getElementById('detailYear').textContent = inputData.year;
    document.getElementById('detailFuel').textContent = inputData.fuel_type;
    document.getElementById('detailTransmission').textContent = inputData.transmission;
    document.getElementById('detailKm').textContent = formatNumber(inputData.kilometers_driven) + ' km';
    document.getElementById('detailEngine').textContent = inputData.engine + ' cc';
    document.getElementById('detailPower').textContent = inputData.power + ' bhp';

    container.classList.add('visible');
    container.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// ── Validation ───────────────────────────────────────────────
function validateInput(data) {
    const errors = [];
    if (!data.brand) errors.push('Please select a brand');
    if (!data.model) errors.push('Please select a model');
    if (!data.year || data.year < 1990 || data.year > 2026) errors.push('Year must be between 1990 and 2026');
    if (!data.kilometers_driven || data.kilometers_driven < 0) errors.push('Kilometers must be a positive number');
    if (!data.fuel_type) errors.push('Please select fuel type');
    if (!data.transmission) errors.push('Please select transmission');
    if (!data.engine || data.engine <= 0) errors.push('Engine CC must be positive');
    if (!data.mileage && data.mileage !== 0) errors.push('Mileage is required');
    if (!data.power || data.power <= 0) errors.push('Power must be positive');
    if (!data.seats || data.seats < 2 || data.seats > 10) errors.push('Seats must be between 2 and 10');
    return errors;
}

// ── Model Info ───────────────────────────────────────────────
async function loadModelInfo() {
    try {
        const res = await fetch('/api/model-info');
        if (!res.ok) return;
        state.modelInfo = await res.json();
        populateBrandDropdown();
    } catch (err) {
        console.error('Failed to load model info:', err);
    }
}

function populateBrandDropdown() {
    const brandSelect = document.getElementById('brand');
    if (!state.modelInfo?.valid_brands) return;

    brandSelect.innerHTML = '<option value="">Select Brand</option>';
    state.modelInfo.valid_brands.forEach(b => {
        const opt = document.createElement('option');
        opt.value = b;
        opt.textContent = b;
        brandSelect.appendChild(opt);
    });
}

// ── History ──────────────────────────────────────────────────
async function loadHistory(page = 1) {
    state.historyPage = page;
    const container = document.getElementById('historyTableBody');
    const pagination = document.getElementById('historyPagination');

    try {
        const res = await fetch(`/api/predictions?page=${page}&per_page=15`);
        if (!res.ok) throw new Error('Failed to load history');
        const data = await res.json();
        state.historyData = data;

        if (data.predictions.length === 0) {
            container.innerHTML = `
                <tr>
                    <td colspan="7">
                        <div class="empty-state">
                            <div class="empty-state-icon">📊</div>
                            <div class="empty-state-title">No predictions yet</div>
                            <div class="empty-state-text">Make your first prediction to see it here</div>
                        </div>
                    </td>
                </tr>`;
            pagination.innerHTML = '';
            return;
        }

        container.innerHTML = data.predictions.map(p => `
            <tr>
                <td><strong>${p.brand}</strong> ${p.model}</td>
                <td>${p.year}</td>
                <td>${p.fuel_type}</td>
                <td>${p.transmission}</td>
                <td>${formatNumber(p.kilometers_driven)} km</td>
                <td class="price-cell">${formatCurrency(p.predicted_price)}</td>
                <td>${formatDate(p.timestamp)}</td>
            </tr>
        `).join('');

        // Pagination
        const prevDisabled = page <= 1 ? 'disabled' : '';
        const nextDisabled = page >= data.pages ? 'disabled' : '';
        pagination.innerHTML = `
            <button class="pagination-btn" ${prevDisabled} onclick="loadHistory(${page - 1})">← Previous</button>
            <span class="pagination-info">Page ${data.page} of ${data.pages}</span>
            <button class="pagination-btn" ${nextDisabled} onclick="loadHistory(${page + 1})">Next →</button>
        `;

    } catch (err) {
        console.error('Failed to load history:', err);
    }
}

// History search
function filterHistory() {
    const query = document.getElementById('historySearch').value.toLowerCase();
    const rows = document.querySelectorAll('#historyTableBody tr');
    rows.forEach(row => {
        const text = row.textContent.toLowerCase();
        row.style.display = text.includes(query) ? '' : 'none';
    });
}

// ── Analytics ────────────────────────────────────────────────
async function loadAnalytics() {
    try {
        const res = await fetch('/api/analytics');
        if (!res.ok) return;
        const data = await res.json();

        document.getElementById('statTotal').textContent = formatNumber(data.total_predictions);
        document.getElementById('statAvgPrice').textContent = data.average_price > 0
            ? formatCurrency(data.average_price) : '—';
        document.getElementById('statTopBrand').textContent = data.most_predicted_brand || '—';
        document.getElementById('statPriceRange').textContent = data.price_range.max > 0
            ? `${formatCurrencyShort(data.price_range.min)} – ${formatCurrencyShort(data.price_range.max)}` : '—';

    } catch (err) {
        console.error('Failed to load analytics:', err);
    }
}

// ── Model Page ───────────────────────────────────────────────
async function loadModelPage() {
    if (!state.modelInfo) {
        try {
            const res = await fetch('/api/model-info');
            if (!res.ok) return;
            state.modelInfo = await res.json();
        } catch { return; }
    }

    const info = state.modelInfo;
    const metrics = info.test_metrics || {};

    document.getElementById('modelName').textContent = info.model_name || 'N/A';
    document.getElementById('modelVersion').textContent = `v${info.model_version || '?'}`;
    document.getElementById('modelTrainDate').textContent = info.training_date
        ? formatDate(info.training_date) : 'N/A';
    document.getElementById('modelDatasetRows').textContent = formatNumber(info.dataset_rows || 0);

    document.getElementById('metricR2').textContent = (metrics.R2 !== undefined)
        ? (metrics.R2 * 100).toFixed(1) + '%' : 'N/A';
    document.getElementById('metricMAE').textContent = metrics.MAE
        ? formatCurrencyShort(metrics.MAE) : 'N/A';
    document.getElementById('metricRMSE').textContent = metrics.RMSE
        ? formatCurrencyShort(metrics.RMSE) : 'N/A';
    document.getElementById('metricMAPE').textContent = (metrics.MAPE !== undefined)
        ? metrics.MAPE.toFixed(1) + '%' : 'N/A';
    document.getElementById('metricLatency').textContent = info.inference_latency_ms
        ? info.inference_latency_ms.toFixed(1) + 'ms' : 'N/A';
    document.getElementById('metricCVR2').textContent = info.cv_r2_mean
        ? (info.cv_r2_mean * 100).toFixed(1) + '%' : 'N/A';

    // Model comparison table
    const compBody = document.getElementById('comparisonBody');
    if (info.all_model_results && compBody) {
        compBody.innerHTML = info.all_model_results.map(r => {
            const isBest = r.name === info.model_name;
            return `
                <tr class="${isBest ? 'best-model' : ''}">
                    <td>${r.name} ${isBest ? '<span class="badge badge-success">Best</span>' : ''}</td>
                    <td>${(r.metrics.R2 * 100).toFixed(1)}%</td>
                    <td>${formatCurrencyShort(r.metrics.MAE)}</td>
                    <td>${formatCurrencyShort(r.metrics.RMSE)}</td>
                    <td>${r.metrics.MAPE.toFixed(1)}%</td>
                    <td>${(r.cv_r2_mean * 100).toFixed(1)}%</td>
                </tr>`;
        }).join('');
    }
}

// ── Formatting Utilities ─────────────────────────────────────
function formatCurrency(amount) {
    if (amount == null) return '—';
    return '₹' + Math.round(amount).toLocaleString('en-IN');
}

function formatCurrencyShort(amount) {
    if (amount == null) return '—';
    if (amount >= 10000000) return '₹' + (amount / 10000000).toFixed(1) + ' Cr';
    if (amount >= 100000) return '₹' + (amount / 100000).toFixed(1) + ' L';
    if (amount >= 1000) return '₹' + (amount / 1000).toFixed(1) + 'K';
    return '₹' + Math.round(amount).toLocaleString('en-IN');
}

function formatNumber(num) {
    if (num == null) return '—';
    return num.toLocaleString('en-IN');
}

function formatDate(dateStr) {
    if (!dateStr) return '—';
    const d = new Date(dateStr);
    return d.toLocaleDateString('en-IN', {
        day: 'numeric',
        month: 'short',
        year: 'numeric',
    });
}

// ── Toast Notifications ──────────────────────────────────────
function showToast(message, type = 'success') {
    const container = document.getElementById('toastContainer');
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.innerHTML = `${type === 'success' ? '✓' : '⚠'} ${message}`;
    container.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(20px)';
        toast.style.transition = 'all 300ms ease-out';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}
