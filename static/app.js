// DermaVision AI — Frontend Application Logic

// Tab Switching
function switchTab(tabId) {
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
    
    event.currentTarget.classList.add('active');
    document.getElementById(tabId).classList.add('active');
}

// Preset Scenario Loader
async function loadPreset(presetKey) {
    try {
        const response = await fetch('/api/presets');
        const presets = await response.json();
        
        const preset = presets[presetKey];
        if (!preset) return;
        
        // Fill form fields
        for (const [key, value] of Object.entries(preset)) {
            const field = document.getElementById(key);
            if (!field) continue;
            
            if (field.type === 'checkbox') {
                field.checked = Boolean(value);
            } else {
                field.value = value;
            }
        }
        
        // Trigger prediction automatically on preset click
        document.getElementById('dermavision-form').dispatchEvent(new Event('submit'));
        
    } catch (err) {
        console.error('Error loading preset:', err);
    }
}

// Form Submission & API Call
async function handlePrediction(event) {
    event.preventDefault();
    
    const btn = document.getElementById('btn-predict');
    const originalBtnText = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Processing PyTorch Tensor & Score Engine...';
    
    const form = document.getElementById('dermavision-form');
    const formData = new FormData(form);
    
    // Construct raw input object
    const rawInput = {
        age: parseFloat(formData.get('age') || 50),
        fitspatrick: parseInt(formData.get('fitspatrick') || 2),
        diameter_1: parseFloat(formData.get('diameter_1') || 10.0),
        diameter_2: parseFloat(formData.get('diameter_2') || 8.0),
        gender: formData.get('gender') || 'MALE',
        background_father: formData.get('background_father') || 'UNK',
        background_mother: formData.get('background_mother') || 'UNK',
        region: formData.get('region') || 'FACE',
        
        // Checkboxes
        itch: document.getElementById('itch').checked,
        grew: document.getElementById('grew').checked,
        hurt: document.getElementById('hurt').checked,
        changed: document.getElementById('changed').checked,
        bleed: document.getElementById('bleed').checked,
        elevation: document.getElementById('elevation').checked,
        biopsed: document.getElementById('biopsed').checked,
        skin_cancer_history: document.getElementById('skin_cancer_history').checked,
        cancer_history: document.getElementById('cancer_history').checked,
        smoke: document.getElementById('smoke').checked,
        drink: false,
        pesticide: false,
        has_piped_water: true,
        has_sewage_system: true
    };
    
    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(rawInput)
        });
        
        const result = await response.json();
        
        if (result.success) {
            renderDashboard(result.data);
        } else {
            alert('Prediction error: ' + (result.error || 'Unknown error'));
        }
    } catch (err) {
        console.error('API Call Failed:', err);
        alert('Network error connecting to DermaVision PyTorch API server.');
    } finally {
        btn.disabled = false;
        btn.innerHTML = originalBtnText;
    }
}

// Render Results Dashboard
function renderDashboard(data) {
    // Hide empty state & show dashboard
    document.getElementById('empty-state').classList.add('hidden');
    const dashboard = document.getElementById('results-dashboard');
    dashboard.classList.remove('hidden');
    
    // Update live indicator badge
    const liveIndicator = document.getElementById('live-indicator');
    liveIndicator.textContent = 'Analysis Complete';
    liveIndicator.style.background = 'rgba(16, 185, 129, 0.2)';
    liveIndicator.style.color = '#10b981';
    
    // Primary Diagnostic Card
    document.getElementById('diag-name').textContent = data.diagnosis_name;
    document.getElementById('diag-code').textContent = data.diagnosis_code;
    document.getElementById('diag-confidence').textContent = `${data.prediction_confidence}%`;
    document.getElementById('diag-description').textContent = data.diagnosis_description;
    
    // Diagnostic Card styling accent based on condition severity
    const diagCard = document.getElementById('diagnostic-card');
    if (['MEL', 'SCC', 'BCC'].includes(data.diagnosis_code)) {
        diagCard.style.borderColor = 'rgba(239, 68, 68, 0.6)';
        diagCard.style.background = 'linear-gradient(135deg, rgba(239, 68, 68, 0.15), rgba(245, 158, 11, 0.15))';
    } else {
        diagCard.style.borderColor = 'rgba(6, 182, 212, 0.4)';
        diagCard.style.background = 'linear-gradient(135deg, rgba(6, 182, 212, 0.15), rgba(59, 130, 246, 0.15))';
    }
    
    // Skin Health Score
    document.getElementById('health-score-value').textContent = data.skin_health_score;
    document.getElementById('health-bar').style.width = `${data.skin_health_score}%`;
    const healthBadge = document.getElementById('health-category-badge');
    healthBadge.textContent = data.skin_health_category;
    healthBadge.className = `metric-category-badge cat-${data.skin_health_category.toLowerCase()}`;
    
    // Pigmentation Risk Score
    document.getElementById('pigmentation-score-value').textContent = data.pigmentation_risk_score;
    document.getElementById('pigmentation-bar').style.width = `${data.pigmentation_risk_score}%`;
    const pigBadge = document.getElementById('pigmentation-category-badge');
    pigBadge.textContent = data.pigmentation_risk_category;
    pigBadge.className = `metric-category-badge cat-${data.pigmentation_risk_category.toLowerCase()}`;
    
    // Clinical Severity Score
    document.getElementById('severity-score-value').textContent = data.skin_severity_score;
    document.getElementById('severity-bar').style.width = `${data.skin_severity_score}%`;
    const sevBadge = document.getElementById('severity-category-badge');
    sevBadge.textContent = data.severity_category;
    sevBadge.className = `metric-category-badge cat-${data.severity_category.toLowerCase()}`;
    
    // Class Probability Breakdown Bars
    const probContainer = document.getElementById('prob-bars');
    probContainer.innerHTML = '';
    
    const probs = data.class_probabilities || {};
    const sortedClasses = Object.keys(probs).sort((a, b) => probs[b] - probs[a]);
    
    sortedClasses.forEach(cls => {
        const pct = probs[cls];
        const isTop = (cls === data.diagnosis_code);
        
        const itemHtml = `
            <div class="prob-item">
                <div class="prob-info">
                    <span class="prob-name">${cls} (${getDiagnosisShortName(cls)})</span>
                    <span class="prob-pct">${pct}%</span>
                </div>
                <div class="prob-track">
                    <div class="prob-fill ${isTop ? 'top-class' : ''}" style="width: ${pct}%"></div>
                </div>
            </div>
        `;
        probContainer.innerHTML += itemHtml;
    });
    
    // Recommendation
    document.getElementById('recommendation-text').textContent = data.recommendation;
    
    // Smooth scroll down to dashboard on mobile
    if (window.innerWidth <= 1100) {
        dashboard.scrollIntoView({ behavior: 'smooth' });
    }
}

function getDiagnosisShortName(code) {
    const names = {
        'ACK': 'Actinic Keratosis',
        'BCC': 'Basal Cell Carcinoma',
        'MEL': 'Melanoma',
        'NEV': 'Nevus',
        'SCC': 'Squamous Cell Carcinoma',
        'SEK': 'Seborrheic Keratosis'
    };
    return names[code] || code;
}
