// DermaVision AI — Frontend Application Logic (Unified Modules 1–5)

let currentPatientContext = null;

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

// Form Submission & Multimodal GenAI API Call
async function handlePrediction(event) {
    event.preventDefault();
    
    const btn = document.getElementById('btn-predict');
    const originalBtnText = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Synthesizing Multimodal Neural Intelligence (M1–M5)...';
    
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
        const response = await fetch('/api/module5/generate-report', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(rawInput)
        });
        
        const result = await response.json();
        
        if (result.success) {
            currentPatientContext = result.data.context;
            renderDashboard(result.data.report, result.data.context);
        } else {
            alert('Analysis error: ' + (result.error || 'Unknown error'));
        }
    } catch (err) {
        console.error('API Call Failed:', err);
        alert('Network error connecting to DermaVision AI server.');
    } finally {
        btn.disabled = false;
        btn.innerHTML = originalBtnText;
    }
}

// Render Results Dashboard across Modules 1 to 5
function renderDashboard(report, context) {
    // Hide empty state & show dashboard
    document.getElementById('empty-state').classList.add('hidden');
    const dashboard = document.getElementById('results-dashboard');
    dashboard.classList.remove('hidden');
    
    // Update live indicator badge
    const liveIndicator = document.getElementById('live-indicator');
    liveIndicator.textContent = 'Multimodal Synthesis Complete';
    liveIndicator.style.background = 'rgba(16, 185, 129, 0.2)';
    liveIndicator.style.color = '#10b981';
    
    // 1. Primary Diagnostic Card
    document.getElementById('diag-name').textContent = report.diagnosis_name;
    document.getElementById('diag-code').textContent = report.diagnosis_code;
    document.getElementById('diag-confidence').textContent = `${context.confidence.toFixed(1)}%`;
    document.getElementById('diag-description').textContent = report.plain_language_explanation;
    
    // Diagnostic Card styling accent based on condition severity
    const diagCard = document.getElementById('diagnostic-card');
    if (['MEL', 'SCC', 'BCC'].includes(report.diagnosis_code)) {
        diagCard.style.borderColor = 'rgba(239, 68, 68, 0.6)';
        diagCard.style.background = 'linear-gradient(135deg, rgba(239, 68, 68, 0.15), rgba(245, 158, 11, 0.15))';
    } else {
        diagCard.style.borderColor = 'rgba(6, 182, 212, 0.4)';
        diagCard.style.background = 'linear-gradient(135deg, rgba(6, 182, 212, 0.15), rgba(59, 130, 246, 0.15))';
    }
    
    // 2. Clinical Metric Score Cards
    document.getElementById('health-score-value').textContent = context.skin_health_score.toFixed(1);
    document.getElementById('health-bar').style.width = `${context.skin_health_score}%`;
    const healthBadge = document.getElementById('health-category-badge');
    healthBadge.textContent = context.skin_health_category;
    healthBadge.className = `metric-category-badge cat-${context.skin_health_category.toLowerCase()}`;
    
    document.getElementById('pigmentation-score-value').textContent = context.pigmentation_risk_score.toFixed(1);
    document.getElementById('pigmentation-bar').style.width = `${context.pigmentation_risk_score}%`;
    const pigBadge = document.getElementById('pigmentation-category-badge');
    pigBadge.textContent = context.pigmentation_risk_category;
    pigBadge.className = `metric-category-badge cat-${context.pigmentation_risk_category.toLowerCase()}`;
    
    document.getElementById('severity-score-value').textContent = context.skin_severity_score.toFixed(1);
    document.getElementById('severity-bar').style.width = `${context.skin_severity_score}%`;
    const sevBadge = document.getElementById('severity-category-badge');
    sevBadge.textContent = context.severity_category;
    sevBadge.className = `metric-category-badge cat-${context.severity_category.toLowerCase()}`;
    
    // 3. Class Probability Breakdown Bars
    const probContainer = document.getElementById('prob-bars');
    probContainer.innerHTML = '';
    
    const probs = context.class_probabilities || {
        [report.diagnosis_code]: context.confidence,
        'NEV': report.diagnosis_code === 'NEV' ? context.confidence : (100 - context.confidence) * 0.4,
        'BCC': report.diagnosis_code === 'BCC' ? context.confidence : (100 - context.confidence) * 0.3,
        'ACK': report.diagnosis_code === 'ACK' ? context.confidence : (100 - context.confidence) * 0.15,
        'SEK': report.diagnosis_code === 'SEK' ? context.confidence : (100 - context.confidence) * 0.1,
        'MEL': report.diagnosis_code === 'MEL' ? context.confidence : (100 - context.confidence) * 0.05
    };
    const sortedClasses = Object.keys(probs).sort((a, b) => probs[b] - probs[a]);
    
    sortedClasses.forEach(cls => {
        const pct = typeof probs[cls] === 'number' ? probs[cls].toFixed(1) : probs[cls];
        const isTop = (cls === report.diagnosis_code);
        
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
    
    // 4. Recommendation & Plain Explanation
    document.getElementById('recommendation-text').textContent = report.executive_summary;
    
    // 5. Module 4: Trajectory & Forecast
    const traj = context.temporal_forecast_module4;
    const trajBadge = document.getElementById('trajectory-badge');
    trajBadge.textContent = traj.trajectory_category;
    trajBadge.className = 'trajectory-badge ' + (
        traj.trajectory_category.includes('Improving') ? 'badge-improving' :
        traj.trajectory_category.includes('Worsening') ? 'badge-worsening' : 'badge-stable'
    );
    
    document.getElementById('h1-health').textContent = `${traj.forecast_1_month.skin_health_score.toFixed(1)}/100`;
    document.getElementById('h1-pig').textContent = `${traj.forecast_1_month.pigmentation_percentage.toFixed(1)}%`;
    
    document.getElementById('h3-health').textContent = `${traj.forecast_3_months.skin_health_score.toFixed(1)}/100`;
    document.getElementById('h3-pig').textContent = `${traj.forecast_3_months.pigmentation_percentage.toFixed(1)}%`;
    
    document.getElementById('h6-health').textContent = `${traj.forecast_6_months.skin_health_score.toFixed(1)}/100`;
    document.getElementById('h6-pig').textContent = `${traj.forecast_6_months.pigmentation_percentage.toFixed(1)}%`;
    
    // 6. Module 5B: VAE Similar Cases
    const vae = context.case_similarity_module5;
    document.getElementById('anomaly-badge').textContent = `Anomaly Score: ${vae.query_anomaly_score} (${vae.anomaly_interpretation})`;
    
    const casesContainer = document.getElementById('similar-cases-container');
    casesContainer.innerHTML = '';
    
    (vae.top_similar_cases || []).forEach(sc => {
        const cCard = `
            <div class="similar-case-card">
                <div class="sc-header">
                    <span>${sc.case_id}</span>
                    <span class="sc-sim">${sc.similarity_score.toFixed(1)}% Match</span>
                </div>
                <div class="sc-diag">${sc.diagnosis_name || sc.diagnosis_code} (${sc.diagnosis_code})</div>
                <div class="sc-outcome">${sc.clinical_outcome || 'Favorable treatment response.'}</div>
            </div>
        `;
        casesContainer.innerHTML += cCard;
    });
    
    // 7. Module 5C: Skincare Regimen & Active Ingredients
    document.getElementById('urgency-pill').textContent = report.urgency_level;
    
    // Ingredient Chips
    const targetChips = document.getElementById('target-actives-chips');
    targetChips.innerHTML = (report.target_actives || []).map(a => `<span class="chip chip-green">${a}</span>`).join('');
    
    const contraChips = document.getElementById('contraindicated-chips');
    contraChips.innerHTML = (report.contraindicated || []).map(a => `<span class="chip chip-red">${a}</span>`).join('');
    
    // AM Steps
    const amContainer = document.getElementById('am-steps-container');
    amContainer.innerHTML = (report.am_regimen || []).map(step => `
        <div class="routine-step-item">
            <div class="step-num-prod">Step ${step.step}: ${step.product} <span class="step-active">(${step.active})</span></div>
            <div class="step-inst">${step.instruction}</div>
        </div>
    `).join('');
    
    // PM Steps
    const pmContainer = document.getElementById('pm-steps-container');
    pmContainer.innerHTML = (report.pm_regimen || []).map(step => `
        <div class="routine-step-item">
            <div class="step-num-prod">Step ${step.step}: ${step.product} <span class="step-active">(${step.active})</span></div>
            <div class="step-inst">${step.instruction}</div>
        </div>
    `).join('');
    
    // Reset Chatbot with tailored greeting
    const chatWindow = document.getElementById('chat-window');
    chatWindow.innerHTML = `
        <div class="chat-msg bot-msg">
            <div class="avatar"><i class="fa-solid fa-microscope"></i></div>
            <div class="msg-text">
                Assessment complete for <strong>${report.diagnosis_name}</strong>. I can answer questions about your recommended actives (<strong>${report.target_actives[0]}</strong>), SPF application, or your projected 3-month forecast. What would you like to know?
            </div>
        </div>
    `;
    
    // Smooth scroll down to dashboard on mobile
    if (window.innerWidth <= 1100) {
        dashboard.scrollIntoView({ behavior: 'smooth' });
    }
}

// Chatbot Interaction Logic
async function handleChatSubmit(event) {
    event.preventDefault();
    const input = document.getElementById('chat-input');
    const msg = input.value.trim();
    if (!msg) return;
    
    appendChatMessage('user', msg);
    input.value = '';
    
    const sendBtn = document.getElementById('btn-send-chat');
    sendBtn.disabled = true;
    
    try {
        const response = await fetch('/api/module5/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: msg,
                patient_context: currentPatientContext
            })
        });
        
        const data = await response.json();
        if (data.success && data.data) {
            appendChatMessage('bot', data.data.bot_response);
        } else {
            appendChatMessage('bot', "I couldn't process that question. Please try asking about sunscreen, ingredients, or your 3-month trajectory.");
        }
    } catch (err) {
        appendChatMessage('bot', "Connection error communicating with the consultation assistant.");
    } finally {
        sendBtn.disabled = false;
    }
}

function askQuickPrompt(promptText) {
    document.getElementById('chat-input').value = promptText;
    document.getElementById('chat-form').dispatchEvent(new Event('submit'));
}

function appendChatMessage(role, text) {
    const chatWindow = document.getElementById('chat-window');
    const msgDiv = document.createElement('div');
    msgDiv.className = `chat-msg ${role}-msg`;
    
    const icon = role === 'bot' ? '<i class="fa-solid fa-microscope"></i>' : '<i class="fa-solid fa-user"></i>';
    // Format bold markdown
    const formattedText = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    
    msgDiv.innerHTML = `
        <div class="avatar">${icon}</div>
        <div class="msg-text">${formattedText}</div>
    `;
    chatWindow.appendChild(msgDiv);
    chatWindow.scrollTop = chatWindow.scrollHeight;
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
