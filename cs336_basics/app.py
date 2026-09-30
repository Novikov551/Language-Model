import os
import sys
import subprocess
import threading
import time
import csv
import torch
from flask import Flask, render_template_string, request, jsonify

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from cs336_basics.Modules.BPETokenizer import BPETokenizer
from cs336_basics.Modules.Functions import generate
from cs336_basics.Modules.TransformerLM import TransformerLanguageModel

# ---------- Глобальные объекты для генерации ----------
tokenizer = None
generation_model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VOCAB_SIZE = 10000
D_MODEL = 512
CONTEXT_LENGTH = 256
NUM_HEADS = 16
NUM_LAYERS = 4
D_FF = 1344

def load_generation_model():
    global generation_model, tokenizer
    base_dir = current_dir
    vocab_path = os.path.join(base_dir, "Datasets", "TinyStories-Train.pickle")
    checkpoint_path = os.path.join(base_dir, "Checkpoints", "best_model.pt")
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Словарь не найден: {vocab_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Модель не найдена: {checkpoint_path}")
    special_tokens = ["<|endoftext|>"]
    tokenizer = BPETokenizer.from_files(vocab_path, vocab_path, special_tokens)
    model = TransformerLanguageModel(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        context_length=CONTEXT_LENGTH,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        num_layers=NUM_LAYERS,
        theta=10000.0,
        eps=1e-5,
        mask=True,
        device=device,
        dtype=torch.float32,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    generation_model = model
    print("Модель для генерации загружена")

# ---------- Управление обучением ----------
training_process = None
training_log_list = []
training_status = {"running": False, "pid": None, "start_time": None, "error": None}
training_lock = threading.Lock()

def append_log(text, flush=True):
    """Добавляет строку в лог и сразу выводит в консоль сервера."""
    with training_lock:
        training_log_list.append(str(text))
        if len(training_log_list) > 1000:
            training_log_list.pop(0)
    # Печатаем в консоль сервера (для отладки)
    print(f"[LOG] {text}")
    if flush:
        sys.stdout.flush()

def run_training_script(params):
    global training_process, training_status
    script_path = os.path.join(current_dir, "ModelTraining.py")
    if not os.path.exists(script_path):
        append_log(f"Ошибка: скрипт обучения не найден: {script_path}")
        with training_lock:
            training_status["running"] = False
            training_status["error"] = "Скрипт обучения не найден"
        return

    # Используем текущий интерпретатор Python (из виртуального окружения) с флагом -u
    python_exe = sys.executable
    cmd = [python_exe, "-u", script_path]

    # Формируем аргументы командной строки
    for key, value in params.items():
        if value is None or value == "":
            continue
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
        elif key == "betas":
            cmd.extend([f"--{key}", str(value[0]), str(value[1])])
        else:
            cmd.extend([f"--{key}", str(value)])

    append_log("Запуск: " + " ".join(cmd))
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"   # полное отключение буферизации

    try:
        # Запускаем процесс
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,   # объединяем stderr с stdout
            text=True,
            bufsize=1,                  # строчная буферизация
            env=env,
            cwd=current_dir            # запускаем из корня проекта
        )
        with training_lock:
            training_process = process
            training_status["running"] = True
            training_status["pid"] = process.pid
            training_status["start_time"] = time.time()
            training_status["error"] = None

        # Читаем вывод построчно (блокирующий цикл)
        for line in iter(process.stdout.readline, ''):
            if not line:
                break
            append_log(line.rstrip())

        process.wait()
        with training_lock:
            training_status["running"] = False
            training_status["pid"] = None
            if process.returncode != 0:
                training_status["error"] = f"Код возврата {process.returncode}"
                append_log(f"Ошибка: код {process.returncode}")
            else:
                append_log("Обучение успешно завершено")
    except Exception as e:
        append_log(f"Исключение: {str(e)}")
        with training_lock:
            training_status["running"] = False
            training_status["error"] = str(e)

def get_metrics_data():
    csv_path = os.path.join(current_dir, "Metrics", "metrics.csv")
    if not os.path.exists(csv_path):
        return {"steps": [], "train_loss": [], "val_loss": [], "train_ppl": [], "val_ppl": [], "lr": [], "time": []}
    steps = []
    train_loss = []
    val_loss = []
    train_ppl = []
    val_ppl = []
    lr = []
    time_sec = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                step = int(row['step'])
                steps.append(step)
                train_loss.append(float(row['train_loss']))
                val_loss.append(float(row['val_loss']))
                train_ppl.append(float(row.get('train_ppl', 0)))
                val_ppl.append(float(row.get('val_ppl', 0)))
                lr.append(float(row.get('lr', 0)))
                time_sec.append(float(row.get('time_seconds', 0)))
            except (KeyError, ValueError):
                continue
    return {
        "steps": steps,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_ppl": train_ppl,
        "val_ppl": val_ppl,
        "lr": lr,
        "time": time_sec
    }

# ---------- Flask приложение ----------
app = Flask(__name__)

MAIN_TEMPLATE = """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLM Studio</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { box-sizing: border-box; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; max-width: 1300px; margin: auto; padding: 20px; background: #f5f7fa; }
        h1 { color: #2c3e50; }
        .tabs { display: flex; gap: 10px; margin-bottom: 20px; border-bottom: 1px solid #ccc; }
        .tab-button { background: none; border: none; font-size: 18px; padding: 10px 20px; cursor: pointer; color: #7f8c8d; }
        .tab-button.active { color: #3498db; border-bottom: 2px solid #3498db; }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        .card { background: white; border-radius: 8px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        textarea, input, select { width: 100%; padding: 8px; margin: 5px 0 10px; border: 1px solid #ddd; border-radius: 4px; }
        button { background: #3498db; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; font-size: 14px; }
        button:hover { background: #2980b9; }
        .log-area { background: #1e1e1e; color: #d4d4d4; font-family: monospace; padding: 10px; height: 300px; overflow-y: auto; white-space: pre-wrap; }
        .param-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 15px; margin-bottom: 20px; }
        .param-item { display: flex; flex-direction: column; }
        .param-item label { font-weight: 600; margin-bottom: 5px; font-size: 14px; }
        .param-item input, .param-item select { width: 100%; }
        .checkbox-item { flex-direction: row; align-items: center; gap: 10px; }
        .checkbox-item label { margin-bottom: 0; cursor: pointer; }
        .double-input { display: flex; gap: 10px; }
        .double-input input { width: 50%; }
        .button-group { display: flex; gap: 15px; margin-top: 10px; }
        .smoothing-bar { display: flex; align-items: center; gap: 20px; flex-wrap: wrap; background: #f9f9f9; padding: 10px; border-radius: 6px; }
        .chart-row { display: flex; flex-wrap: wrap; gap: 20px; margin-bottom: 20px; }
        .chart-box { flex: 1; min-width: 300px; }
    </style>
</head>
<body>
    <h1>🤖 LLM Studio</h1>
    <div class="tabs">
        <button class="tab-button active" onclick="switchTab('generate')">✍️ Генерация текста</button>
        <button class="tab-button" onclick="switchTab('train')">⚙️ Обучение модели</button>
    </div>

    <!-- Генерация -->
    <div id="generate-tab" class="tab-content active">
        <div class="card">
            <h3>Введите начало текста</h3>
            <textarea id="prompt" rows="4" placeholder="Например: Once upon a time..."></textarea>
            <div style="display: flex; gap: 15px; flex-wrap: wrap; align-items: flex-end;">
                <div><label>Температура:</label> <input type="number" id="temp" value="0.8" step="0.05" min="0.1" max="1.5" style="width:90px;"></div>
                <div><label>Max токенов:</label> <input type="number" id="max_tokens" value="120" step="10" min="10" max="300" style="width:90px;"></div>
                <div><label>Top-p:</label> <input type="number" id="top_p" value="0.9" step="0.05" min="0" max="1" style="width:90px;"></div>
                <div><button onclick="generate()">🚀 Сгенерировать</button></div>
                <div><button onclick="reloadModel()">🔄 Перезагрузить модель</button></div>
            </div>
            <h3>Результат:</h3>
            <div id="generated" class="log-area" style="background:#f0f0f0; color:#333; height:200px;"></div>
            <div id="gen_status" style="margin-top:10px; color:gray;"></div>
        </div>
    </div>

    <!-- Обучение -->
    <div id="train-tab" class="tab-content">
        <div class="card">
            <h3>Гиперпараметры обучения</h3>
            <form id="train-form">
                <div class="param-grid">
                    <!-- Данные -->
                    <div class="param-item"><label>--train_data</label><input type="text" id="train_data" value="./Datasets/TinyStories-train.npy"></div>
                    <div class="param-item"><label>--val_data</label><input type="text" id="val_data" value="./Datasets/TinyStories-valid.npy"></div>
                    <div class="param-item"><label>--batch_size</label><input type="number" id="batch_size" value="20"></div>
                    <div class="param-item"><label>--context_length</label><input type="number" id="context_length" value="256"></div>
                    <!-- Модель -->
                    <div class="param-item"><label>--d_model</label><input type="number" id="d_model" value="512"></div>
                    <div class="param-item"><label>--num_heads</label><input type="number" id="num_heads" value="16"></div>
                    <div class="param-item"><label>--num_layers</label><input type="number" id="num_layers" value="4"></div>
                    <div class="param-item"><label>--d_ff</label><input type="number" id="d_ff" value="1344"></div>
                    <div class="param-item"><label>--vocab_size</label><input type="number" id="vocab_size" value="10000"></div>
                    <div class="param-item"><label>--rope_theta</label><input type="text" id="rope_theta" value="10000.0"></div>
                    <div class="param-item checkbox-item"><input type="checkbox" id="no_rope"> <label>Отключить RoPE</label></div>
                    <!-- Оптимизатор -->
                    <div class="param-item"><label>--lr</label><input type="text" id="lr" value="3e-4"></div>
                    <div class="param-item"><label>--lr_min</label><input type="text" id="lr_min" value="1e-5"></div>
                    <div class="param-item"><label>--warmup_steps</label><input type="number" id="warmup_steps" value="500"></div>
                    <div class="param-item"><label>--weight_decay</label><input type="text" id="weight_decay" value="0.01"></div>
                    <div class="param-item"><label>--betas (beta1 beta2)</label>
                        <div class="double-input">
                            <input type="text" id="beta1" value="0.9" placeholder="beta1">
                            <input type="text" id="beta2" value="0.95" placeholder="beta2">
                        </div>
                    </div>
                    <div class="param-item"><label>--eps</label><input type="text" id="eps" value="1e-8"></div>
                    <div class="param-item"><label>--grad_clip</label><input type="text" id="grad_clip" value="1.0"></div>
                    <!-- Обучение -->
                    <div class="param-item"><label>--max_iters</label><input type="number" id="max_iters" value="5000"></div>
                    <div class="param-item"><label>--eval_interval</label><input type="number" id="eval_interval" value="500"></div>
                    <div class="param-item"><label>--log_interval</label><input type="number" id="log_interval" value="100"></div>
                    <div class="param-item"><label>--save_interval</label><input type="number" id="save_interval" value="1000"></div>
                    <div class="param-item"><label>--resume</label><input type="text" id="resume" placeholder="path/to/checkpoint.pt"></div>
                    <div class="param-item"><label>--device</label>
                        <select id="device">
                            <option value="cuda">cuda</option>
                            <option value="cpu">cpu</option>
                            <option value="mps">mps</option>
                        </select>
                    </div>
                    <div class="param-item"><label>--seed</label><input type="number" id="seed" value="42"></div>
                    <div class="param-item checkbox-item"><input type="checkbox" id="use_wandb"> <label>Использовать wandb</label></div>
                    <div class="param-item"><label>--wandb_project</label><input type="text" id="wandb_project" value="transformer-lm"></div>
                    <div class="param-item"><label>--wandb_run_name</label><input type="text" id="wandb_run_name" placeholder="auto"></div>
                </div>
                <div class="button-group">
                    <button type="button" onclick="startTraining()">▶️ Запустить обучение</button>
                    <button type="button" onclick="stopTraining()">⏹️ Остановить</button>
                </div>
            </form>
        </div>

        <!-- Панель сглаживания -->
        <div class="card">
            <h3>📊 Настройка графиков</h3>
            <div class="smoothing-bar">
                <label><input type="checkbox" id="smoothingCheckbox" onchange="toggleSmoothing()"> Включить сглаживание (скользящее среднее)</label>
                <label>Окно сглаживания: 
                    <select id="smoothWindow" onchange="applySmoothingAndUpdate()">
                        <option value="3">3</option>
                        <option value="5" selected>5</option>
                        <option value="10">10</option>
                        <option value="20">20</option>
                        <option value="50">50</option>
                    </select>
                </label>
            </div>
        </div>

        <!-- Графики -->
        <div class="card">
            <h3>📉 Графики метрик</h3>
            <div class="chart-row">
                <div class="chart-box"><canvas id="lossChart"></canvas></div>
                <div class="chart-box"><canvas id="pplChart"></canvas></div>
            </div>
            <div class="chart-row">
                <div class="chart-box"><canvas id="lrChart"></canvas></div>
                <div class="chart-box"></div>
            </div>
        </div>

        <!-- Логи -->
        <div class="card">
            <h3>📄 Логи обучения</h3>
            <div id="train-logs" class="log-area">Ожидание запуска...</div>
            <div id="train-status" style="margin-top:10px;"></div>
        </div>
    </div>

    <script>
        let logInterval = null, chartInterval = null;
        let lossChart, pplChart, lrChart;
        let rawMetrics = { steps: [], train_loss: [], val_loss: [], train_ppl: [], val_ppl: [], lr: [] };
        
        function movingAverage(data, window) {
            if (!data.length || window <= 1) return data.slice();
            const result = [];
            for (let i = 0; i < data.length; i++) {
                let left = Math.max(0, i - Math.floor(window/2));
                let right = Math.min(data.length - 1, i + Math.floor(window/2));
                let sum = 0;
                for (let j = left; j <= right; j++) sum += data[j];
                result.push(sum / (right - left + 1));
            }
            return result;
        }
        
        function applySmoothingAndUpdate() {
            const enabled = document.getElementById('smoothingCheckbox').checked;
            const win = parseInt(document.getElementById('smoothWindow').value);
            if (!enabled || rawMetrics.steps.length === 0) {
                updateChartData(lossChart, rawMetrics.steps, rawMetrics.train_loss, rawMetrics.val_loss);
                updateChartData(pplChart, rawMetrics.steps, rawMetrics.train_ppl, rawMetrics.val_ppl);
                updateChartData(lrChart, rawMetrics.steps, rawMetrics.lr, null);
                return;
            }
            updateChartData(lossChart, rawMetrics.steps, movingAverage(rawMetrics.train_loss, win), movingAverage(rawMetrics.val_loss, win));
            updateChartData(pplChart, rawMetrics.steps, movingAverage(rawMetrics.train_ppl, win), movingAverage(rawMetrics.val_ppl, win));
            updateChartData(lrChart, rawMetrics.steps, movingAverage(rawMetrics.lr, win), null);
        }
        
        function updateChartData(chart, labels, data1, data2) {
            if (!chart) return;
            chart.data.labels = labels;
            chart.data.datasets[0].data = data1;
            if (data2 !== null && chart.data.datasets[1]) chart.data.datasets[1].data = data2;
            chart.update('none');
        }
        
        function initCharts() {
            lossChart = new Chart(document.getElementById('lossChart'), {
                type: 'line', data: { datasets: [{ label: 'Train Loss', borderColor: 'blue', fill: false, tension: 0.1 },
                                                  { label: 'Val Loss', borderColor: 'red', fill: false, tension: 0.1 }] },
                options: { responsive: true, maintainAspectRatio: true, scales: { y: { title: { display: true, text: 'Loss' } }, x: { title: { display: true, text: 'Step' } } } }
            });
            pplChart = new Chart(document.getElementById('pplChart'), {
                type: 'line', data: { datasets: [{ label: 'Train Perplexity', borderColor: 'green', fill: false, tension: 0.1 },
                                                 { label: 'Val Perplexity', borderColor: 'orange', fill: false, tension: 0.1 }] },
                options: { responsive: true, maintainAspectRatio: true, scales: { y: { title: { display: true, text: 'Perplexity' } }, x: { title: { display: true, text: 'Step' } } } }
            });
            lrChart = new Chart(document.getElementById('lrChart'), {
                type: 'line', data: { datasets: [{ label: 'Learning Rate', borderColor: 'purple', fill: false, tension: 0.1 }] },
                options: { responsive: true, maintainAspectRatio: true, scales: { y: { type: 'logarithmic', title: { display: true, text: 'LR' } }, x: { title: { display: true, text: 'Step' } } } }
            });
        }
        
        async function fetchAndUpdateCharts() {
            const res = await fetch('/api/metrics');
            const data = await res.json();
            rawMetrics = { steps: data.steps, train_loss: data.train_loss, val_loss: data.val_loss, train_ppl: data.train_ppl, val_ppl: data.val_ppl, lr: data.lr };
            applySmoothingAndUpdate();
        }
        
        function toggleSmoothing() { applySmoothingAndUpdate(); }
        
        async function generate() {
            const prompt = document.getElementById('prompt').value.trim();
            if(!prompt) { alert("Введите текст"); return; }
            const temp = parseFloat(document.getElementById('temp').value);
            const max_tokens = parseInt(document.getElementById('max_tokens').value);
            const top_p = parseFloat(document.getElementById('top_p').value);
            document.getElementById('generated').innerHTML = "⏳ Генерация...";
            document.getElementById('gen_status').innerText = "Генерация...";
            try {
                const res = await fetch('/generate', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({prompt, temperature: temp, max_new_tokens: max_tokens, top_p}) });
                const data = await res.json();
                if(data.error) document.getElementById('generated').innerHTML = "Ошибка: " + data.error;
                else document.getElementById('generated').innerHTML = data.generated_text;
            } catch(e) { document.getElementById('generated').innerHTML = "Ошибка: " + e.message; }
            document.getElementById('gen_status').innerText = "Готово";
        }
        
        async function reloadModel() {
            document.getElementById('gen_status').innerText = "Перезагрузка модели...";
            const res = await fetch('/reload_model', {method: 'POST'});
            const data = await res.json();
            document.getElementById('gen_status').innerText = data.message;
        }
        
        async function startTraining() {
            const params = {
                train_data: document.getElementById('train_data').value,
                val_data: document.getElementById('val_data').value,
                batch_size: parseInt(document.getElementById('batch_size').value),
                context_length: parseInt(document.getElementById('context_length').value),
                d_model: parseInt(document.getElementById('d_model').value),
                num_heads: parseInt(document.getElementById('num_heads').value),
                num_layers: parseInt(document.getElementById('num_layers').value),
                d_ff: document.getElementById('d_ff').value ? parseInt(document.getElementById('d_ff').value) : null,
                vocab_size: parseInt(document.getElementById('vocab_size').value),
                rope_theta: parseFloat(document.getElementById('rope_theta').value),
                no_rope: document.getElementById('no_rope').checked,
                lr: parseFloat(document.getElementById('lr').value),
                lr_min: parseFloat(document.getElementById('lr_min').value),
                warmup_steps: parseInt(document.getElementById('warmup_steps').value),
                weight_decay: parseFloat(document.getElementById('weight_decay').value),
                betas: [parseFloat(document.getElementById('beta1').value), parseFloat(document.getElementById('beta2').value)],
                eps: parseFloat(document.getElementById('eps').value),
                grad_clip: parseFloat(document.getElementById('grad_clip').value),
                max_iters: parseInt(document.getElementById('max_iters').value),
                eval_interval: parseInt(document.getElementById('eval_interval').value),
                log_interval: parseInt(document.getElementById('log_interval').value),
                save_interval: parseInt(document.getElementById('save_interval').value),
                resume: document.getElementById('resume').value || null,
                device: document.getElementById('device').value,
                seed: parseInt(document.getElementById('seed').value),
                use_wandb: document.getElementById('use_wandb').checked,
                wandb_project: document.getElementById('wandb_project').value,
                wandb_run_name: document.getElementById('wandb_run_name').value || null
            };
            const res = await fetch('/start_training', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(params) });
            const data = await res.json();
            alert(data.message);
            if(data.status === 'started') { startLogPolling(); startChartPolling(); }
        }
        
        function stopTraining() { fetch('/stop_training', {method: 'POST'}).then(res=>res.json()).then(data=>alert(data.message)); }
        
        function startLogPolling() { if(logInterval) clearInterval(logInterval); logInterval = setInterval(fetchLogs, 1000); fetchLogs(); }
        function stopLogPolling() { if(logInterval) clearInterval(logInterval); logInterval = null; }
        
        async function fetchLogs() {
            const res = await fetch('/training_logs');
            const data = await res.json();
            const logsDiv = document.getElementById('train-logs');
            logsDiv.innerText = data.logs.join('\\n');
            logsDiv.scrollTop = logsDiv.scrollHeight;
            document.getElementById('train-status').innerHTML = data.running ? "🏃 Обучение запущено" : (data.error ? "❌ Ошибка: "+data.error : "✅ Обучение завершено");
        }
        
        function startChartPolling() { if(chartInterval) clearInterval(chartInterval); chartInterval = setInterval(fetchAndUpdateCharts, 3000); }
        function stopChartPolling() { if(chartInterval) clearInterval(chartInterval); chartInterval = null; }
        
        function switchTab(tab) {
            document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
            document.getElementById(tab+'-tab').classList.add('active');
            document.querySelectorAll('.tab-button').forEach(btn => btn.classList.remove('active'));
            if(tab === 'train') { startLogPolling(); startChartPolling(); fetchAndUpdateCharts(); }
            else { stopLogPolling(); stopChartPolling(); }
        }
        
        window.onload = () => { initCharts(); switchTab('generate'); };
    </script>
</body>
</html>
"""

# ---------- Маршруты Flask ----------
@app.route('/')
def index():
    return render_template_string(MAIN_TEMPLATE)

@app.route('/generate', methods=['POST'])
def generate_text():
    if generation_model is None or tokenizer is None:
        return jsonify({'error': 'Модель не загружена'}), 503
    data = request.get_json()
    prompt = data.get('prompt', '').strip()
    if not prompt:
        return jsonify({'error': 'Пустой запрос'}), 400
    temperature = data.get('temperature', 0.8)
    max_new_tokens = data.get('max_new_tokens', 120)
    top_p = data.get('top_p', 0.9)
    try:
        token_ids = tokenizer.encode(prompt)
        prompt_tensor = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
        stop_id = tokenizer.inverted_vocab.get("<|endoftext|>".encode("UTF-8"))
        output_tensor = generate(
            generation_model, prompt_tensor,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=None,
            top_p=top_p,
            stop_token_id=stop_id
        )
        output_tokens = output_tensor[0].tolist()
        generated_text = tokenizer.decode(output_tokens)
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].lstrip()
        return jsonify({'generated_text': generated_text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/reload_model', methods=['POST'])
def reload_model():
    try:
        load_generation_model()
        return jsonify({'message': 'Модель успешно перезагружена'})
    except Exception as e:
        return jsonify({'message': f'Ошибка: {str(e)}'}), 500

@app.route('/start_training', methods=['POST'])
def start_training():
    global training_process, training_log_list, training_status
    with training_lock:
        if training_status["running"]:
            return jsonify({'status': 'already_running', 'message': 'Обучение уже запущено'})
        training_log_list.clear()
        training_log_list.append("=== Новая сессия обучения ===")
        training_status["error"] = None
    params = request.get_json()
    thread = threading.Thread(target=run_training_script, args=(params,), daemon=True)
    thread.start()
    return jsonify({'status': 'started', 'message': 'Обучение запущено'})

@app.route('/training_logs', methods=['GET'])
def training_logs():
    with training_lock:
        safe_logs = [str(x) for x in training_log_list]
        running = bool(training_status.get("running", False))
        error_val = training_status.get("error")
        if callable(error_val):
            error_val = str(error_val)
        elif error_val is not None and not isinstance(error_val, str):
            error_val = str(error_val)
        return jsonify({
            'logs': safe_logs,
            'running': running,
            'error': error_val
        })

@app.route('/stop_training', methods=['POST'])
def stop_training():
    global training_process
    with training_lock:
        if training_process and training_status.get("running", False):
            try:
                training_process.terminate()
                time.sleep(2)
                if training_process.poll() is None:
                    training_process.kill()
                training_status["running"] = False
                append_log("Обучение остановлено пользователем")
                return jsonify({'message': 'Процесс остановлен'})
            except Exception as e:
                return jsonify({'message': f'Ошибка: {e}'}), 500
        else:
            return jsonify({'message': 'Активный процесс не найден'}), 400

@app.route('/api/metrics', methods=['GET'])
def metrics_api():
    data = get_metrics_data()
    return jsonify(data)

if __name__ == '__main__':
    os.makedirs(os.path.join(current_dir, "Checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(current_dir, "Metrics"), exist_ok=True)
    load_generation_model()
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)