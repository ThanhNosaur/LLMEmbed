"""Build a standalone Kaggle notebook from the tested Python sources."""
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent


def cell(kind, source):
    value = dict(cell_type=kind, metadata={}, source=source.strip() + '\n')
    if kind == 'code':
        value.update(execution_count=None, outputs=[])
    return value


def build():
    paths = sorted((ROOT / 'llmembed').glob('*.py')) + sorted((ROOT / 'tests').glob('*.py'))
    paths += [ROOT / 'requirements.txt', ROOT / 'README.md']
    files = {p.relative_to(ROOT).as_posix(): p.read_text(encoding='utf-8') for p in paths}
    setup = '''from pathlib import Path
import json, os, sys, subprocess
CODE = Path('/kaggle/working/llmembed_code')
CODE.mkdir(parents=True, exist_ok=True)
FILES = json.loads(PAYLOAD)
for name, content in FILES.items():
    path = CODE / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8')
os.chdir(CODE)
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
print('Code:', CODE)
'''.replace('PAYLOAD', repr(json.dumps(files, ensure_ascii=False)))
    cells = [cell('markdown', '''# LLMEmbed++ — Exp 0–5 trên Kaggle

Notebook tự chứa mã nguồn, không cần clone repo. **Settings → GPU T4 ×2; Internet → On.**
Với LLaMA-2, chấp nhận điều kiện model trên Hugging Face và tạo Kaggle Secret `HF_TOKEN` có quyền đọc model.

Mặc định chạy thử **256 mẫu, 2 epochs, seed 42, NF4** trên SST-2. Đổi cấu hình bên dưới để chạy đầy đủ.
NF4 là biến thể tiết kiệm VRAM; dùng FP16 cho đối chiếu không lượng tử hóa. Không có kết quả dự đoán sẵn.
'''), cell('code', setup), cell('code', '''# Cài thư viện trước khi import transformers. Giữ torch/CUDA có sẵn của Kaggle.
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', str(CODE / 'requirements.txt')], check=True)
import torch
if not torch.cuda.is_available():
    raise RuntimeError('Bật GPU trong Settings của notebook rồi chạy lại.')
# The previous Kaggle run loaded CPU-only bitsandbytes despite CUDA torch.
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-cache-dir', '--force-reinstall',
                '--no-deps', '--index-url', 'https://pypi.org/simple', 'bitsandbytes==0.50.2'], check=True)
subprocess.run([sys.executable, '-c',
    'import importlib.metadata; from bitsandbytes.cextension import lib; '
    'assert lib.compiled_with_cuda, "bitsandbytes CUDA backend unavailable"; '
    'print("bitsandbytes:", importlib.metadata.version("bitsandbytes"))'], check=True)
print('torch:', torch.__version__)
print('GPUs:', [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
'''), cell('code', '''# Token chỉ ở biến môi trường, không ghi vào notebook, config hay kết quả.
if not os.environ.get('HF_TOKEN'):
    try:
        from kaggle_secrets import UserSecretsClient
        os.environ['HF_TOKEN'] = UserSecretsClient().get_secret('HF_TOKEN')
        print('Đã nạp HF_TOKEN từ Kaggle Secrets.')
    except Exception:
        print('Chưa có HF_TOKEN. LLaMA-2 cần quyền truy cập; Exp 5c dùng Qwen công khai.')
'''), cell('markdown', '''## Cấu hình chạy

- `smoke`: kiểm tra pipeline; số liệu không dùng báo cáo nghiên cứu.
- `full`: toàn bộ train/test, 50 epochs, 3 seeds. Nên chạy **một dataset mỗi phiên**.
- Exp 5A chính là Exp 4; chọn thêm `exp5b`, `exp5c` để đổi backbone.
- `mr` tải IMDb và chia lại 40k/10k khi `MR_SOURCE='imdb_resplit'`. **Đây là proxy**, chưa phải bộ chia MR của tác giả.
- R8/R52 tải bản TextGCN có đúng số mẫu; chưa xác nhận tiền xử lý trùng repo LLMEmbed.
'''), cell('code', '''RUN_MODE = 'smoke'  # 'full' khi đã kiểm tra chạy được
DATASETS = ['sst2']  # 'mr', 'agnews', 'r8', 'r52'
EXPERIMENTS = ['exp0', 'exp1', 'exp2', 'exp3', 'exp4']
PRECISION = 'nf4'  # 'fp16' trên T4 x2 để so sánh không lượng tử hóa
PROTOCOL = 'controlled'
MR_SOURCE = 'imdb_resplit'
DATA_DIR = None  # Ví dụ '/kaggle/input/my-data'; chứa mr/train.json, mr/test.json
REUSE_CACHE = None  # Ví dụ '/kaggle/input/previous-run/llmembed_runs/full/features'
RESUME_DIR = None  # Thư mục output phiên cũ nếu cần tiếp tục cả classifier
STAGE = 'all'  # 'data', 'extract', 'train', 'all'
OUTPUT = Path('/kaggle/working/llmembed_runs') / RUN_MODE
LIMIT = 256 if RUN_MODE == 'smoke' else 0
EPOCHS = 2 if RUN_MODE == 'smoke' else 50
SEEDS = [42] if RUN_MODE == 'smoke' else [42, 43, 44]

command = [sys.executable, '-u', '-m', 'llmembed.run',
    '--datasets', *DATASETS, '--experiments', *EXPERIMENTS,
    '--output', str(OUTPUT), '--precision', PRECISION, '--protocol', PROTOCOL,
    '--mr-source', MR_SOURCE, '--limit', str(LIMIT), '--epochs', str(EPOCHS),
    '--seeds', *map(str, SEEDS), '--stage', STAGE,
    '--extract-batch', '2', '--encoder-batch', '16', '--micro-batch', '64',
    '--chunk-size', '256', '--overlap', '32', '--max-chunks', '0']
if DATA_DIR:
    command += ['--data-dir', DATA_DIR]
if REUSE_CACHE:
    command += ['--reuse-cache', REUSE_CACHE]
if not os.environ.get('HF_TOKEN') and any(experiment != 'exp5c' for experiment in EXPERIMENTS):
    raise RuntimeError('Bật HF_TOKEN trong Add-ons → Secrets trước khi chạy LLaMA-2.')
print(' '.join(command))
subprocess.run(command + ['--plan'], check=True)
'''), cell('code', '''# Tùy chọn: tiếp tục output đã lưu ở một Kaggle Dataset.
if RESUME_DIR:
    import shutil
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError('Chọn OUTPUT trống trước khi khôi phục phiên cũ để tránh ghi đè kết quả.')
    shutil.copytree(RESUME_DIR, OUTPUT, dirs_exist_ok=True)

# Kiểm tra bằng mô hình tí hon, không tải model 7B.
subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(CODE / 'tests'), '-v'], check=True)
'''), cell('code', '''# Run: dừng ngay khi có lỗi; cache/checkpoint đã hoàn thành vẫn được giữ.
subprocess.run(command, check=True)
'''), cell('code', '''# Kết quả thật; accuracy ở thang 0..1.
import pandas as pd
from IPython.display import display, Image
if (OUTPUT / 'ablation.csv').exists():
    display(pd.read_csv(OUTPUT / 'ablation.csv'))
    display(pd.read_csv(OUTPUT / 'metrics.csv'))
    for path in list((OUTPUT / 'results').glob('*/source_weights.png'))[:3]:
        display(Image(filename=str(path)))
    for path in list((OUTPUT / 'explanations').glob('*/tokens_0.png'))[:2]:
        display(Image(filename=str(path)))
print('Lưu toàn bộ output bằng Save Version để dùng lại cache/checkpoint:', OUTPUT)
'''), cell('markdown', '''## Đọc kết quả và chạy tiếp

`metrics.csv`: từng seed; `ablation.csv`: mean/std và delta theo điểm phần trăm so với Exp 0.
`average_five_datasets.csv` chỉ có khi chạy đủ 5 dataset cùng cấu hình/seeds.
`results/*/best.pt`, `last.pt`, `predictions.csv`, `config.json` lưu mô hình, dự đoán và cấu hình.
`features/*/state.json` lưu nguồn model, tỷ lệ token giữ lại và tiến độ cache.

Để tiếp tục sau khi Kaggle kết thúc phiên: Save Version, thêm output cũ bằng Add Input,
đặt `RESUME_DIR` hoặc `REUSE_CACHE` ở trên. Cache chưa hoàn thành cần được sao chép vào OUTPUT ghi được.
`STAGE='train'` chỉ dùng cache hoàn thành, không trích xuất lại.

Trọng số token là cosine cố định, không phải token attention được học; prompt không sinh chuỗi suy luận.
Tốc độ/độ chính xác trong bản thảo là giả thuyết cần đo. Điện năng mặc định để trống.
Xem `README.md` trong thư mục code để chạy sweep 15 fusion, 4 strides và chế độ tương thích mã tác giả.
''')]
    notebook = dict(cells=cells, metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'),
                    language_info=dict(name='python', version='3.11'), accelerator='GPU'), nbformat=4, nbformat_minor=4)
    path = ROOT / 'llmembed_plusplus.ipynb'
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding='utf-8')
    with zipfile.ZipFile(ROOT / 'llmembed_plusplus.zip', 'w', zipfile.ZIP_DEFLATED) as bundle:
        for file in paths + [path, Path(__file__)]:
            bundle.write(file, file.relative_to(ROOT))
    print(path)
    print(ROOT / 'llmembed_plusplus.zip')


if __name__ == '__main__':
    build()
