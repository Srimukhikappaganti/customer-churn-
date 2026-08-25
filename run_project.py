"""Run the main analysis pipeline in order."""
import subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
for script in ['src/analyze_dataset.py','src/topic_analysis.py','src/train_model.py']:
    print(f'\n=== Running {script} ===')
    subprocess.run([sys.executable, str(ROOT/script)], check=True)
print('\nPipeline complete. Start the dashboard with: streamlit run dashboard/app.py')
