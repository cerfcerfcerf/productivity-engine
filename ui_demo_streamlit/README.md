# Streamlit Demo UI

This folder contains a presentable Streamlit UI that wraps the `productivity_engine` library.

## Windows PowerShell setup

```powershell
python -m venv venv
venv\Scripts\activate
pip install -e .
pip install -r ui_demo_streamlit/requirements.txt
streamlit run ui_demo_streamlit/app.py
```

## Demo script for judges

1. Launch the app with the command above.
2. Keep **Load demo dataset** checked.
3. Set:
   - Task type: `deadline`
   - Category: `study`
   - Priority: `2`
   - Now hour: `19`
   - Window: `18` to `21`
   - Deadline hours: `24`
4. Click **Run engine**.
5. Verify all sections render:
   - A) Data Summary
   - B) Risk Result
   - C) Recommended Schedule
   - D) Deadline Escalation
   - E) Evaluation (baseline/adaptive/comparison tables)

## Notes
- Supported uploads: `.csv` and `.json`
- If both upload and demo are available, demo dataset is used when the checkbox is enabled.
- Friendly validation errors are shown in-app for malformed input.
