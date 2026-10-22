STUDENT PERFORMANCE PREDICTION WEBSITE
=======================================

FILES
- app.py: complete interactive frontend + machine learning app
- student_performance_dataset.csv: dataset file (keep in same folder)
- requirements.txt: Python package requirements

RUN LOCALLY IN VS CODE (Windows)
1. Put all three files in the same folder.
2. Open that folder in VS Code.
3. Open Terminal > New Terminal.
4. Install packages:
   py -m pip install -r requirements.txt
5. Start the website:
   py -m streamlit run app.py
6. The browser should open automatically. If not, open the Local URL shown in the terminal,
   usually http://localhost:8501

PAGES
- Dashboard: dataset totals and model accuracy comparison
- Predict Student Result: interactive form and predictions from both models
- Dataset Explorer: preview, filter, and download records
- Model Evaluation: accuracy, precision/recall, confusion matrices
- About Project: project explanation and limitations

IMPORTANT
This app expects these CSV columns:
Study_Hours, Attendance, Previous_Marks, Assignment_Score,
Internal_Marks, Sleep_Hours, Result

The provided project dataset is synthetic, for coursework demonstration only.
Do not use predictions for actual academic decisions.

ONLINE DEPLOYMENT
For a shareable public URL, push app.py, requirements.txt, and the CSV to a GitHub repository,
then deploy from Streamlit Community Cloud: https://share.streamlit.io/
Official guide: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
