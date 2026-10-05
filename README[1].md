# WingWatch (Streamlit)

Predicts student pilot landing performance. Student login can try what-if input changes; faculty login is read-only.

Files: app.py (interface), core.py (model, logins, access rules), requirements.txt, secrets_example.toml.

Deploy: upload these files to a public GitHub repo, then create the app on share.streamlit.io with main file app.py,
and paste the contents of secrets_example.toml (with your own passwords) into the app's Secrets box.

The data is synthetic. Replace make_data() in core.py and the STUDENTS records with real data.
