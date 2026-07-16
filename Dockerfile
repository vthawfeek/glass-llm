# Hugging Face Spaces stopped accepting `sdk: streamlit` for new Spaces, so the dashboard
# ships as a Docker Space. The exposed port must match `app_port:` in README.md.
FROM python:3.12-slim

# Spaces runs the container as UID 1000; give it a writable HOME for Streamlit's config.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
WORKDIR $HOME/app

# Dependencies first so edits to the app don't re-download the CPU torch wheel.
# requirements.txt carries the --extra-index-url for the +cpu build.
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

EXPOSE 8501
CMD ["streamlit", "run", "app_v2.py", "--server.port=8501", "--server.address=0.0.0.0"]
