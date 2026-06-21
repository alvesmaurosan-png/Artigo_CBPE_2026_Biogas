FROM python:3.13.5-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

WORKDIR /workspace
COPY pyproject.toml requirements.lock README.md ./
RUN pip install --no-cache-dir -r requirements.lock
COPY . .
RUN pip install --no-cache-dir --no-deps -e .

CMD ["python", "-m", "cbpe", "reproduce", "--profile", "smoke"]

