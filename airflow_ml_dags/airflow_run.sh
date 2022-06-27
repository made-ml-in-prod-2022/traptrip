source .env && export FERNET_KEY=$(python -c "from cryptography.fernet import Fernet; FERNET_KEY = Fernet.generate_key().decode(); print(FERNET_KEY)") && \
    chmod +x ./docker-postgresql-multiple-databases/create-multiple-postgresql-databases.sh && \
    cd images/airflow-ml-base && docker build -t airflow-ml-base:latest . && cd - && \
    docker compose up -d --build
