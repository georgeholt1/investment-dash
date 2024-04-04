FROM continuumio/miniconda3

WORKDIR /usr/src/app

COPY . .

RUN conda env create -f conda_env.yml

SHELL ["conda", "run", "-n", "investment-dash", "/bin/bash", "-c"]

EXPOSE 8000

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "investment-dash", "gunicorn", "-b", "0.0.0.0:8000", "app:server"]
