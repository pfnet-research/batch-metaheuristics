FROM python:3.8.3-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    curl \
    gnuplot \
    git \
    htop \
    jq \
    procps \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install  --no-cache-dir \
    kurobako \
    Matplotlib \
    numpy \
    Opfunu \
    pandas \
    pyyaml \
    scikit-learn \
    seaborn \
    tqdm \
    cma==2.7 \
    pymoo \
    && pip3 install joblib --no-cache-dir \
    && pip3 install pytest==6.0.0 pytest-cov pytest-xdist --no-cache-dir \
    && pip3 install --upgrade pytest-dependency --no-cache-dir \
    && pip3 install git+https://github.com/ljvmiranda921/pyswarms.git --no-cache-dir

RUN curl -L https://github.com/sile/kurobako/releases/download/0.2.3/kurobako-0.2.3.linux-amd64\
    -o kurobako \
    && chmod +x kurobako \
    && mv kurobako /bin/
