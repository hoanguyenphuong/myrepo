# last updated Mar 25 2025, 11:00am
FROM python:3.12-slim

# Set non-interactive mode for apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Install Java, procps and bash
RUN apt-get update && \
    apt-get install -y --no-install-recommends default-jdk procps bash && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /bin/bash /bin/sh

# Set JAVA_HOME to default Java installation
ENV JAVA_HOME=/usr/lib/jvm/default-java
ENV PATH=$PATH:$JAVA_HOME/bin

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the default JupyterLab port
EXPOSE 8888

# Create a volume mount point for notebooks
VOLUME /app

# Enable JupyterLab via environment variable
ENV JUPYTER_ENABLE_LAB=yes

# Set up the command to run JupyterLab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--notebook-dir=/app"]