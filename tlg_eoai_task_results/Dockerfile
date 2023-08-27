# Use the tensorflow/tensorflow:latest-gpu base image
FROM tensorflow/tensorflow:latest-gpu

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt file to the container
COPY requirements.txt .

# Install Python dependencies from requirements.txt
RUN apt-get update && \
    apt-get install libgl1 -y && \
	apt-get install -y binutils libproj-dev gdal-bin libgdal-dev && \
    apt-get install -y python3-pip

RUN pip3 install -r requirements.txt --trusted-host pypi.org --trusted-host files.pythonhosted.org

# Install Jupyter
RUN pip3 install jupyter

ARG CPLUS_INCLUDE_PATH=/usr/include/gdal
ARG C_INCLUDE_PATH=/usr/include/gdal

# Expose port for Jupyter Hub
EXPOSE 8888

# Start Jupyter Hub when the container is run
ENTRYPOINT ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token="]