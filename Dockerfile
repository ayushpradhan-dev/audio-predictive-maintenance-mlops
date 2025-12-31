# Builder
# This stage creates a clean, locked environment in a folder.
FROM continuumio/miniconda3:latest AS builder

WORKDIR /app

# Copy the lock file.
COPY conda-linux-64.lock .

# Install conda-lock itself into the base environment.
RUN conda install -c conda-forge conda-lock -y

# Use conda-lock's own install command to create a new, clean
# environment from the lock file. The lock file is the last argument.
RUN conda-lock install -p /opt/conda/envs/audio-mlops conda-linux-64.lock && \
    # Clean up the cache.
    conda clean -afy


# Final Image
# This stage builds the final image for production.
FROM continuumio/miniconda3:latest

# Copy the entire, fully-built environment from the builder stage.
COPY --from=builder /opt/conda/envs/audio-mlops /opt/conda/envs/audio-mlops

# Set the working directory.
WORKDIR /app

# Copy the application source code and models.
COPY ./src /app/src
COPY ./models /app/models

# Expose the port.
EXPOSE 8000

# Set the ENTRYPOINT to the AWS Lambda Runtime Interface Client (RIC)
# tells the container to act like a lambda function.
ENTRYPOINT ["/opt/conda/envs/audio-mlops/bin/python", "-m", "awslambdaric"]

# Set the CMD to the path of the Mangum handler.
CMD ["src.main.handler"]