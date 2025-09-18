# Use a slim Python image to keep the final image size small.
FROM python:3.9-slim

# Set an environment variable for a stable working directory.
ENV APP_HOME=/app
WORKDIR $APP_HOME

# Copy the requirements file into the container first.
# This allows Docker to cache this layer if requirements.txt doesn't change.
COPY requirements.txt .

# Install the Python dependencies. This is where the installation happens.
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application source code from the 'src' directory on your machine
# into a directory named 'src' inside the container.
COPY src/ ./src/

# Expose the port on which your FastAPI application will run.
EXPOSE 8000

# This is the command to run your application when the container starts.
# It tells Uvicorn to look for the 'app' object in the 'api' module
# inside the 'src' package.
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]