# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install the package
RUN pip install .

# Make port 80 available to the world outside this container
EXPOSE 80

# Run the command to start your package
CMD ["python3", "-m", "mcda.mcda_run", "-c", "configuration.json"]