# your-capstone-project/frontend/Dockerfile

# Use a lightweight, stable Nginx image as the base
FROM nginx:stable-alpine

# Copy your static frontend files into Nginx's default web directory
# The `public` folder contains your index.html, css, etc.
COPY ./public /usr/share/nginx/html

# Nginx's default configuration already exposes port 80.
# The Nginx server will automatically start when the container runs.

# Expose port 80 (standard HTTP port)
EXPOSE 80

# The default CMD of the Nginx image is sufficient to start the server
CMD ["nginx", "-g", "daemon off;"]