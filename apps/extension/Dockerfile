# Extension dev container (Vite dev server with hot reload). Build context:
# apps/extension. docker-compose bind-mounts the source over /app and keeps an
# anonymous volume for node_modules so installed deps survive the mount.
FROM node:20-slim

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm install

COPY . .

EXPOSE 5173
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "5173"]
