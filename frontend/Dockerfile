# The built frontend served by nginx.  docker build -t legalease-frontend ./frontend
FROM node:22-slim AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY . .
# The browser, not this container, calls the API: the chat components use http://127.0.0.1:8000.
RUN npm run build

FROM nginx:1.29-alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
