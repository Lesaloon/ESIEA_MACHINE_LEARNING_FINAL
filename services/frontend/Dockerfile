FROM nginx:1.27-alpine

COPY services/frontend/nginx.conf /etc/nginx/conf.d/default.conf
COPY services/frontend/index.html /usr/share/nginx/html/index.html

EXPOSE 80
