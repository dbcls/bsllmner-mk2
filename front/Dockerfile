FROM node:24.3.0-bookworm

ARG VERSION=dev

LABEL org.opencontainers.image.source="https://github.com/dbcls/bsllmner-mk2"
LABEL org.opencontainers.image.version="${VERSION}"
LABEL org.opencontainers.image.licenses="MIT"

WORKDIR /app
COPY package.json package-lock.json /app/
RUN npm ci

COPY . .

RUN npm run build

EXPOSE 3000

CMD ["npm", "run", "preview"]
