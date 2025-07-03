FROM node:24.3.0-bookworm

WORKDIR /app
COPY package.json package-lock.json /app/
RUN npm ci

COPY . .

RUN npm run build

EXPOSE 3000

CMD ["npm", "run", "preview"]
