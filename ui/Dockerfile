FROM node:20

WORKDIR /ui

COPY . .

RUN npm ci --legacy-peer-deps

CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "3000"]