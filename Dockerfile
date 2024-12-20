FROM node:18

RUN apt-get update && \
    apt-get install -y \
    build-essential \
    python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY package*.json ./

RUN npm install

ENV PORT 3000

ENV MODEL_URL 'https://storage.googleapis.com/chal/model.json'

COPY . .

EXPOSE 3000

CMD [ "npm", "start" ]