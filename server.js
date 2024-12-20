require('dotenv').config();
const Hapi = require('@hapi/hapi');
const tf = require('@tensorflow/tfjs');
const crypto = require('crypto');
const { Firestore } = require('@google-cloud/firestore');

async function loadPredictionModel() {
    return tf.loadGraphModel(process.env.MODEL_URL);
}

async function savePredictionData(docId, predictionData) {
    const db = new Firestore();
    const predictionsCollection = db.collection('predictions');
    return predictionsCollection.doc(docId).set(predictionData);
}

async function fetchAllPredictionData() {
    const db = new Firestore();
    const predictionsCollection = db.collection('predictions');
    const allDocuments = await predictionsCollection.get();
    return allDocuments;
}

async function processImagePrediction(model, inputImage) {
    try {
        const tensor = tf.node
            .decodeJpeg(inputImage)
            .resizeNearestNeighbor([224, 224])
            .expandDims()
            .toFloat();

        const prediction = model.predict(tensor);
        const predictionScores = await prediction.data();
        const confidence = Math.max(...predictionScores) * 100;

        const predictionResult = confidence <= 50 ? 'Non-cancer' : 'Cancer';
        const predictionAdvice =
            predictionResult === 'Cancer'
                ? "Segera periksa ke dokter!"
                : "Penyakit kanker tidak terdeteksi.";

        return { predictionResult, predictionAdvice };
    } catch (error) {
        throw new Error('Terjadi kesalahan dalam melakukan prediksi');
    }
}

(async () => {
    const server = Hapi.server({
        port: process.env.PORT || 3000,
        host: '0.0.0.0',
        routes: {
            cors: {
                origin: ['*'],
            },
        },
    });

    const model = await loadPredictionModel();
    server.app.model = model;

    server.route([
        {
            path: '/predict',
            method: 'POST',
            handler: async (request, h) => {
                const { image } = request.payload;
                const { model } = request.server.app;

                const id = crypto.randomUUID();
                const createdAt = new Date().toISOString();

                try {
                    const { predictionResult, predictionAdvice } = await processImagePrediction(model, image);

                    const responseData = {
                        id,
                        result: predictionResult,
                        suggestion: predictionAdvice,
                        createdAt,
                    };

                    await savePredictionData(id, responseData);

                    return h
                        .response({
                            status: 'success',
                            message: 'Model is predicted successfully',
                            data: responseData,
                        })
                        .code(201);
                } catch (error) {
                    return h
                        .response({
                            status: 'fail',
                            message: error.message,
                        })
                        .code(400);
                }
            },
            options: {
                payload: {
                    allow: 'multipart/form-data',
                    multipart: true,
                    maxBytes: 1000000,
                },
            },
        },
        {
            path: '/predict/histories',
            method: 'GET',
            handler: async (request, h) => {
                const allDocuments = await fetchAllPredictionData();
                const formattedData = [];

                allDocuments.forEach((doc) => {
                    const docData = doc.data();
                    formattedData.push({
                        id: doc.id,
                        history: {
                            result: docData.result,
                            suggestion: docData.suggestion,
                            createdAt: docData.createdAt,
                        },
                    });
                });

                return h.response({
                    status: 'success',
                    data: formattedData,
                }).code(200);
            },
        },
    ]);

    server.ext('onPreResponse', (request, h) => {
        const response = request.response;

        if (response.isBoom) {
            const errorMessage = response.message || 'Internal Server Error';
            return h
                .response({
                    status: 'fail',
                    message: errorMessage,
                })
                .code(response.output.statusCode);
        }

        return h.continue;
    });

    await server.start();
    console.log(`Server started at: ${server.info.uri}`);
})();
