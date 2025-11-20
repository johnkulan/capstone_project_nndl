class DataLoader {
    constructor() {
        this.trainData = null;
        this.testData = null;
        this.processedTrainData = null;
        this.processedTestData = null;
        this.balancedTrainData = null;
        this.featureNames = [];
        this.sectorCategories = [
            'Energy', 'Healthcare', 'Technology', 'Industrials', 
            'Financials', 'Consumer Discretionary', 'Real Estate',
            'Communication Services', 'Consumer Staples', 'Utilities', 'Materials'
        ];
        this.numericColumns = [
            'Market Cap', 'IPO Price', 'Price Target', 'Beta (5Y)', 'Revenue',
            'Rev. Growth', 'Net Income', 'EPS', 'Debt / Equity', 'Current Ratio',
            'Quick Ratio', 'Profit Margin', 'ROE', 'ROA'
        ];
    }

    log(message) {
        const logElement = document.getElementById('consoleLog');
        if (logElement) {
            const timestamp = new Date().toLocaleTimeString();
            const logEntry = document.createElement('div');
            logEntry.className = 'log-entry';
            logEntry.innerHTML = `<span class="log-time">[${timestamp}]</span> ${message}`;
            logElement.appendChild(logEntry);
            logElement.scrollTop = logElement.scrollHeight;
        }
        console.log(message);
    }

    async loadCSV(filename) {
        return new Promise((resolve, reject) => {
            this.log(`Loading ${filename}...`);
            
            Papa.parse(`data/${filename}`, {
                download: true,
                header: true,
                dynamicTyping: true,
                skipEmptyLines: true,
                complete: (results) => {
                    if (results.errors.length > 0) {
                        this.log(`❌ Error parsing ${filename}: ${results.errors[0].message}`);
                        reject(new Error(`CSV parsing errors: ${results.errors.map(e => e.message).join(', ')}`));
                    } else {
                        this.log(`✅ Successfully loaded ${filename} with ${results.data.length} rows`);
                        resolve(results.data);
                    }
                },
                error: (error) => {
                    this.log(`❌ Failed to load ${filename}: ${error}`);
                    reject(error);
                }
            });
        });
    }

    async loadData() {
        try {
            this.log('📥 Starting data loading process...');
            
            // Load both datasets
            this.trainData = await this.loadCSV('train_startups.csv');
            this.testData = await this.loadCSV('test_startups.csv');

            this.log('✅ All data files loaded successfully!');
            return true;
        } catch (error) {
            this.log(`❌ Data loading failed: ${error.message}`);
            return false;
        }
    }

    preprocessData() {
        this.log('🔧 Starting data preprocessing...');
        
        if (!this.trainData || !this.testData) {
            throw new Error('Data not loaded. Call loadData() first.');
        }

        // Process both datasets
        this.processedTrainData = this.processDataset(this.trainData, 'training');
        this.processedTestData = this.processDataset(this.testData, 'testing');

        this.log('✅ Data preprocessing completed!');
        
        return {
            train: this.processedTrainData,
            test: this.processedTestData,
            featureNames: this.featureNames,
            sectors: this.sectorCategories
        };
    }

    processDataset(dataset, datasetName) {
        this.log(`Processing ${datasetName} dataset...`);
        
        const features = [];
        const labels = [];
        const sectors = [];

        dataset.forEach((row, index) => {
            // Skip rows with missing labels
            if (row.high_risk === undefined || row.high_risk === null) {
                this.log(`⚠️ Skipping row ${index} in ${datasetName} dataset: missing label`);
                return;
            }

            const featureVector = [];

            // Process numeric features
            this.numericColumns.forEach(col => {
                let value = row[col];
                
                // Handle missing values
                if (value === undefined || value === null || isNaN(value)) {
                    value = 0;
                }
                
                featureVector.push(Number(value));
            });

            // One-hot encode sector
            const sectorOneHot = new Array(this.sectorCategories.length).fill(0);
            const sectorIndex = this.sectorCategories.indexOf(row.Sector);
            
            if (sectorIndex !== -1) {
                sectorOneHot[sectorIndex] = 1;
            }
            
            featureVector.push(...sectorOneHot);

            // Set feature names on first row
            if (this.featureNames.length === 0 && index === 0) {
                this.featureNames = [...this.numericColumns];
                this.sectorCategories.forEach(sector => {
                    this.featureNames.push(`Sector_${sector}`);
                });
                this.log(`📊 Feature names initialized: ${this.featureNames.length} features total`);
            }

            features.push(featureVector);
            labels.push(Number(row.high_risk));
            sectors.push(row.Sector || 'Unknown');
        });

        this.log(`✅ Processed ${features.length} samples from ${datasetName} dataset`);

        return {
            features: tf.tensor2d(features),
            labels: tf.tensor1d(labels),
            sectors: sectors
        };
    }

    balanceTrainingData() {
        this.log('⚖️ Balancing training data using undersampling...');
        
        if (!this.processedTrainData) {
            throw new Error('Training data not processed. Call preprocessData() first.');
        }

        // Get array values from tensors
        const trainFeaturesArray = this.processedTrainData.features.arraySync();
        const trainLabelsArray = this.processedTrainData.labels.arraySync();
        const trainSectorsArray = this.processedTrainData.sectors;

        // Separate high risk and low risk samples
        const highRiskSamples = [];
        const lowRiskSamples = [];

        trainLabelsArray.forEach((label, index) => {
            const sample = {
                features: trainFeaturesArray[index],
                label: label,
                sector: trainSectorsArray[index]
            };
            
            if (label === 1) {
                highRiskSamples.push(sample);
            } else {
                lowRiskSamples.push(sample);
            }
        });

        this.log(`📊 Before balancing: ${lowRiskSamples.length} low risk vs ${highRiskSamples.length} high risk samples`);

        // Undersample low risk class to match high risk class count
        const balancedLowRiskSamples = this.shuffleArray(lowRiskSamples).slice(0, highRiskSamples.length);

        this.log(`📊 After balancing: ${balancedLowRiskSamples.length} low risk vs ${highRiskSamples.length} high risk samples`);

        // Combine balanced datasets
        const balancedSamples = [...balancedLowRiskSamples, ...highRiskSamples];
        const shuffledSamples = this.shuffleArray(balancedSamples);

        // Convert back to tensors
        const balancedFeatures = [];
        const balancedLabels = [];
        const balancedSectors = [];

        shuffledSamples.forEach(sample => {
            balancedFeatures.push(sample.features);
            balancedLabels.push(sample.label);
            balancedSectors.push(sample.sector);
        });

        this.balancedTrainData = {
            features: tf.tensor2d(balancedFeatures),
            labels: tf.tensor1d(balancedLabels),
            sectors: balancedSectors
        };

        this.log(`✅ Training data balanced! Total samples: ${balancedFeatures.length}`);
        return this.balancedTrainData;
    }

    shuffleArray(array) {
        const shuffled = [...array];
        for (let i = shuffled.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
        }
        return shuffled;
    }

    normalizeData() {
        this.log('📐 Normalizing data to [0, 1] range...');
        
        if (!this.balancedTrainData || !this.processedTestData) {
            throw new Error('Data not balanced. Call balanceTrainingData() first.');
        }

        const trainFeatures = this.balancedTrainData.features;
        const testFeatures = this.processedTestData.features;

        // Calculate min and max from balanced training data only
        const featureMin = trainFeatures.min(0);
        const featureMax = trainFeatures.max(0);
        const featureRange = featureMax.sub(featureMin);

        // Avoid division by zero for constant features
        const safeRange = featureRange.where(
            featureRange.greater(0),
            tf.onesLike(featureRange)
        );

        // Normalize features
        const normalizedTrainFeatures = trainFeatures
            .sub(featureMin)
            .div(safeRange);

        const normalizedTestFeatures = testFeatures
            .sub(featureMin)
            .div(safeRange);

        this.log('✅ Data normalization completed!');

        return {
            train: {
                features: normalizedTrainFeatures,
                labels: this.balancedTrainData.labels,
                sectors: this.balancedTrainData.sectors
            },
            test: {
                features: normalizedTestFeatures,
                labels: this.processedTestData.labels,
                sectors: this.processedTestData.sectors
            },
            featureNames: this.featureNames
        };
    }

    getDataSummary() {
        if (!this.processedTrainData) {
            return 'No data processed yet.';
        }

        const trainLabels = this.processedTrainData.labels.arraySync();
        const testLabels = this.processedTestData.labels.arraySync();

        const trainHighRisk = trainLabels.filter(label => label === 1).length;
        const testHighRisk = testLabels.filter(label => label === 1).length;

        const summary = {
            trainSamples: trainLabels.length,
            testSamples: testLabels.length,
            trainHighRisk: trainHighRisk,
            trainLowRisk: trainLabels.length - trainHighRisk,
            testHighRisk: testHighRisk,
            testLowRisk: testLabels.length - testHighRisk,
            featureCount: this.featureNames.length
        };

        this.log(`📈 Data Summary:
        • Training samples: ${summary.trainSamples} (${summary.trainHighRisk} high risk, ${summary.trainLowRisk} low risk)
        • Test samples: ${summary.testSamples} (${summary.testHighRisk} high risk, ${summary.testLowRisk} low risk)
        • Features: ${summary.featureCount}`);

        if (this.balancedTrainData) {
            const balancedLabels = this.balancedTrainData.labels.arraySync();
            const balancedHighRisk = balancedLabels.filter(label => label === 1).length;
            const balancedLowRisk = balancedLabels.filter(label => label === 0).length;
            
            this.log(`📈 Balanced Training: ${balancedLabels.length} samples (${balancedHighRisk} high risk, ${balancedLowRisk} low risk)`);
        }

        return summary;
    }
}

class GRUModel {
    constructor() {
        this.model = null;
        this.lstmModel = null;
        this.isTraining = false;
    }

    log(message) {
        const logElement = document.getElementById('consoleLog');
        if (logElement) {
            const timestamp = new Date().toLocaleTimeString();
            const logEntry = document.createElement('div');
            logEntry.className = 'log-entry';
            logEntry.innerHTML = `<span class="log-time">[${timestamp}]</span> ${message}`;
            logElement.appendChild(logEntry);
            logElement.scrollTop = logElement.scrollHeight;
        }
        console.log(message);
    }

    buildModel(inputShape, modelType = 'gru', units = 32) {
        this.log(`🏗️ Building ${modelType.toUpperCase()} model...`);
        
        const model = tf.sequential();

        if (modelType === 'gru') {
            model.add(tf.layers.gru({
                units: units,
                returnSequences: false,
                inputShape: [1, inputShape],
                activation: 'tanh'
            }));
        } else if (modelType === 'lstm') {
            model.add(tf.layers.lstm({
                units: units,
                returnSequences: false,
                inputShape: [1, inputShape],
                activation: 'tanh'
            }));
        }

        model.add(tf.layers.dropout({ rate: 0.3 }));
        model.add(tf.layers.dense({ units: 16, activation: 'relu' }));
        model.add(tf.layers.dropout({ rate: 0.2 }));
        model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

        model.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'binaryCrossentropy',
            metrics: ['accuracy']
        });

        this.log(`✅ ${modelType.toUpperCase()} model built successfully!`);
        return model;
    }

    async trainModel(trainData, testData, epochs = 3, batchSize = 16) {
        this.log('🎯 Starting model training process...');
        this.isTraining = true;

        // Reshape data for RNN (samples, timesteps, features)
        const trainFeatures = trainData.features.reshape([
            trainData.features.shape[0], 1, trainData.features.shape[1]
        ]);
        const testFeatures = testData.features.reshape([
            testData.features.shape[0], 1, testData.features.shape[1]
        ]);

        try {
            // Train GRU model
            const inputShape = trainData.features.shape[1];
            this.model = this.buildModel(inputShape, 'gru', 32);
            
            this.log('🧠 Training GRU model...');
            await this.model.fit(trainFeatures, trainData.labels, {
                epochs: epochs,
                batchSize: batchSize,
                validationData: [testFeatures, testData.labels],
                callbacks: {
                    onEpochEnd: async (epoch, logs) => {
                        this.log(`📊 GRU Epoch ${epoch + 1}/${epochs} - Loss: ${logs.loss.toFixed(4)}, Acc: ${logs.acc.toFixed(4)}, Val Loss: ${logs.val_loss.toFixed(4)}, Val Acc: ${logs.val_acc.toFixed(4)}`);
                        
                        // Memory management
                        if (epoch % 5 === 0) {
                            await tf.nextFrame();
                        }
                    }
                }
            });

            // Train LSTM model for comparison
            this.log('🧠 Training LSTM model for comparison...');
            this.lstmModel = this.buildModel(inputShape, 'lstm', 32);
            
            await this.lstmModel.fit(trainFeatures, trainData.labels, {
                epochs: epochs,
                batchSize: batchSize,
                validationData: [testFeatures, testData.labels],
                callbacks: {
                    onEpochEnd: async (epoch, logs) => {
                        this.log(`📊 LSTM Epoch ${epoch + 1}/${epochs} - Loss: ${logs.loss.toFixed(4)}, Acc: ${logs.acc.toFixed(4)}, Val Loss: ${logs.val_loss.toFixed(4)}, Val Acc: ${logs.val_acc.toFixed(4)}`);
                    }
                }
            });

            this.log('✅ Model training completed!');
            return { success: true };
        } finally {
            this.isTraining = false;
            trainFeatures.dispose();
            testFeatures.dispose();
        }
    }

    async evaluateModel(testData) {
        if (!this.model || !this.lstmModel) {
            throw new Error('Models not trained. Train the models first.');
        }

        this.log('📊 Evaluating models...');
        
        const testFeatures = testData.features.reshape([
            testData.features.shape[0], 1, testData.features.shape[1]
        ]);

        try {
            // Evaluate GRU model
            const gruEvaluation = this.model.evaluate(testFeatures, testData.labels);
            const gruLoss = gruEvaluation[0].dataSync()[0];
            const gruAccuracy = gruEvaluation[1].dataSync()[0];

            // Evaluate LSTM model
            const lstmEvaluation = this.lstmModel.evaluate(testFeatures, testData.labels);
            const lstmLoss = lstmEvaluation[0].dataSync()[0];
            const lstmAccuracy = lstmEvaluation[1].dataSync()[0];

            // Get detailed predictions for additional metrics
            const gruPredValues = await this.model.predict(testFeatures).dataSync();
            const lstmPredValues = await this.lstmModel.predict(testFeatures).dataSync();
            const trueLabels = await testData.labels.dataSync();

            // Calculate additional metrics
            const gruMetrics = this.calculateDetailedMetrics(gruPredValues, trueLabels);
            const lstmMetrics = this.calculateDetailedMetrics(lstmPredValues, trueLabels);

            this.log(`✅ GRU Evaluation - Loss: ${gruLoss.toFixed(4)}, Accuracy: ${gruAccuracy.toFixed(4)}`);
            this.log(`✅ LSTM Evaluation - Loss: ${lstmLoss.toFixed(4)}, Accuracy: ${lstmAccuracy.toFixed(4)}`);
            this.log(`📈 GRU Detailed - Precision: ${gruMetrics.precision.toFixed(4)}, Recall: ${gruMetrics.recall.toFixed(4)}`);
            this.log(`📈 LSTM Detailed - Precision: ${lstmMetrics.precision.toFixed(4)}, Recall: ${lstmMetrics.recall.toFixed(4)}`);

            // Show some sample predictions
            const samplePredictions = await this.getSamplePredictions(testData);

            // Update UI with metrics
            this.updateMetricsUI({
                gru: {
                    loss: gruLoss,
                    accuracy: gruAccuracy,
                    precision: gruMetrics.precision,
                    recall: gruMetrics.recall
                },
                lstm: {
                    loss: lstmLoss,
                    accuracy: lstmAccuracy,
                    precision: lstmMetrics.precision,
                    recall: lstmMetrics.recall
                }
            });

            return {
                gru: {
                    loss: gruLoss,
                    accuracy: gruAccuracy,
                    ...gruMetrics
                },
                lstm: {
                    loss: lstmLoss,
                    accuracy: lstmAccuracy,
                    ...lstmMetrics
                }
            };
        } finally {
            testFeatures.dispose();
        }
    }

    calculateDetailedMetrics(predictions, trueLabels) {
        let truePositives = 0;
        let falsePositives = 0;
        let trueNegatives = 0;
        let falseNegatives = 0;

        predictions.forEach((pred, i) => {
            const predictedClass = pred > 0.5 ? 1 : 0;
            const trueClass = trueLabels[i];

            if (predictedClass === 1 && trueClass === 1) truePositives++;
            else if (predictedClass === 1 && trueClass === 0) falsePositives++;
            else if (predictedClass === 0 && trueClass === 0) trueNegatives++;
            else if (predictedClass === 0 && trueClass === 1) falseNegatives++;
        });

        const precision = truePositives / (truePositives + falsePositives) || 0;
        const recall = truePositives / (truePositives + falseNegatives) || 0;

        return {
            truePositives,
            falsePositives,
            trueNegatives,
            falseNegatives,
            precision,
            recall
        };
    }

    async getSamplePredictions(testData, count = 3) {
        if (!this.model) {
            throw new Error('Model not trained. Train the model first.');
        }

        this.log(`🔍 Generating ${count} sample predictions...`);

        const testFeatures = testData.features.reshape([
            testData.features.shape[0], 1, testData.features.shape[1]
        ]);
        const predictions = this.model.predict(testFeatures);
        const predValues = await predictions.dataSync();
        const trueLabels = await testData.labels.dataSync();

        // Select first few samples
        const indices = [];
        for (let i = 0; i < Math.min(count, testData.features.shape[0]); i++) {
            indices.push(i);
        }

        const results = indices.map(idx => ({
            index: idx,
            predictedProbability: predValues[idx],
            predictedClass: predValues[idx] > 0.5 ? 1 : 0,
            trueClass: trueLabels[idx],
            sector: testData.sectors ? testData.sectors[idx] : 'Unknown',
            confidence: Math.max(predValues[idx], 1 - predValues[idx]),
            correct: (predValues[idx] > 0.5) === (trueLabels[idx] === 1)
        }));

        this.displaySamplePredictions(results);
        
        testFeatures.dispose();
        predictions.dispose();

        return results;
    }

    displaySamplePredictions(results) {
        const container = document.getElementById('predictionsContainer');
        container.innerHTML = '<h3 style="color: #4a5568; margin-bottom: 15px;">🎯 Sample Predictions</h3>';

        results.forEach(result => {
            const card = document.createElement('div');
            card.className = `prediction-card ${result.correct ? 'prediction-correct' : 'prediction-incorrect'}`;
            
            const riskLevel = result.predictedClass === 1 ? 'HIGH RISK' : 'LOW RISK';
            const actualLevel = result.trueClass === 1 ? 'HIGH RISK' : 'LOW RISK';
            const confidencePercent = (result.confidence * 100).toFixed(1);
            
            card.innerHTML = `
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <strong style="font-size: 1.1em;">Sample #${result.index + 1}</strong>
                    <span style="background: ${result.correct ? '#68d391' : '#fc8181'}; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.8em;">
                        ${result.correct ? '✓ CORRECT' : '✗ INCORRECT'}
                    </span>
                </div>
                <div style="color: #4a5568; margin-bottom: 5px;">
                    <strong>Sector:</strong> ${result.sector}
                </div>
                <div style="color: ${result.predictedClass === 1 ? '#e53e3e' : '#38a169'}; margin-bottom: 5px;">
                    <strong>Predicted:</strong> ${riskLevel} (${confidencePercent}% confidence)
                </div>
                <div style="color: #4a5568;">
                    <strong>Actual:</strong> ${actualLevel}
                </div>
                <div style="margin-top: 8px; font-size: 0.9em; color: #718096;">
                    Probability: ${result.predictedProbability.toFixed(4)}
                </div>
            `;
            
            container.appendChild(card);
        });
    }

    updateMetricsUI(metrics) {
        // Update GRU metrics
        document.getElementById('gruAccuracy').textContent = metrics.gru.accuracy.toFixed(4);
        document.getElementById('gruLoss').textContent = metrics.gru.loss.toFixed(4);
        document.getElementById('gruPrecision').textContent = metrics.gru.precision.toFixed(4);
        document.getElementById('gruRecall').textContent = metrics.gru.recall.toFixed(4);

        // Update LSTM metrics
        document.getElementById('lstmAccuracy').textContent = metrics.lstm.accuracy.toFixed(4);
        document.getElementById('lstmLoss').textContent = metrics.lstm.loss.toFixed(4);
        document.getElementById('lstmPrecision').textContent = metrics.lstm.precision.toFixed(4);
        document.getElementById('lstmRecall').textContent = metrics.lstm.recall.toFixed(4);
    }

    async saveModel() {
        if (!this.model) {
            throw new Error('No model to save.');
        }

        this.log('💾 Saving GRU model...');
        await this.model.save('downloads://startup-risk-gru-model');
        this.log('✅ GRU model saved successfully!');
    }

    dispose() {
        if (this.model) {
            this.model.dispose();
        }
        if (this.lstmModel) {
            this.lstmModel.dispose();
        }
    }
}

class StartupRiskApp {
    constructor() {
        this.dataLoader = new DataLoader();
        this.model = new GRUModel();
        this.normalizedData = null;
        
        this.initializeApp();
    }

    log(message) {
        const logElement = document.getElementById('consoleLog');
        if (logElement) {
            const timestamp = new Date().toLocaleTimeString();
            const logEntry = document.createElement('div');
            logEntry.className = 'log-entry';
            logEntry.innerHTML = `<span class="log-time">[${timestamp}]</span> ${message}`;
            logElement.appendChild(logEntry);
            logElement.scrollTop = logElement.scrollHeight;
        }
        console.log(message);
    }

    updateStatus(message) {
        const statusElement = document.getElementById('status');
        if (statusElement) {
            statusElement.innerHTML = message;
        }
    }

    initializeApp() {
        this.log('🚀 Startup Risk Prediction Demo Initialized');
        this.log('💡 Click "Start Process" to begin automated training');
        this.updateStatus('Ready to start. Click "Start Process" to begin automated training.');
        
        this.setupEventListeners();
    }

    setupEventListeners() {
        document.getElementById('startBtn').addEventListener('click', () => this.startProcess());
        document.getElementById('loadDataBtn').addEventListener('click', () => this.loadData());
        document.getElementById('trainBtn').addEventListener('click', () => this.trainModel());
        document.getElementById('evaluateBtn').addEventListener('click', () => this.evaluateModel());
        document.getElementById('saveModelBtn').addEventListener('click', () => this.saveModel());
        document.getElementById('resetBtn').addEventListener('click', () => this.resetApp());
    }

    setButtonState(buttonId, enabled) {
        const button = document.getElementById(buttonId);
        if (button) {
            button.disabled = !enabled;
            if (buttonId === 'startBtn') {
                const loading = button.querySelector('.loading');
                if (loading) {
                    loading.style.display = enabled ? 'inline-block' : 'none';
                }
            }
        }
    }

    async startProcess() {
        this.log('🎯 === STARTING AUTOMATED TRAINING PROCESS ===');
        this.updateStatus('<span class="loading"></span>Starting automated process...');
        
        // Disable start button and enable next steps
        this.setButtonState('startBtn', false);
        
        try {
            // Step 1: Load Data
            this.setButtonState('loadDataBtn', true);
            const dataLoaded = await this.loadData();
            if (!dataLoaded) {
                throw new Error('Failed to load data');
            }
            
            // Step 2: Balance Data
            await new Promise(resolve => setTimeout(resolve, 1000));
            this.setButtonState('trainBtn', true);
            const modelTrained = await this.trainModel();
            if (!modelTrained) {
                throw new Error('Failed to train model');
            }
            
            // Step 3: Evaluate Model
            await new Promise(resolve => setTimeout(resolve, 1000));
            this.setButtonState('evaluateBtn', true);
            await this.evaluateModel();
            
            // Step 4: Enable model saving
            this.setButtonState('saveModelBtn', true);
            
            this.log('✅ === AUTOMATED PROCESS COMPLETED SUCCESSFULLY ===');
            this.updateStatus('✅ Process completed! Model is trained and ready for predictions.');
            
        } catch (error) {
            this.log(`❌ Process failed: ${error.message}`);
            this.updateStatus(`❌ Process failed: ${error.message}`);
            // Re-enable start button to allow retry
            this.setButtonState('startBtn', true);
        }
    }

    async loadData() {
        this.updateStatus('<span class="loading"></span>Loading and preprocessing data...');
        this.log('📥 Starting data loading process...');
        
        try {
            const success = await this.dataLoader.loadData();
            if (!success) {
                this.log('❌ Failed to load data files');
                return false;
            }

            this.log('🔧 Preprocessing data...');
            this.dataLoader.preprocessData();
            
            this.log('⚖️ Balancing training data...');
            this.dataLoader.balanceTrainingData();
            
            this.normalizedData = this.dataLoader.normalizeData();
            
            const summary = this.dataLoader.getDataSummary();
            this.log(`📊 Dataset loaded: ${summary.trainSamples} original training, ${this.normalizedData.train.features.shape[0]} balanced training, ${summary.testSamples} test samples`);
            
            this.updateStatus('✅ Data loaded, balanced, and preprocessed successfully');
            return true;
        } catch (error) {
            this.log(`❌ Error in loadData: ${error.message}`);
            this.updateStatus('❌ Failed to load data');
            return false;
        }
    }

    async trainModel() {
        if (!this.normalizedData) {
            this.log('❌ No data available for training');
            return false;
        }

        this.updateStatus('<span class="loading"></span>Training GRU and LSTM models...');
        this.log('🧠 Beginning model training with 3 epochs...');
        
        try {
            await this.model.trainModel(
                this.normalizedData.train, 
                this.normalizedData.test,
                3,  // epochs
                16   // batch size
            );
            
            this.updateStatus('✅ Model training completed');
            this.log('🎉 Training finished! Models are ready for evaluation.');
            return true;
        } catch (error) {
            this.log(`❌ Error training model: ${error.message}`);
            this.updateStatus('❌ Model training failed');
            return false;
        }
    }

    async evaluateModel() {
        if (!this.model.model) {
            this.log('❌ No trained model available for evaluation');
            return;
        }

        this.updateStatus('<span class="loading"></span>Evaluating model performance...');
        
        try {
            const metrics = await this.model.evaluateModel(this.normalizedData.test);
            
            this.log(`📈 Final GRU Metrics - Accuracy: ${metrics.gru.accuracy.toFixed(4)}`);
            this.log(`📈 Final LSTM Metrics - Accuracy: ${metrics.lstm.accuracy.toFixed(4)}`);
            
            this.updateStatus('✅ Model evaluation completed');
            return metrics;
        } catch (error) {
            this.log(`❌ Error evaluating model: ${error.message}`);
            this.updateStatus('❌ Model evaluation failed');
        }
    }

    async saveModel() {
        if (!this.model.model) {
            this.log('❌ No model to save');
            return;
        }

        this.updateStatus('<span class="loading"></span>Saving model...');
        try {
            await this.model.saveModel();
            this.updateStatus('✅ Model saved successfully');
        } catch (error) {
            this.log(`❌ Error saving model: ${error.message}`);
        }
    }

    resetApp() {
        this.log('🔄 Resetting application...');
        
        if (this.model) {
            this.model.dispose();
        }
        
        // Clear all data
        this.dataLoader = new DataLoader();
        this.model = new GRUModel();
        this.normalizedData = null;
        
        // Clear UI
        document.getElementById('consoleLog').innerHTML = '';
        document.getElementById('status').textContent = 'Application reset. Click "Start Process" to begin.';
        
        // Reset metrics
        document.getElementById('gruAccuracy').textContent = '-';
        document.getElementById('gruLoss').textContent = '-';
        document.getElementById('gruPrecision').textContent = '-';
        document.getElementById('gruRecall').textContent = '-';
        document.getElementById('lstmAccuracy').textContent = '-';
        document.getElementById('lstmLoss').textContent = '-';
        document.getElementById('lstmPrecision').textContent = '-';
        document.getElementById('lstmRecall').textContent = '-';
        
        document.getElementById('predictionsContainer').innerHTML = '<p>Predictions will appear here after evaluation...</p>';
        
        // Reset buttons
        this.setButtonState('startBtn', true);
        this.setButtonState('loadDataBtn', false);
        this.setButtonState('trainBtn', false);
        this.setButtonState('evaluateBtn', false);
        this.setButtonState('saveModelBtn', false);
        
        this.log('✅ Application reset complete. Ready to start again.');
    }
}

// Initialize application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.app = new StartupRiskApp();
});
