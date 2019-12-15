﻿using System;
using System.IO;
using System.Linq;
using CNTK;
using CNTKUtil;
using XPlot.Plotly;
using Microsoft.ML;
using Microsoft.ML.Data;
using System.Diagnostics;
using System.Text;

namespace HousePricePrediction
{
    /// <summary>
    /// The HouseBlockData class holds one single housing block data record.
    /// </summary>
    public class HouseBlockData
    {
        [LoadColumn(0)] public float Longitude { get; set; }
        [LoadColumn(1)] public float Latitude { get; set; }
        [LoadColumn(2)] public float HousingMedianAge { get; set; }
        [LoadColumn(3)] public float TotalRooms { get; set; }
        [LoadColumn(4)] public float TotalBedrooms { get; set; }
        [LoadColumn(5)] public float Population { get; set; }
        [LoadColumn(6)] public float Households { get; set; }
        [LoadColumn(7)] public float MedianIncome { get; set; }
        [LoadColumn(8)] public float MedianHouseValue { get; set; }

        public float[] GetFeatures()
        {
            var latLonCross = Latitude * Longitude;

            return new float[] { Longitude, Latitude, HousingMedianAge, TotalRooms, TotalBedrooms, Population, Households, MedianIncome };
        }

        public float GetLabel() => MedianHouseValue / 1000.0f;
    }

    class Program
    {
        /// <summary>
        /// The main entry point of the application.
        /// </summary>
        /// <param name="args">The command line arguments.</param>
        [STAThread]
        public static void Main(string[] args)
        {
            var run = DateTime.Now;
            var runId = $"HousePricePrediction_{run.ToString("yyyyddMM_HHmmss")}";

            using (var logger = new ConsoleLogger(@"..\..\..\..\Results", runId))
            {
                logger.WriteLine("===============================================================================================");
                logger.WriteLine($"Assignment: California House Price Prediction ({DateTime.Now.ToShortDateString()} - {DateTime.Now.ToLongTimeString()} => {runId})");
                logger.WriteLine("===============================================================================================");
                logger.WriteLine("");

                // create the machine learning context
                var context = new MLContext();

                // check the current device for running neural networks
                logger.WriteLine($"Using device: {NetUtil.CurrentDevice.AsString()}");

                // load the dataset
                logger.WriteLine("Loading data...");
                var data = context.Data.LoadFromTextFile<HouseBlockData>(
                    path: "california_housing.csv",
                    hasHeader: true,
                    separatorChar: ',');

                // split into training and testing partitions
                var partitions = context.Data.TrainTestSplit(data, 0.2);

                // load training and testing data
                var training = context.Data.CreateEnumerable<HouseBlockData>(partitions.TrainSet, reuseRowObject: false);
                var testing = context.Data.CreateEnumerable<HouseBlockData>(partitions.TestSet, reuseRowObject: false);

                // set up data arrays
                var training_data = training.Select(v => v.GetFeatures()).ToArray();
                var training_labels = training.Select(v => v.GetLabel()).ToArray();
                var testing_data = testing.Select(v => v.GetFeatures()).ToArray();
                var testing_labels = testing.Select(v => v.GetLabel()).ToArray();

                // build features and labels
                var features = NetUtil.Var(new int[] { 8 }, DataType.Float);
                var labels = NetUtil.Var(new int[] { 1 }, DataType.Float);

                // build the network
                var network = features
                    .Dense(8, CNTKLib.ReLU)
                    .Dense(8, CNTKLib.ReLU)
                    .Dense(1)
                    .ToNetwork();
                logger.WriteLine("Model architecture:");
                logger.WriteLine(network.ToSummary());

                // set up the loss function and the classification error function
                var lossFunc = NetUtil.MeanSquaredError(network.Output, labels);
                var errorFunc = NetUtil.MeanAbsoluteError(network.Output, labels);

                // set up a trainer that uses the RMSProp algorithm
                var learner = network.GetAdamLearner(
                    learningRateSchedule: (0.001, 1),
                    momentumSchedule: (0.9, 1),
                    unitGain: false);

                // set up a trainer and an evaluator
                var trainer = network.GetTrainer(learner, lossFunc, errorFunc);
                var evaluator = network.GetEvaluator(errorFunc);

                // train the model
                logger.WriteLine("     \t     Train\t     Train\t      Test");
                logger.WriteLine("Epoch\t      Loss\t     Error\t     Error");
                logger.WriteLine("-----\t----------\t----------\t----------");

                var sw = Stopwatch.StartNew();
                var maxEpochs = 50;
                var batchSize = 16;
                var loss = new double[maxEpochs];
                var trainingError = new double[maxEpochs];
                var testingError = new double[maxEpochs];
                var batchCount = 0;
                for (int epoch = 0; epoch < maxEpochs; epoch++)
                {
                    // train one epoch on batches
                    loss[epoch] = 0.0;
                    trainingError[epoch] = 0.0;
                    batchCount = 0;
                    training_data.Index().Shuffle().Batch(batchSize, (indices, begin, end) =>
                    {
                        // get the current batch
                        var featureBatch = features.GetBatch(training_data, indices, begin, end);
                        var labelBatch = labels.GetBatch(training_labels, indices, begin, end);

                        // train the network on the batch
                        var result = trainer.TrainBatch(
                                new[] {
                            (features, featureBatch),
                            (labels,  labelBatch)
                                },
                                false
                            );
                        loss[epoch] += result.Loss;
                        trainingError[epoch] += result.Evaluation;
                        batchCount++;
                    });

                    // show results
                    loss[epoch] /= batchCount;
                    trainingError[epoch] /= batchCount;
                    logger.Write($"{epoch,5}\t{loss[epoch].Output10()}\t{trainingError[epoch].Output10()}\t");

                    // test one epoch on batches
                    testingError[epoch] = 0.0;
                    batchCount = 0;
                    testing_data.Batch(batchSize, (data, begin, end) =>
                    {
                        // get the current batch for testing
                        var featureBatch = features.GetBatch(testing_data, begin, end);
                        var labelBatch = labels.GetBatch(testing_labels, begin, end);

                        // test the network on the batch
                        testingError[epoch] += evaluator.TestBatch(
                                new[] {
                            (features, featureBatch),
                            (labels,  labelBatch)
                                }
                            );
                        batchCount++;
                    });
                    testingError[epoch] /= batchCount;
                    logger.WriteLine($"{testingError[epoch].Output10()}");
                }

                sw.Stop();
                logger.WriteLine($"model training done (epochs={maxEpochs}, batchSize={batchSize}): training time: {sw.Elapsed}");
                logger.WriteLine("");

                // show final results
                var finalError = testingError[maxEpochs - 1];
                logger.WriteLine("");
                logger.WriteLine($"Final test MAE: {finalError:0.00}");

                // plot the error graph
                var chart = Chart.Plot(
                    new[]
                    {
                    new Graph.Scatter()
                    {
                        x = Enumerable.Range(0, maxEpochs).ToArray(),
                        y = trainingError,
                        name = "training",
                        mode = "lines+markers"
                    },
                    new Graph.Scatter()
                    {
                        x = Enumerable.Range(0, maxEpochs).ToArray(),
                        y = testingError,
                        name = "testing",
                        mode = "lines+markers"
                    }
                    }
                );
                chart.WithXTitle("Epoch");
                chart.WithYTitle("Mean absolute error (MAE)");
                chart.WithTitle("California House Training");

                // save chart
                var htmlFilePath = logger.BuildFilePath($"{runId}_chart.html", deleteIfExists: true);
                File.WriteAllText(htmlFilePath, chart.GetHtml(), Encoding.UTF8);
            }
        }
    }
}
