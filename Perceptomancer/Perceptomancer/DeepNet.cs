using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Perceptomancer
{
    [Serializable]
    public class DeepNet
    {
        private readonly List<DeepLayer> _layers = new List<DeepLayer>();

        private List<double[]> sigmas;
        private List<double[,]> deltas;

        public void AddLayer(DeepLayer layer)
        {
            layer.Attach(_layers.Count == 0 ?
                null : _layers[_layers.Count - 1]);
            _layers.Add(layer);
        }

        public double[] Feed(double[] inputValues)
        {
            double[] tmpValues = inputValues;
            foreach(var layer in _layers)
                tmpValues = layer.Feed(tmpValues);
            return tmpValues;
        }

        public bool Train(TraningSet traningSet, double learningRate, double maxError, int maxIterations)
        {
            double err;
            int iteration = maxIterations;
            while (true)
            {
                ApplyBackPropagation(traningSet.Input, traningSet.Output, learningRate);
                err = GeneralError(traningSet.Input, traningSet.Output);

                bool writeError = (iteration - maxIterations) % 1000 == 0;
                maxIterations--;
                bool cancel = maxIterations <= 0;
                bool complete = err <= maxError;

                if (writeError)
                    Debug.WriteLine($"Error: {err}, Iterations: {(iteration - maxIterations)}");
                if (complete)
                    return true;
                if (cancel)
                    return false;
            }
        }

        private double IndividualError(double[] actualOutput, double[] desiredOutput)
        {
            double err = 0;
            for (int i = 0; i < actualOutput.Length; i++)
                err += Math.Pow(actualOutput[i] - desiredOutput[i], 2);
            return err;
        }

        private double GeneralError(List<double[]> inputValues, List<double[]> desiredOutput)
        {
            double err = 0;
            for (int i = 0; i < inputValues.Count; i++)
                err += IndividualError(Feed(inputValues[i]), desiredOutput[i]);
            return err;
        }

        private void ApplyBackPropagation(List<double[]> input, List<double[]> desiredOutput, double learningRate)
        {
            SetDeltas();
            for (int i = 0; i < input.Count; i++)
            {
                Feed(input[i]);
                SetSigmas(desiredOutput[i]);
                UpdateBias(learningRate);
                AddDelta();
            }
            UpdateWeights(learningRate);

        }

        private void SetSigmas(double[] desiredOutput)
        {
            sigmas = new List<double[]>();
            for (int i = 0; i < _layers.Count; i++)
            {
                sigmas.Add(new double[_layers[i].NumberOfNeurons]);
            }
            for (int i = _layers.Count - 1; i >= 1; i--)
            {
                for (int j = 0; j < _layers[i].NumberOfNeurons; j++)
                {
                    if (i == _layers.Count - 1)
                    {
                        double y = _layers[i].Neurons[j].LastActivation;
                        sigmas[i][j] = (_layers[i].ActivationFunction.Activate(y) - desiredOutput[j]) * _layers[i].ActivationFunction.Derive(y);
                    }
                    else
                    {
                        double sum = 0;
                        for (int k = 0; k < _layers[i + 1].NumberOfNeurons; k++)
                        {
                            sum += _layers[i + 1].Neurons[k].Weights[j] * sigmas[i + 1][k];
                        }
                        sigmas[i][j] = _layers[i].ActivationFunction.Derive(_layers[i].Neurons[j].LastActivation) * sum;
                    }
                }
            }
        }

        private void SetDeltas()
        {
            deltas = new List<double[,]>();
            for (int i = 0; i < _layers.Count; i++)
            {
                deltas.Add(new double[_layers[i].NumberOfNeurons, _layers[i].Neurons[0].Weights.Length]);
            }
        }

        private void UpdateBias(double learningRate)
        {
            for (int i = 0; i < _layers.Count; i++)
            {
                for (int j = 0; j < _layers[i].NumberOfNeurons; j++)
                {
                    _layers[i].Neurons[j].Bias -= learningRate * sigmas[i][j];
                }
            }
        }

        private void AddDelta()
        {
            for (int i = 1; i < _layers.Count; i++)
            {
                for (int j = 0; j < _layers[i].NumberOfNeurons; j++)
                {
                    for (int k = 0; k < _layers[i].Neurons[j].Weights.Length; k++)
                    {
                        deltas[i][j, k] += sigmas[i][j] * _layers[i].ActivationFunction.Activate(_layers[i - 1].Neurons[k].LastActivation);
                    }
                }
            }
        }

        void UpdateWeights(double learningRate)
        {
            for (int i = 0; i < _layers.Count; i++)
            {
                for (int j = 0; j < _layers[i].NumberOfNeurons; j++)
                {
                    for (int k = 0; k < _layers[i].Neurons[j].Weights.Length; k++)
                    {
                        _layers[i].Neurons[j].Weights[k] -= learningRate * deltas[i][j, k];
                    }
                }
            }
        }
    }
}
