using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Formatters.Binary;

namespace Perceptomancer
{
    [Serializable]
    public class Perceptron
    {
        List<Layer> layers;

        public Perceptron(int[] neuronsPerlayer, int randomizationSeed = 1)
        {
            layers = new List<Layer>();
            Random r = new Random(randomizationSeed);

            for (int i = 0; i < neuronsPerlayer.Length; i++)
            {
                layers.Add(new Layer(neuronsPerlayer[i], i == 0 ? neuronsPerlayer[i] : neuronsPerlayer[i - 1], r));
            }
        }
        public double[] Activate(double[] inputs)
        {
            double[] outputs = new double[0];
            for (int i = 1; i < layers.Count; i++)
            {
                outputs = layers[i].Activate(inputs);
                inputs = outputs;
            }
            return outputs;
        }
        double IndividualError(double[] realOutput, double[] desiredOutput)
        {
            double err = 0;
            for (int i = 0; i < realOutput.Length; i++)
            {
                err += Math.Pow(realOutput[i] - desiredOutput[i], 2);
            }
            return err;
        }
        double GeneralError(List<double[]> input, List<double[]> desiredOutput)
        {
            double err = 0;
            for (int i = 0; i < input.Count; i++)
            {
                err += IndividualError(Activate(input[i]), desiredOutput[i]);
            }
            return err;
        }
        List<string> log;
        public bool Learn(List<double[]> input, List<double[]> desiredOutput, double alpha, double maxError, int maxIterations, String net_path = null, int iter_save = 1)
        {
            double err = 99999;
            log = new List<string>();
            int it = maxIterations;
            while (err > maxError)
            {
                ApplyBackPropagation(input, desiredOutput, alpha);
                err = GeneralError(input, desiredOutput);


                if ((it - maxIterations) % 1000 == 0)
                {
                    Console.WriteLine(err + " iterations: " + (it - maxIterations));
                }


                if (net_path != null)
                {
                    if ((it - maxIterations) % iter_save == 0)
                    {
                        save_net(net_path);
                        Console.WriteLine("Save net to " + net_path);
                    }
                }

                log.Add(err.ToString());
                maxIterations--;

                if (Console.KeyAvailable)
                {
                    System.IO.File.WriteAllLines(@"LogTail.txt", log.ToArray());
                    return true;
                }

                if (maxIterations <= 0)
                {
                    Console.WriteLine("MINIMO LOCAL");
                    System.IO.File.WriteAllLines(@"LogTail.txt", log.ToArray());
                    return false;
                }

            }

            System.IO.File.WriteAllLines(@"LogTail.txt", log.ToArray());
            return true;
        }

        List<double[]> sigmas;
        List<double[,]> deltas;

        void SetSigmas(double[] desiredOutput)
        {
            sigmas = new List<double[]>();
            for (int i = 0; i < layers.Count; i++)
            {
                sigmas.Add(new double[layers[i].numberOfNeurons]);
            }
            for (int i = layers.Count - 1; i >= 0; i--)
            {
                for (int j = 0; j < layers[i].numberOfNeurons; j++)
                {
                    if (i == layers.Count - 1)
                    {
                        double y = layers[i].neurons[j].lastActivation;
                        sigmas[i][j] = (Neuron.Sigmoid(y) - desiredOutput[j]) * Neuron.SigmoidDerivated(y);
                    }
                    else
                    {
                        double sum = 0;
                        for (int k = 0; k < layers[i + 1].numberOfNeurons; k++)
                        {
                            sum += layers[i + 1].neurons[k].weights[j] * sigmas[i + 1][k];
                        }
                        sigmas[i][j] = Neuron.SigmoidDerivated(layers[i].neurons[j].lastActivation) * sum;
                    }
                }
            }
        }
        void SetDeltas()
        {
            deltas = new List<double[,]>();
            for (int i = 0; i < layers.Count; i++)
            {
                deltas.Add(new double[layers[i].numberOfNeurons, layers[i].neurons[0].weights.Length]);
            }
        }
        void AddDelta()
        {
            for (int i = 1; i < layers.Count; i++)
            {
                for (int j = 0; j < layers[i].numberOfNeurons; j++)
                {
                    for (int k = 0; k < layers[i].neurons[j].weights.Length; k++)
                    {
                        deltas[i][j, k] += sigmas[i][j] * Neuron.Sigmoid(layers[i - 1].neurons[k].lastActivation);
                    }
                }
            }
        }
        void UpdateBias(double alpha)
        {
            for (int i = 0; i < layers.Count; i++)
            {
                for (int j = 0; j < layers[i].numberOfNeurons; j++)
                {
                    layers[i].neurons[j].bias -= alpha * sigmas[i][j];
                }
            }
        }
        void UpdateWeights(double alpha)
        {
            for (int i = 0; i < layers.Count; i++)
            {
                for (int j = 0; j < layers[i].numberOfNeurons; j++)
                {
                    for (int k = 0; k < layers[i].neurons[j].weights.Length; k++)
                    {
                        layers[i].neurons[j].weights[k] -= alpha * deltas[i][j, k];
                    }
                }
            }
        }
        void ApplyBackPropagation(List<double[]> input, List<double[]> desiredOutput, double alpha)
        {
            SetDeltas();
            for (int i = 0; i < input.Count; i++)
            {
                Activate(input[i]);
                SetSigmas(desiredOutput[i]);
                UpdateBias(alpha);
                AddDelta();
            }
            UpdateWeights(alpha);

        }

        public void save_net(String neuralNetworkPath)
        {
            FileStream fs = new FileStream(neuralNetworkPath, FileMode.Create);
            BinaryFormatter formatter = new BinaryFormatter();
            try
            {
                formatter.Serialize(fs, this);
            }
            catch (SerializationException e)
            {
                Console.WriteLine("Failed to serialize. Reason: " + e.Message);
                throw;
            }
            finally
            {
                fs.Close();
            }
        }

        public static Perceptron Load(String neuralNetworkPath)
        {
            FileStream fs = new FileStream(neuralNetworkPath, FileMode.Open);
            Perceptron p = null;
            try
            {
                BinaryFormatter formatter = new BinaryFormatter();

                // Deserialize the hashtable from the file and 
                // assign the reference to the local variable.
                p = (Perceptron)formatter.Deserialize(fs);
            }
            catch (SerializationException e)
            {
                Console.WriteLine("Failed to deserialize. Reason: " + e.Message);
                throw;
            }
            finally
            {
                fs.Close();
            }

            return p;
        }
    }

    [Serializable]
    public class Layer
    {
        public List<Neuron> neurons;
        public int numberOfNeurons;
        public double[] output;

        public Layer(int _numberOfNeurons, int numberOfInputs, Random r)
        {
            numberOfNeurons = _numberOfNeurons;
            neurons = new List<Neuron>();
            for (int i = 0; i < numberOfNeurons; i++)
            {
                neurons.Add(new Neuron(numberOfInputs, r));
            }
        }

        public double[] Activate(double[] inputs)
        {
            List<double> outputs = new List<double>();
            for (int i = 0; i < numberOfNeurons; i++)
            {
                outputs.Add(neurons[i].Activate(inputs));
            }
            output = outputs.ToArray();
            return outputs.ToArray();
        }

    }

    [Serializable]
    public class Neuron
    {
        public double[] weights;
        public double lastActivation;
        public double bias;

        public Neuron(int numberOfInputs, Random r)
        {
            bias = 10 * r.NextDouble() - 5;
            weights = new double[numberOfInputs];
            for (int i = 0; i < numberOfInputs; i++)
            {
                weights[i] = 10 * r.NextDouble() - 5;
            }
        }
        public double Activate(double[] inputs)
        {
            double activation = bias;

            for (int i = 0; i < weights.Length; i++)
            {
                activation += weights[i] * inputs[i];
            }

            lastActivation = activation;
            return Sigmoid(activation);
        }
        public static double Sigmoid(double input)
        {
            return 1 / (1 + Math.Exp(-input));
        }
        public static double SigmoidDerivated(double input)
        {
            double y = Sigmoid(input);
            return y * (1 - y);
        }

    }
}
