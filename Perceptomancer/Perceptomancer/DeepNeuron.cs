using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Perceptomancer
{
    public class DeepNeuron
    {
        public double[] Weights;
        public double LastActivation;
        public double Bias;
        public IActivationFunction ActivationFunction;

        public DeepNeuron(int numberOfInputs, IActivationFunction activationFunction, Random r)
        {
            ActivationFunction = activationFunction;
            Bias = 10 * r.NextDouble() - 5;
            Weights = new double[numberOfInputs];
            for (int i = 0; i < numberOfInputs; i++)
            {
                Weights[i] = 10 * r.NextDouble() - 5;
            }
        }

        public double Activate(double[] inputs)
        {
            double activation = Bias;

            for (int i = 0; i < Weights.Length; i++)
            {
                activation += Weights[i] * inputs[i];
            }

            LastActivation = activation;
            return ActivationFunction.Activate(activation);
        }
    }
}
